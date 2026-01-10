import orjson
from tqdm import tqdm
import os
import re
from heapq import heappush, heappop, nlargest
import threading
from typing import Callable, Optional, List, Dict, Set, Tuple, Any
import pandas as pd
import shutil
from collections import defaultdict
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

# --- Configuration ---
ROOT_PATH = "../data/pretrain_combined_subreddit_list" 
DATA_PATH_RAW = os.path.join(ROOT_PATH, "raw_filtered_data") 
DATA_PATH_GRAPH_OUTPUT = os.path.join(ROOT_PATH, "graph_data") 

KEEP_COUNT = 50
MIN_SCORE_EXISTING_SUBMISSION = 25
MIN_SCORE_CONTR_INIT = 1

SUBREDDIT_TAXONOMY_CSV = "../data/online_social_networks/graphs/combined_subreddit_counts.csv"
SUBREDDIT_COLUMN_NAME = "subreddit"

# --- Graph I/O Paths ---
COMMUNITY_NODES_PARQUET_PATH = os.path.join(DATA_PATH_GRAPH_OUTPUT, "unique_community_nodes.parquet") 
TEMP_DECOMPRESS_DIR = os.path.join(ROOT_PATH, "tmp_decompressed") 
TEMP_SUBMISSION_FRAGMENTS_DIR = os.path.join(ROOT_PATH, "tmp_submission_fragments") 
USER_COMMUNITY_ACTIVE_EDGES_PARQUET_PATH = os.path.join(DATA_PATH_GRAPH_OUTPUT, "user_community_active_edges.parquet")
USER_NODES_PARQUET_PATH = os.path.join(DATA_PATH_GRAPH_OUTPUT, "unique_user_nodes.parquet") 

# --- Global Variables, Schemas, and Helper Functions ---
subreddit_regex = re.compile(r'"subreddit":"([^"]+)"')
linkid_regex = re.compile(r'"link_id":"([^"]+)"')
allowed_subreddits_set: Set[str] = set() 
SUBREDDITS_TO_PROCESS_IN_THIS_RUN: Set[str] = set() 
user_community_active_edges_writer: Optional[pq.ParquetWriter] = None
USER_COMMUNITY_ACTIVE_EDGES_SCHEMA = pa.schema([
    ('user_id', pa.string()),
    ('community_id', pa.string())
])

def clean_text(text: str) -> str:
    if not isinstance(text, str): return ""
    text = re.sub(r'[^\w\s.,!?;:\'"()-]', '', text)
    text = re.sub(r'http\S+|www\S+', '[URL]', text)
    text = re.sub(r'u/\S+', '[USER]', text)
    text = re.sub(r'@[a-zA-Z0-9_]+', '[USER]', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_data(data: str, is_root: bool) -> Optional[Dict]:
    try:
        data_dict: dict = orjson.loads(data)
    except orjson.JSONDecodeError:
        return None
    if 'id' not in data_dict or data_dict.get('id') is None:
        return None
    
    author = data_dict.get("author")
    # Original logic to discard deleted/removed authors was here.
    # It is now deferred to the phase where author data is explicitly collected
    # to enforce the "discard row" policy before saving to the graph.

    if not is_root:  # Comment (RC files)
        link_id_raw = data_dict.get("link_id", "")
        link_id = link_id_raw[3:] if link_id_raw.startswith("t3_") else ""
        if not link_id: return None
        body = data_dict.get("body")
        cleaned_text_content = clean_text(body)
        return {
            "node_type": "comment", "subreddit": data_dict.get("subreddit"), "id": data_dict.get("id"),
            "parent_id": data_dict.get("parent_id"), "link_id": link_id, "score": data_dict.get("score"),
            "text_content": cleaned_text_content, "author": author, "created_utc": data_dict.get("created_utc"),
            "title": None, "preview": None,
        }
    else:  # Submission (RS files)
        preview = data_dict.get("preview", None)
        if preview and "images" in preview and preview["images"]:
            preview = preview["images"][0].get("source", {}).get("url")
        selftext = data_dict.get("selftext", "")
        title = data_dict.get("title", "")
        cleaned_title = clean_text(title)
        cleaned_selftext = clean_text(selftext)
        text_content = cleaned_selftext if cleaned_selftext else cleaned_title
        return {
            "node_type": "submission", "subreddit": data_dict.get("subreddit"), "id": data_dict.get("id"),
            "score": data_dict.get("score"), "text_content": text_content, "title": cleaned_title,
            "preview": preview, "author": author, "created_utc": data_dict.get("created_utc"),
            "link_id": None, "parent_id": None,
        }

def thread_decompress(index: int, file_path: str):
    output_path = os.path.join(TEMP_DECOMPRESS_DIR, f"tmp-{index}.jsonl")
    print(
        f"[DECOMPRESSION] Starting for '{os.path.basename(file_path)}' to '{os.path.basename(output_path)}'"
    )
    command_str = f"zstd -d --long=31 -o {output_path} {file_path}"
    status = os.system(command_str)
    if status != 0:
        print(
            f"[DECOMPRESSION] ERROR for '{os.path.basename(file_path)}'. Command exited with status: {status}"
        )
    else:
        print(f"[DECOMPRESSION] Finished for '{os.path.basename(file_path)}'")


def parallel_process(
    file_list: List[str],
    process_fn: Callable,
    aux_data: Optional[Dict] = None,
    collected_unique_authors: Optional[Set[str]] = None
) -> List[None]:
    files_with_indices = list(enumerate(sorted(file_list)))
    if not files_with_indices:
        print("No files provided for processing. Skipping this phase.")
        return []

    os.makedirs(TEMP_DECOMPRESS_DIR, exist_ok=True)
    print(f"Starting parallel processing for {len(files_with_indices)} files.")

    initial_idx, initial_file = files_with_indices[0]
    current_decompress_thread = threading.Thread(
        target=thread_decompress, args=(initial_idx, initial_file)
    )
    current_decompress_thread.start()
    print(
        f"[ORCHESTRATOR] Initiated decompression for the first file: '{os.path.basename(initial_file)}'"
    )

    results = [] 
    
    for (i, file), (j, next_file) in tqdm(
        zip(files_with_indices, files_with_indices[1:]),
        total=len(files_with_indices) - 1,
        desc=f"[ORCHESTRATOR] File processing progress",
        position=0,
    ):
        current_decompress_thread.join()
        decompressed_file_path = os.path.join(TEMP_DECOMPRESS_DIR, f"tmp-{i}.jsonl")

        current_decompress_thread = threading.Thread(
            target=thread_decompress, args=(j, next_file)
        )
        current_decompress_thread.start()

        if (os.path.exists(decompressed_file_path) and os.path.getsize(decompressed_file_path) > 0):
            file_result = process_fn(
                file, i, decompressed_file_path, aux_data, collected_unique_authors
            )
            os.remove(decompressed_file_path)
        else:
            print(
                f"[ORCHESTRATOR] Skipping processing of '{os.path.basename(file)}' (index {i}) as decompression failed or file is empty."
            )

    last_idx, last_file = files_with_indices[-1]
    current_decompress_thread.join()

    decompressed_file_path_last = os.path.join(TEMP_DECOMPRESS_DIR, f"tmp-{last_idx}.jsonl")
    
    if (os.path.exists(decompressed_file_path_last) and os.path.getsize(decompressed_file_path_last) > 0):
        file_result = process_fn(
            last_file, last_idx, decompressed_file_path_last, aux_data, collected_unique_authors
        )
        os.remove(decompressed_file_path_last)
    else:
        print(
            f"[ORCHESTRATOR] Skipping processing of '{os.path.basename(last_file)}' (index {last_idx}) as decompression failed or file is empty."
        )

    if os.path.exists(TEMP_DECOMPRESS_DIR) and not os.listdir(TEMP_DECOMPRESS_DIR):
        try:
            os.rmdir(TEMP_DECOMPRESS_DIR)
        except OSError as e:
            print(f"[ORCHESTRATOR] Could not remove temporary directory {TEMP_DECOMPRESS_DIR}: {e}")
    return results


# --- Submission Processing ---
def process_submissions_and_collect_fragment(
    original_file_path: str,
    index: int,
    decompressed_file_path: str,
    aux_data: Optional[Dict],
    collected_unique_authors: Set[str]
) -> None:
    global user_community_active_edges_writer
    local_heaps: Dict[str, List[Tuple]] = defaultdict(list)
    current_file_user_community_active_edges = []
    try:
        with open(decompressed_file_path, "r") as f:
            for line_idx, line in enumerate(f):
                subreddit_match = subreddit_regex.search(line)
                if subreddit_match is None: continue
                subreddit = subreddit_match.group(1).lower()
                if subreddit not in SUBREDDITS_TO_PROCESS_IN_THIS_RUN: continue
                data = extract_data(line, is_root=True)
                if data is None or not data.get("text_content"): continue
                score = data.get("score", 0)
                min_score = MIN_SCORE_CONTR_INIT 
                if not isinstance(score, (int, float)) or score < min_score: continue
                if data.get("subreddit", "").lower() != subreddit: continue
                heappush(local_heaps[subreddit], (score, data.get("created_utc", 0), data.get("id"), data))
                if len(local_heaps[subreddit]) > KEEP_COUNT: heappop(local_heaps[subreddit])
                author = data.get("author")
                # CRITICAL CHANGE 2/2: DISCARD if author is missing
                if author and author not in ["[deleted]", "[removed]", "null", None]:
                    current_file_user_community_active_edges.append({'user_id': author, 'community_id': subreddit})
                    collected_unique_authors.add(author)
            file_suffix = os.path.basename(original_file_path).replace('.zst', '.jsonl')
            for sub, heap in local_heaps.items():
                fragment_output_dir = os.path.join(TEMP_SUBMISSION_FRAGMENTS_DIR, sub)
                os.makedirs(fragment_output_dir, exist_ok=True)
                fragment_file_path = os.path.join(fragment_output_dir, f"top_{KEEP_COUNT}_{file_suffix}")
                with open(fragment_file_path, "ab") as f_frag:
                    for _, _, _, submission_data_dict in heap: 
                        f_frag.write(orjson.dumps(submission_data_dict, option=orjson.OPT_APPEND_NEWLINE))
    except FileNotFoundError:
        print(f"ERROR: Decompressed file not found: {decompressed_file_path}. Skipping.")
    except Exception as e:
        print(f"ERROR: Error processing {decompressed_file_path}: {e}. Skipping.")
    if current_file_user_community_active_edges:
        table = pa.Table.from_pylist(current_file_user_community_active_edges, schema=USER_COMMUNITY_ACTIVE_EDGES_SCHEMA)
        user_community_active_edges_writer.write_table(table)
    return None

# --- contr_init Logic ---
def apply_contr_init_filter(all_submissions: List[Dict[str, Any]], required_score: int, keep_count: int) -> List[Dict[str, Any]]:
    if not all_submissions: return []
    sorted_submissions = sorted(
        [(-s.get("score", 0), s.get("created_utc", 0), s) for s in all_submissions],
        key=lambda x: (x[0], x[1])
    )
    best_submission_data = sorted_submissions[0][2]
    final_selection = [best_submission_data]
    high_score_submissions = []
    for i in range(1, len(sorted_submissions)):
        submission_data = sorted_submissions[i][2]
        score = submission_data.get("score", 0)
        if score >= required_score: high_score_submissions.append(submission_data)
    remaining_slots = keep_count - 1
    final_selection.extend(high_score_submissions[:remaining_slots])
    return final_selection


# --- Comment Processing ---
def process_comments_and_collect(
    original_file_path: str,
    index: int,
    decompressed_file_path: str,
    aux_data: Dict[str, Set], 
    collected_unique_authors: Set[str]
) -> None:
    global user_community_active_edges_writer 
    ids_to_keep = aux_data['ids_to_keep']
    current_file_user_community_active_edges = []
    try:
        with open(decompressed_file_path, "r") as f:
            for line_idx, line in enumerate(f): 
                subreddit_match = subreddit_regex.search(line)
                if subreddit_match is None: continue
                subreddit = subreddit_match.group(1).lower()
                if subreddit not in SUBREDDITS_TO_PROCESS_IN_THIS_RUN or not ids_to_keep.get(subreddit): continue
                link_id_match = linkid_regex.search(line)
                if link_id_match is None: continue
                link_id = link_id_match.group(1)[3:]
                if link_id in ids_to_keep[subreddit]:
                    data = extract_data(line, is_root=False)
                    if data is None or not data.get("text_content"): continue
                    if data.get("score", 0) < 0: continue 
                    subreddit_output_dir = os.path.join(DATA_PATH_RAW, subreddit)
                    os.makedirs(subreddit_output_dir, exist_ok=True)
                    with open(os.path.join(subreddit_output_dir, "RC.txt"), "ab") as f_rc:
                        f_rc.write(orjson.dumps(data, option=orjson.OPT_APPEND_NEWLINE))
                    author = data.get("author")
                    # CRITICAL CHANGE 2/2: DISCARD if author is missing
                    if author and author not in ["[deleted]", "[removed]", "null", None]:
                        current_file_user_community_active_edges.append({'user_id': author, 'community_id': subreddit})
                        collected_unique_authors.add(author)
    except FileNotFoundError:
        print(f"ERROR: Decompressed file not found: {decompressed_file_path}. Skipping.")
    except Exception as e:
        print(f"ERROR: Error processing {decompressed_file_path}: {e}. Skipping.")
    if current_file_user_community_active_edges:
        table = pa.Table.from_pylist(current_file_user_community_active_edges, schema=USER_COMMUNITY_ACTIVE_EDGES_SCHEMA)
        user_community_active_edges_writer.write_table(table)
    return None
# ---------------------------------------------------------------------------------------------------
# --- Main Function ---
# ---------------------------------------------------------------------------------------------------

def main():
    """Entry point to process the raw archives and save filtered data."""

    global allowed_subreddits_set
    global SUBREDDITS_TO_PROCESS_IN_THIS_RUN
    global user_community_active_edges_writer

    os.makedirs(DATA_PATH_GRAPH_OUTPUT, exist_ok=True) 

    # --- PHASE 0: Setup and Success-Based Filtering ---
    
    # 0.1 Load ALL target subreddits from CSV
    try:
        df_subreddits = pd.read_csv(SUBREDDIT_TAXONOMY_CSV)
        if SUBREDDIT_COLUMN_NAME not in df_subreddits.columns:
            raise ValueError(f"Column '{SUBREDDIT_COLUMN_NAME}' not found in {SUBREDDIT_TAXONOMY_CSV}")
        csv_subreddits_set = set(df_subreddits[SUBREDDIT_COLUMN_NAME].astype(str).str.lower().tolist())
        allowed_subreddits_set = csv_subreddits_set
        print(f"INFO: Loaded {len(csv_subreddits_set)} total subreddits from CSV.")
    except FileNotFoundError:
        print(f"ERROR: Subreddit taxonomy CSV file not found at {SUBREDDIT_TAXONOMY_CSV}. Exiting.")
        return
    except Exception as e:
        print(f"ERROR: Error loading subreddit taxonomy CSV file: {e}. Exiting.")
        return

    # 0.2 Load EXISTING SUCCESSES from the current run's Parquet file
    existing_success_set = set()
    if os.path.exists(COMMUNITY_NODES_PARQUET_PATH):
        try:
            parquet_table = pq.read_table(COMMUNITY_NODES_PARQUET_PATH, columns=['community_id'])
            existing_success_set = set(parquet_table['community_id'].to_pylist())
            print(f"INFO: Loaded {len(existing_success_set)} successfully processed subreddits from Parquet: {os.path.basename(COMMUNITY_NODES_PARQUET_PATH)}.")
            
            if os.path.exists(DATA_PATH_RAW):
                print(f"INFO: **PRESERVING** existing raw filtered data directory (Incremental mode): {DATA_PATH_RAW}")

        except Exception as e:
            print(f"WARNING: Error reading existing Parquet file. Treating all CSV subreddits as 'new'. Error: {e}")
    else:
        print(f"INFO: Existing community nodes Parquet not found. **STARTING FROM SCRATCH**.")
        if os.path.exists(DATA_PATH_RAW):
            print(f"INFO: Removing stale raw filtered data directory (Scratch run cleanup): {DATA_PATH_RAW}")
            shutil.rmtree(DATA_PATH_RAW)


    # Determine which communities are NEW OR FAILED PREVIOUSLY (in CSV, but NOT in success Parquet)
    SUBREDDITS_TO_PROCESS_IN_THIS_RUN = csv_subreddits_set - existing_success_set
    
    print(f"INFO: Found {len(SUBREDDITS_TO_PROCESS_IN_THIS_RUN):,} subreddits to check/re-check in this run.")
    print(f"INFO: Total subreddits in the CSV: {len(allowed_subreddits_set):,}")
    
    # --- ADDED BLOCK: Print the list of NEW subreddits ---
    if SUBREDDITS_TO_PROCESS_IN_THIS_RUN:
        print("\n✅ Subreddits being Processed (New or Retrying):")
        for i, sub in enumerate(sorted(list(SUBREDDITS_TO_PROCESS_IN_THIS_RUN))):
            print(f"  {i+1}. {sub}")
        print("-" * 35)
    # ---------------------------------------------------

    if not SUBREDDITS_TO_PROCESS_IN_THIS_RUN:
        print("\n*** ALL TARGET SUBREDDITS FOUND DATA FOR. SCRIPT FINISHED. ***")
        return
    
    # 0.3 Setup directories and clean up temporary files for THIS run
    for path in [DATA_PATH_RAW, TEMP_SUBMISSION_FRAGMENTS_DIR]:
        if path == TEMP_SUBMISSION_FRAGMENTS_DIR and os.path.exists(path):
             print(f"INFO: Removing existing temporary directory: {path}")
             shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
    
    # Clean up previous active edges file if it exists
    if os.path.exists(USER_COMMUNITY_ACTIVE_EDGES_PARQUET_PATH):
        os.remove(USER_COMMUNITY_ACTIVE_EDGES_PARQUET_PATH)

    # -------------------------------------------------------------------------

    all_unique_authors = set()

    RAW_ARCHIVES_BASE_PATH = "/mnt/DATA/reddit/"

    submission_files_to_process = [
        # os.path.join(RAW_ARCHIVES_BASE_PATH, "RS_2016-02.zst"),
        # os.path.join(RAW_ARCHIVES_BASE_PATH, "RS_2016-03.zst"),
        # os.path.join(RAW_ARCHIVES_BASE_PATH, "RS_2017-02.zst"),
        # os.path.join(RAW_ARCHIVES_BASE_PATH, "RS_2017-03.zst"),
        os.path.join(RAW_ARCHIVES_BASE_PATH, "RS_2021-08.zst"),
        os.path.join(RAW_ARCHIVES_BASE_PATH, "RS_2021-09.zst"),
        os.path.join(RAW_ARCHIVES_BASE_PATH, "RS_2021-10.zst"),
        os.path.join(RAW_ARCHIVES_BASE_PATH, "RS_2021-11.zst"),
        os.path.join(RAW_ARCHIVES_BASE_PATH, "RS_2021-12.zst"),
    ]

    comment_files_to_process = [
        # os.path.join(RAW_ARCHIVES_BASE_PATH, "RC_2016-02.zst"),
        # os.path.join(RAW_ARCHIVES_BASE_PATH, "RC_2016-03.zst"),
        # os.path.join(RAW_ARCHIVES_BASE_PATH, "RC_2017-02.zst"),
        # os.path.join(RAW_ARCHIVES_BASE_PATH, "RC_2017-03.zst"),
        os.path.join(RAW_ARCHIVES_BASE_PATH, "RC_2021-08.zst"),
        os.path.join(RAW_ARCHIVES_BASE_PATH, "RC_2021-09.zst"),
        os.path.join(RAW_ARCHIVES_BASE_PATH, "RC_2021-10.zst"),
        os.path.join(RAW_ARCHIVES_BASE_PATH, "RC_2021-11.zst"),
        os.path.join(RAW_ARCHIVES_BASE_PATH, "RC_2021-12.zst"),
    ]
    # --- END OF FILE DEFINITION ---

    # --- Initialize Parquet Writer ---
    try:
        user_community_active_edges_writer = pq.ParquetWriter(USER_COMMUNITY_ACTIVE_EDGES_PARQUET_PATH, USER_COMMUNITY_ACTIVE_EDGES_SCHEMA)
    except Exception as e:
        print(f"ERROR: Could not initialize Parquet writer: {e}. Exiting.")
        return


    print("\n--- PHASE 1.1: Processing Submission Archives (RS files) for targeted subreddits ---")
    
    _ = parallel_process(
        submission_files_to_process,
        process_submissions_and_collect_fragment, 
        aux_data={}, 
        collected_unique_authors=all_unique_authors
    )

    ids_to_keep = {sub: set() for sub in SUBREDDITS_TO_PROCESS_IN_THIS_RUN}

    # --- PHASE 1.2: Consolidating and Applying contr_init Logic ---
    
    print("\n--- PHASE 1.2: Consolidating top submissions from fragments and writing to POST.txt ---")
    
    subreddits_with_fragments = [d for d in os.listdir(TEMP_SUBMISSION_FRAGMENTS_DIR) if os.path.isdir(os.path.join(TEMP_SUBMISSION_FRAGMENTS_DIR, d))]

    if not subreddits_with_fragments:
        print("WARNING: No submission fragments found for targeted subreddits. Skipping POST.txt generation and comment processing.")
        if os.path.exists(TEMP_SUBMISSION_FRAGMENTS_DIR): shutil.rmtree(TEMP_SUBMISSION_FRAGMENTS_DIR)
        if user_community_active_edges_writer: user_community_active_edges_writer.close()
        print("\n\n*** FINAL CHECK: UPDATING SKIP LIST (No new data found in this run) ***")
        if existing_success_set:
            final_success_set = list(existing_success_set)
            community_nodes_df = pd.DataFrame(final_success_set, columns=['community_id'])
            community_nodes_schema = pa.schema([('community_id', pa.string())])
            pq.write_table(pa.Table.from_pandas(community_nodes_df, schema=community_nodes_schema), COMMUNITY_NODES_PARQUET_PATH)
            print(f"INFO: Parquet skip list updated with {len(final_success_set)} existing successful communities.")
        else:
            print("INFO: Parquet skip list remains empty as no data was found previously or in this run.")
        return 

    for sub in tqdm(sorted(subreddits_with_fragments), desc="Consolidating submissions from fragments"):
        all_submissions: List[Dict[str, Any]] = []
        subreddit_fragment_dir = os.path.join(TEMP_SUBMISSION_FRAGMENTS_DIR, sub)
        fragment_files = [os.path.join(subreddit_fragment_dir, f) for f in os.listdir(subreddit_fragment_dir) if f.endswith('.jsonl')]

        for frag_file in fragment_files:
            try:
                with open(frag_file, "rb") as f: 
                    for line in f:
                        all_submissions.append(orjson.loads(line))
            except Exception as e:
                print(f"WARNING: Could not load fragment file {frag_file}: {e}. Skipping.")
                continue
            
            os.remove(frag_file)

        final_top_submissions_for_sub = apply_contr_init_filter(
            all_submissions, 
            required_score=MIN_SCORE_EXISTING_SUBMISSION, 
            keep_count=KEEP_COUNT
        )

        subreddit_output_dir = os.path.join(DATA_PATH_RAW, sub)
        os.makedirs(subreddit_output_dir, exist_ok=True)
        
        # Only write POST.txt if we actually found submissions to keep
        if final_top_submissions_for_sub:
             post_file_path = os.path.join(subreddit_output_dir, "POST.txt")
             with open(post_file_path, "wb") as f_post: 
                 for data in final_top_submissions_for_sub:
                     f_post.write(orjson.dumps(data, option=orjson.OPT_APPEND_NEWLINE))
             for data in final_top_submissions_for_sub:
                 ids_to_keep[sub].add(data['id'])
        else:
             print(f"INFO: No final submissions found for {sub} (failed contr_init filter). Skipping POST.txt.")
    
    if os.path.exists(TEMP_SUBMISSION_FRAGMENTS_DIR):
        shutil.rmtree(TEMP_SUBMISSION_FRAGMENTS_DIR)


    print("\n--- PHASE 2: Processing Comment Archives (RC files) for targeted subreddits ---")
    
    _ = parallel_process(
        comment_files_to_process,
        process_comments_and_collect, 
        aux_data={'ids_to_keep': ids_to_keep},
        collected_unique_authors=all_unique_authors
    )

    # --- Finalize Outputs ---
    try:
        if user_community_active_edges_writer:
            user_community_active_edges_writer.close()
            print(f"User-Community Active edges finalized at {USER_COMMUNITY_ACTIVE_EDGES_PARQUET_PATH}.")
    except Exception as e:
        print(f"ERROR: Error closing user_community_active_edges_writer: {e}")

    if all_unique_authors:
        user_nodes_df = pd.DataFrame(list(all_unique_authors), columns=['user_id'])
        user_nodes_schema = pa.schema([('user_id', pa.string())])
        pq.write_table(pa.Table.from_pandas(user_nodes_df, schema=user_nodes_schema), USER_NODES_PARQUET_PATH)
        print(f"Unique user nodes saved to {USER_NODES_PARQUET_PATH}. Total: {len(user_nodes_df):,}")
    else:
        print("No unique user node data collected.")

    # ---------------------------------------------------------------------
    # --- FINAL STEP: UPDATE SUCCESS-BASED SKIP LIST ---
    # ---------------------------------------------------------------------
    print("\n*** FINAL STEP: UPDATING SUCCESS-BASED SKIP LIST ***")
    
    # 1. Determine which subreddits currently have data folders (Success Set)
    current_success_set = set(
        [d for d in os.listdir(DATA_PATH_RAW) 
         if os.path.isdir(os.path.join(DATA_PATH_RAW, d)) 
         and os.path.exists(os.path.join(DATA_PATH_RAW, d, "POST.txt"))] # Only count if POST.txt exists
    )

    # 2. Combine with the previously successful set to ensure old successes are not lost
    final_success_set = list(current_success_set.union(existing_success_set))
    
    # 3. Overwrite the Parquet file with the final Success Set
    if final_success_set:
        community_nodes_df = pd.DataFrame(final_success_set, columns=['community_id'])
        community_nodes_schema = pa.schema([('community_id', pa.string())])
        pq.write_table(pa.Table.from_pandas(community_nodes_df, schema=community_nodes_schema), COMMUNITY_NODES_PARQUET_PATH)
        print(f"INFO: Parquet skip list updated with {len(final_success_set)} total successful communities.")

    # 4. Check for subreddits in CSV that still need data
    subreddits_to_check_next = csv_subreddits_set - current_success_set
    print(f"INFO: {len(subreddits_to_check_next):,} subreddits still need data (they will be searched next run with new archives).")
    
    
    print("\n--- SCRIPT 1 COMPLETE ---")
    print(f"Raw filtered data saved to: {DATA_PATH_RAW}")
    print(f"Successful communities list updated at: {COMMUNITY_NODES_PARQUET_PATH}")


if __name__ == "__main__":
    main()