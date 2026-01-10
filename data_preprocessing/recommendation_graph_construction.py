import orjson
import os
import re
from typing import Optional, List, Dict, Set, Tuple
import pandas as pd
import shutil
from collections import defaultdict
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from combine_utils import (
    combine_nodes_to_tree,
    trim_and_get_size,
    get_flat_nodes_and_edges_from_trimmed_tree,
)

# --- Configuration ---
ROOT_PATH = "../data/recommendation_3" 
DATA_PATH_RAW = os.path.join(ROOT_PATH, "raw_filtered_data") 
DATA_PATH_GRAPH_OUTPUT = os.path.join(ROOT_PATH, "graph_data")

MAX_TREE_DEPTH = 5
TRIM_BRANCH_FACTOR = 2

# --- Inputs from Script 1 ---
# This is the "source of truth" for communities that have text
COMMUNITY_SUCCESS_LOG_PARQUET_PATH = os.path.join(DATA_PATH_GRAPH_OUTPUT, "community_archives_success_log.parquet")
USER_NODES_PARQUET_PATH = os.path.join(DATA_PATH_GRAPH_OUTPUT, "unique_user_nodes.parquet") 
USER_COMMUNITY_ACTIVE_EDGES_PARQUET_PATH = os.path.join(DATA_PATH_GRAPH_OUTPUT, "user_community_active_edges.parquet")

# --- Outputs of this script ---
TEXT_NODES_PARQUET_PATH = os.path.join(DATA_PATH_GRAPH_OUTPUT, "all_text_nodes.parquet")
TEXT_COMMUNITY_EDGES_PARQUET_PATH = os.path.join(DATA_PATH_GRAPH_OUTPUT, "text_community_edges.parquet")
TEXT_USER_EDGES_PARQUET_PATH = os.path.join(DATA_PATH_GRAPH_OUTPUT, "text_user_edges.parquet")
UNIQUE_COMMUNITY_NODES_PARQUET_PATH = os.path.join(DATA_PATH_GRAPH_OUTPUT, "unique_community_nodes.parquet")

# --- Global Parquet Writers ---
text_nodes_writer: Optional[pq.ParquetWriter] = None
text_community_edges_writer: Optional[pq.ParquetWriter] = None
text_user_edges_writer: Optional[pq.ParquetWriter] = None
community_nodes_writer: Optional[pq.ParquetWriter] = None


# --- Global Data Structures ---
unique_community_nodes: Set[str] = set()


# Define PyArrow Schemas
TEXT_NODES_SCHEMA = pa.schema([
    ('id', pa.string()),
    ('node_type', pa.string()),
    ('text_content', pa.string()),
    ('subreddit', pa.string()),
    ('author', pa.string()),
    ('title', pa.string()),
])

TEXT_COMMUNITY_EDGES_SCHEMA = pa.schema([
    ('source_id', pa.string()),
    ('target_id', pa.string()),
    ('edge_type', pa.string()) # e.g., 'posts_in'
])

TEXT_USER_EDGES_SCHEMA = pa.schema([
    ('source_id', pa.string()),
    ('target_id', pa.string()),
    ('edge_type', pa.string()) # e.g., 'posted_by' or 'replies_to'
])

UNIQUE_COMMUNITY_NODES_SCHEMA = pa.schema([
    ('id', pa.string()),
    ('node_type', pa.string()), # Will be 'community'
])

# Schema for user nodes (read/written in filtering)
USER_NODES_SCHEMA = pa.schema([
    ('user_id', pa.string())
])

# Schema for user-community edges (read/written in filtering)
USER_COMMUNITY_ACTIVE_EDGES_SCHEMA = pa.schema([
    ('user_id', pa.string()),
    ('community_id', pa.string())
])


# --- Helper Functions ---

def load_jsonl_data(file_path: str) -> List[Dict]:
    """Loads data from a JSONL file."""
    data = []
    if not os.path.exists(file_path):
        return data
    try:
        with open(file_path, "rb") as f:
            for line in f:
                try:
                    data.append(orjson.loads(line))
                except orjson.JSONDecodeError:
                    print(f"WARNING: Malformed JSON line in {file_path}")
                    continue
    except Exception as e:
        print(f"ERROR: Could not load data from {file_path}: {e}")
    return data


def process_subreddit_conversation_trees(
    subreddit: str,
    all_submissions_data: Dict[str, Dict], # Map of submission ID to its data
    comments_by_link_id: Dict[str, List[Dict]] # Map of submission ID to its comments
):
    """
    Processes all conversations for a given subreddit: builds, trims trees,
    and collects nodes and edges for Parquet writing, using the depth/branch-factor logic.
    """
    global text_nodes_writer, text_community_edges_writer, text_user_edges_writer
    # Note: unique_community_nodes is no longer added here

    current_subreddit_text_nodes = []
    current_subreddit_text_community_edges = []
    current_subreddit_text_user_edges = []

    # This print is now less useful as we loop through all subs, but fine to keep
    # print(f"\n--- Processing conversation trees for subreddit: {subreddit} ---")

    submissions_for_this_subreddit = {
        sub_id: data for sub_id, data in all_submissions_data.items()
        if data.get('subreddit', '').lower() == subreddit
    }


    for submission_id, submission_data in submissions_for_this_subreddit.items():
        raw_comments_list = comments_by_link_id.get(submission_id, [])

        # Combine submission and its comments into a single list for tree building
        all_nodes_for_tree = [submission_data.copy()] + [c.copy() for c in raw_comments_list]

        # Build the conversation tree
        conversation_tree_root = combine_nodes_to_tree(all_nodes_for_tree, max_depth=MAX_TREE_DEPTH)

        if conversation_tree_root:
            # Apply trimming logic (modifies the tree in-place)
            _ = trim_and_get_size(conversation_tree_root, max_trim_depth=MAX_TREE_DEPTH, trim_branch_factor=TRIM_BRANCH_FACTOR)
            
            # Extract flat list of nodes and edges from the trimmed tree
            trimmed_text_nodes_flat = []
            trimmed_text_community_edges_flat = []
            trimmed_text_user_edges_flat = []

            # Get the required keys dynamically from the schema
            required_keys = TEXT_NODES_SCHEMA.names
            
            get_flat_nodes_and_edges_from_trimmed_tree(
                conversation_tree_root,
                trimmed_text_nodes_flat,
                trimmed_text_community_edges_flat,
                trimmed_text_user_edges_flat,
                required_keys=required_keys
            )
            
            # Add to subreddit's overall lists
            current_subreddit_text_nodes.extend(trimmed_text_nodes_flat)
            current_subreddit_text_community_edges.extend(trimmed_text_community_edges_flat)
            current_subreddit_text_user_edges.extend(trimmed_text_user_edges_flat)
        else:
            # If tree building fails or no comments, still include the top submission
            required_keys = TEXT_NODES_SCHEMA.names
            
            # Dynamically select only the required fields
            node_data_for_parquet = {
                k: submission_data.get(k) for k in required_keys
            }
            current_subreddit_text_nodes.append(node_data_for_parquet)
            
            # Add Text -> Community edge
            if submission_data.get('subreddit'):
                current_subreddit_text_community_edges.append({
                    'source_id': submission_data['id'],
                    'target_id': submission_data['subreddit'].lower(),
                    'edge_type': 'posts_in'
                })

            # Add Text -> User edge
            if submission_data.get('author'):
                current_subreddit_text_user_edges.append({
                    'source_id': submission_data['id'],
                    'target_id': submission_data['author'],
                    'edge_type': 'posted_by'
                })

    # Write collected data for this subreddit as a chunk
    if current_subreddit_text_nodes:
        table = pa.Table.from_pylist(current_subreddit_text_nodes, schema=TEXT_NODES_SCHEMA)
        text_nodes_writer.write_table(table)
        # print(f"  --> Appended {len(current_subreddit_text_nodes)} text nodes for {subreddit}.")
    
    if current_subreddit_text_community_edges:
        table = pa.Table.from_pylist(current_subreddit_text_community_edges, schema=TEXT_COMMUNITY_EDGES_SCHEMA)
        text_community_edges_writer.write_table(table)
        # print(f"  --> Appended {len(current_subreddit_text_community_edges)} text-community edges for {subreddit}.")

    if current_subreddit_text_user_edges:
        table = pa.Table.from_pylist(current_subreddit_text_user_edges, schema=TEXT_USER_EDGES_SCHEMA)
        text_user_edges_writer.write_table(table)
        # print(f"  --> Appended {len(current_subreddit_text_user_edges)} text-user/reply edges for {subreddit}.")


def filter_graph_for_user_balance(graph_output_path: str):
    """
    Reads the generated graph files and filters the user nodes to balance
    users-with-text and users-without-text.
    
    Overwrites:
    - unique_user_nodes.parquet
    - user_community_active_edges.parquet
    """
    print("\n--- PHASE 3: Balancing User Nodes ---")

    # Define paths
    text_nodes_path = os.path.join(graph_output_path, "all_text_nodes.parquet")
    user_nodes_path = os.path.join(graph_output_path, "unique_user_nodes.parquet")
    user_comm_edges_path = os.path.join(graph_output_path, "user_community_active_edges.parquet")

    # Check if all required files exist
    for f_path in [text_nodes_path, user_nodes_path, user_comm_edges_path]:
        if not os.path.exists(f_path):
            print(f"WARNING: Cannot perform user balancing. Missing file: {os.path.basename(f_path)}")
            return
            
    # 1. Get users with text (from the trimmed text nodes)
    try:
        text_nodes_df = pq.read_table(text_nodes_path, columns=['author']).to_pandas()
        users_with_text = set(text_nodes_df['author'].dropna().unique())
        print(f"Found {len(users_with_text):,} users with text in the trimmed graph.")
    except Exception as e:
        print(f"ERROR: Could not read text nodes for balancing: {e}")
        return

    # 2. Get all users (from the original user node list)
    try:
        all_users_df = pq.read_table(user_nodes_path).to_pandas()
        all_users = set(all_users_df['user_id'])
        print(f"Found {len(all_users):,} total users in the graph.")
    except Exception as e:
        print(f"ERROR: Could not read user nodes for balancing: {e}")
        return

    # 3. Calculate user sets
    users_with_no_text = list(all_users - users_with_text)
    num_to_keep = len(users_with_text)
    num_no_text = len(users_with_no_text)
    
    print(f"Found {num_no_text:,} users with no text in the trimmed graph.")

    # 4. Sample users with no text to match the number of users with text
    if num_no_text > num_to_keep:
        print(f"Sampling {num_to_keep:,} users from the {num_no_text:,} users with no text.")
        sampled_users_to_keep = set(np.random.choice(users_with_no_text, size=num_to_keep, replace=False))
    else:
        print(f"Keeping all {num_no_text:,} users with no text (less than or equal to users with text).")
        sampled_users_to_keep = set(users_with_no_text)

    # 5. Create final set of all users to keep
    final_users_to_keep = users_with_text.union(sampled_users_to_keep)
    print(f"Final balanced user count: {len(final_users_to_keep):,}")

    # 6. Filter and overwrite unique_user_nodes.parquet
    try:
        final_user_nodes_df = all_users_df[all_users_df['user_id'].isin(final_users_to_keep)]
        pq.write_table(
            pa.Table.from_pandas(final_user_nodes_df, schema=USER_NODES_SCHEMA),
            user_nodes_path
        )
        print(f"Successfully overwrote {os.path.basename(user_nodes_path)} with {len(final_user_nodes_df):,} balanced users.")
    except Exception as e:
        print(f"ERROR: Failed to overwrite user nodes: {e}")
        return

    # 7. Filter and overwrite user_community_active_edges.parquet
    try:
        # Read the original edge file
        user_comm_edges_df = pq.read_table(user_comm_edges_path).to_pandas()
        
        # Filter edges where the 'user_id' is in our final set
        final_user_comm_edges_df = user_comm_edges_df[user_comm_edges_df['user_id'].isin(final_users_to_keep)]
        
        # Overwrite the file
        pq.write_table(
            pa.Table.from_pandas(final_user_comm_edges_df, schema=USER_COMMUNITY_ACTIVE_EDGES_SCHEMA),
            user_comm_edges_path
        )
        print(f"Successfully overwrote {os.path.basename(user_comm_edges_path)} with {len(final_user_comm_edges_df):,} balanced edges.")
    except Exception as e:
        print(f"ERROR: Failed to overwrite user-community edges: {e}")


# --- Main Function ---

def main():
    """Entry point to build and trim trees and generate graph Parquet files."""

    global text_nodes_writer, text_community_edges_writer, text_user_edges_writer, community_nodes_writer, unique_community_nodes

    if not os.path.isdir(DATA_PATH_GRAPH_OUTPUT):
        os.makedirs(DATA_PATH_GRAPH_OUTPUT, exist_ok=True)
        
    # Clean up old edge files to ensure a fresh run
    # We DO NOT clean user_nodes or user_community_edges, as the filter step needs them
    for file_path in [
        TEXT_NODES_PARQUET_PATH, 
        TEXT_COMMUNITY_EDGES_PARQUET_PATH, 
        TEXT_USER_EDGES_PARQUET_PATH,
        UNIQUE_COMMUNITY_NODES_PARQUET_PATH
    ]:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed existing file: {os.path.basename(file_path)}")

    # --- Initialize Parquet Writers ---
    try:
        text_nodes_writer = pq.ParquetWriter(TEXT_NODES_PARQUET_PATH, TEXT_NODES_SCHEMA)
        text_community_edges_writer = pq.ParquetWriter(TEXT_COMMUNITY_EDGES_PARQUET_PATH, TEXT_COMMUNITY_EDGES_SCHEMA)
        text_user_edges_writer = pq.ParquetWriter(TEXT_USER_EDGES_PARQUET_PATH, TEXT_USER_EDGES_SCHEMA)
        community_nodes_writer = pq.ParquetWriter(UNIQUE_COMMUNITY_NODES_PARQUET_PATH, UNIQUE_COMMUNITY_NODES_SCHEMA)
        print("INFO: Initialized Parquet writers for text_nodes, text_community_edges, text_user_edges, and unique_community_nodes.")
    except Exception as e:
        print(f"ERROR: Could not initialize Parquet writers: {e}. Exiting.")
        return

    # --- FIX R1: Read communities from success log, not directory listing ---
    if not os.path.exists(COMMUNITY_SUCCESS_LOG_PARQUET_PATH):
        print(f"ERROR: Community success log not found at {COMMUNITY_SUCCESS_LOG_PARQUET_PATH}. Cannot proceed.")
        return
        
    try:
        community_log_df = pq.read_table(COMMUNITY_SUCCESS_LOG_PARQUET_PATH).to_pandas()
        subreddits_to_process = list(community_log_df['community_id'])
    except Exception as e:
        print(f"ERROR: Could not read community success log: {e}. Exiting.")
        return

    if not subreddits_to_process:
        print(f"WARNING: No subreddits found in the success log. Exiting.")
        return

    print(f"INFO: Found {len(subreddits_to_process)} subreddits to process from success log.")

    # Process each subreddit
    for subreddit in tqdm(sorted(subreddits_to_process), desc="Overall subreddit processing"):
        # Add to set *before* processing. This is now safe because
        # the log file guarantees this sub has text.
        unique_community_nodes.add(subreddit)
        
        subreddit_path = os.path.join(DATA_PATH_RAW, subreddit)
        post_file = os.path.join(subreddit_path, "POST.txt")
        rc_file = os.path.join(subreddit_path, "RC.txt")

        # We can trust the log, but a sanity check is good.
        if not os.path.exists(post_file):
            print(f"WARNING: Logged subreddit {subreddit} has no POST.txt. Skipping.")
            continue

        submissions_list = load_jsonl_data(post_file)
        all_submissions_data = {s['id']: s for s in submissions_list}

        comments_list = load_jsonl_data(rc_file)
        comments_by_link_id = defaultdict(list)
        for comment in comments_list:
            if comment.get('link_id'):
                comments_by_link_id[comment['link_id']].append(comment)
        
        del submissions_list
        del comments_list

        process_subreddit_conversation_trees(
            subreddit,
            all_submissions_data,
            comments_by_link_id
        )

        del all_submissions_data
        del comments_by_link_id


    # --- Finalize Parquet Writers ---
    try:
        if text_nodes_writer:
            text_nodes_writer.close()
            print(f"\nText nodes finalized at {TEXT_NODES_PARQUET_PATH}.")
        if text_community_edges_writer:
            text_community_edges_writer.close()
            print(f"Text-Community edges finalized at {TEXT_COMMUNITY_EDGES_PARQUET_PATH}.")
        if text_user_edges_writer:
            text_user_edges_writer.close()
            print(f"Text-User/Reply edges finalized at {TEXT_USER_EDGES_PARQUET_PATH}.")
            
        if unique_community_nodes and community_nodes_writer:
            community_nodes_list = [
                {'id': subreddit_name, 'node_type': 'community'}
                for subreddit_name in sorted(list(unique_community_nodes))
            ]
            table = pa.Table.from_pylist(community_nodes_list, schema=UNIQUE_COMMUNITY_NODES_SCHEMA)
            community_nodes_writer.write_table(table)
            community_nodes_writer.close()
            print(f"Unique community nodes ({len(community_nodes_list)} total) finalized at {UNIQUE_COMMUNITY_NODES_PARQUET_PATH}.")

    except Exception as e:
        print(f"ERROR: Error closing Parquet writers: {e}")

    print("\n--- SCRIPT 2A COMPLETE (File Generation) ---")
    
    # --- NEW: Run final filtering step ---
    filter_graph_for_user_balance(DATA_PATH_GRAPH_OUTPUT)

    print("\n--- SCRIPT 2B COMPLETE (User Filtering) ---")
    print("Final graph data is now organized in Parquet files under:", DATA_PATH_GRAPH_OUTPUT)


if __name__ == "__main__":
    main()

# import orjson
# import os
# import re
# from typing import Optional, List, Dict, Set, Tuple, Any
# import pandas as pd
# import shutil
# from collections import defaultdict
# from tqdm import tqdm
# import pyarrow as pa
# import pyarrow.parquet as pq
# from combine_utils import (
#     combine_nodes_to_tree,
#     trim_and_get_size,
#     get_flat_nodes_and_edges_from_trimmed_tree,
# )

# # --- Configuration ---
# ROOT_PATH = "../data/recommendation_2" 
# DATA_PATH_RAW = os.path.join(ROOT_PATH, "raw_filtered_data") 
# DATA_PATH_GRAPH_OUTPUT = os.path.join(ROOT_PATH, "graph_data")

# MAX_TREE_DEPTH = 5
# TRIM_BRANCH_FACTOR = 2

# # Outputs of this script
# TEXT_NODES_PARQUET_PATH = os.path.join(DATA_PATH_GRAPH_OUTPUT, "all_text_nodes.parquet")
# TEXT_COMMUNITY_EDGES_PARQUET_PATH = os.path.join(DATA_PATH_GRAPH_OUTPUT, "text_community_edges.parquet")
# TEXT_USER_EDGES_PARQUET_PATH = os.path.join(DATA_PATH_GRAPH_OUTPUT, "text_user_edges.parquet")
# UNIQUE_COMMUNITY_NODES_PARQUET_PATH = os.path.join(DATA_PATH_GRAPH_OUTPUT, "unique_community_nodes.parquet")

# # --- Global Parquet Writers ---
# text_nodes_writer: Optional[pq.ParquetWriter] = None
# text_community_edges_writer: Optional[pq.ParquetWriter] = None
# text_user_edges_writer: Optional[pq.ParquetWriter] = None
# community_nodes_writer: Optional[pq.ParquetWriter] = None


# # --- Global Data Structures ---
# unique_community_nodes: Set[str] = set()


# # Define PyArrow Schemas
# TEXT_NODES_SCHEMA = pa.schema([
#     ('id', pa.string()),
#     ('node_type', pa.string()),
#     ('text_content', pa.string()),
#     ('subreddit', pa.string()),
#     ('author', pa.string()),
#     ('title', pa.string()),
# ])

# TEXT_COMMUNITY_EDGES_SCHEMA = pa.schema([
#     ('source_id', pa.string()),
#     ('target_id', pa.string()),
#     ('edge_type', pa.string())
# ])

# TEXT_USER_EDGES_SCHEMA = pa.schema([
#     ('source_id', pa.string()),
#     ('target_id', pa.string()),
#     ('edge_type', pa.string())
# ])

# UNIQUE_COMMUNITY_NODES_SCHEMA = pa.schema([
#     ('id', pa.string()),
#     ('node_type', pa.string()),
# ])


# # --- Helper Functions ---

# def load_jsonl_data(file_path: str) -> List[Dict]:
#     """Loads data from a JSONL file."""
#     data = []
#     if not os.path.exists(file_path):
#         return data
#     try:
#         with open(file_path, "rb") as f:
#             for line in f:
#                 try:
#                     data.append(orjson.loads(line))
#                 except orjson.JSONDecodeError:
#                     print(f"WARNING: Malformed JSON line in {file_path}")
#                     continue
#     except Exception as e:
#         print(f"ERROR: Could not load data from {file_path}: {e}")
#     return data

# def process_subreddit_conversation_trees(
#     subreddit: str,
#     all_submissions_data: Dict[str, Dict],
#     comments_by_link_id: Dict[str, List[Dict]]
# ):
#     """
#     Processes all conversations for a given subreddit: builds, trims trees,
#     and collects nodes and edges for Parquet writing.
#     """
#     global text_nodes_writer, text_community_edges_writer, text_user_edges_writer, unique_community_nodes
#     unique_community_nodes.add(subreddit)

#     current_subreddit_text_nodes = []
#     current_subreddit_text_community_edges = []
#     current_subreddit_text_user_edges = []

#     submissions_for_this_subreddit = {
#         sub_id: data for sub_id, data in all_submissions_data.items()
#         if data.get('subreddit', '').lower() == subreddit
#     }

#     for submission_id, submission_data in tqdm(
#         submissions_for_this_subreddit.items(),
#         desc=f"Building/Trimming trees for {subreddit}",
#         leave=False
#     ):
#         raw_comments_list = comments_by_link_id.get(submission_id, [])
#         all_nodes_for_tree = [submission_data.copy()] + [c.copy() for c in raw_comments_list]
#         conversation_tree_root = combine_nodes_to_tree(all_nodes_for_tree, max_depth=MAX_TREE_DEPTH)

#         if conversation_tree_root:
#             _ = trim_and_get_size(conversation_tree_root, max_trim_depth=MAX_TREE_DEPTH, trim_branch_factor=TRIM_BRANCH_FACTOR)
            
#             trimmed_text_nodes_flat, trimmed_text_community_edges_flat, trimmed_text_user_edges_flat = [], [], []
#             required_keys = TEXT_NODES_SCHEMA.names
            
#             get_flat_nodes_and_edges_from_trimmed_tree(
#                 conversation_tree_root,
#                 trimmed_text_nodes_flat,
#                 trimmed_text_community_edges_flat,
#                 trimmed_text_user_edges_flat,
#                 required_keys=required_keys
#             )
            
#             current_subreddit_text_nodes.extend(trimmed_text_nodes_flat)
#             current_subreddit_text_community_edges.extend(trimmed_text_community_edges_flat)
#             current_subreddit_text_user_edges.extend(trimmed_text_user_edges_flat)
#         else:
#             required_keys = TEXT_NODES_SCHEMA.names
#             node_data_for_parquet = {k: submission_data.get(k) for k in required_keys}
#             current_subreddit_text_nodes.append(node_data_for_parquet)
            
#             if submission_data.get('subreddit'):
#                 current_subreddit_text_community_edges.append({
#                     'source_id': submission_data['id'],
#                     'target_id': submission_data['subreddit'],
#                     'edge_type': 'posts_in'
#                 })
#             if submission_data.get('author'):
#                 current_subreddit_text_user_edges.append({
#                     'source_id': submission_data['id'],
#                     'target_id': submission_data['author'],
#                     'edge_type': 'posted_by'
#                 })

#     if current_subreddit_text_nodes:
#         text_nodes_writer.write_table(pa.Table.from_pylist(current_subreddit_text_nodes, schema=TEXT_NODES_SCHEMA))
#     if current_subreddit_text_community_edges:
#         text_community_edges_writer.write_table(pa.Table.from_pylist(current_subreddit_text_community_edges, schema=TEXT_COMMUNITY_EDGES_SCHEMA))
#     if current_subreddit_text_user_edges:
#         text_user_edges_writer.write_table(pa.Table.from_pylist(current_subreddit_text_user_edges, schema=TEXT_USER_EDGES_SCHEMA))


# # --- Main Function ---
# def main():
#     """Entry point to build and trim trees and generate graph Parquet files."""
#     global text_nodes_writer, text_community_edges_writer, text_user_edges_writer, community_nodes_writer

#     os.makedirs(DATA_PATH_GRAPH_OUTPUT, exist_ok=True)
        
#     for file_path in [TEXT_NODES_PARQUET_PATH, TEXT_COMMUNITY_EDGES_PARQUET_PATH, TEXT_USER_EDGES_PARQUET_PATH, UNIQUE_COMMUNITY_NODES_PARQUET_PATH]:
#         if os.path.exists(file_path):
#             os.remove(file_path)

#     try:
#         text_nodes_writer = pq.ParquetWriter(TEXT_NODES_PARQUET_PATH, TEXT_NODES_SCHEMA)
#         text_community_edges_writer = pq.ParquetWriter(TEXT_COMMUNITY_EDGES_PARQUET_PATH, TEXT_COMMUNITY_EDGES_SCHEMA)
#         text_user_edges_writer = pq.ParquetWriter(TEXT_USER_EDGES_PARQUET_PATH, TEXT_USER_EDGES_SCHEMA)
#         community_nodes_writer = pq.ParquetWriter(UNIQUE_COMMUNITY_NODES_PARQUET_PATH, UNIQUE_COMMUNITY_NODES_SCHEMA)
#     except Exception as e:
#         print(f"ERROR: Could not initialize Parquet writers: {e}. Exiting.")
#         return

#     subreddit_dirs = [d for d in os.listdir(DATA_PATH_RAW) if os.path.isdir(os.path.join(DATA_PATH_RAW, d))]
#     if not subreddit_dirs:
#         print(f"WARNING: No subreddit directories found in {DATA_PATH_RAW}. Exiting.")
#         return

#     print(f"INFO: Found {len(subreddit_dirs)} subreddits to process from {DATA_PATH_RAW}.")

#     for subreddit in tqdm(sorted(subreddit_dirs), desc="Overall subreddit processing"):
#         subreddit_path = os.path.join(DATA_PATH_RAW, subreddit)
#         post_file = os.path.join(subreddit_path, "POST.txt")
#         rc_file = os.path.join(subreddit_path, "RC.txt")

#         if not os.path.exists(post_file):
#             continue

#         submissions_list = load_jsonl_data(post_file)
        
#         # ### START: MODIFICATION TO FIX AUTHORS ###
#         for submission in submissions_list:
#             if submission.get('author') in ['[deleted]', '[removed]', 'null', None]:
#                 submission['author'] = '__missing__'
#         # ### END: MODIFICATION ###
        
#         all_submissions_data = {s['id']: s for s in submissions_list}

#         comments_list = load_jsonl_data(rc_file)
        
#         # ### START: MODIFICATION TO FIX AUTHORS ###
#         for comment in comments_list:
#             if comment.get('author') in ['[deleted]', '[removed]', 'null', None]:
#                 comment['author'] = '__missing__'
#         # ### END: MODIFICATION ###

#         comments_by_link_id = defaultdict(list)
#         for comment in comments_list:
#             if comment.get('link_id'):
#                 comments_by_link_id[comment['link_id']].append(comment)
        
#         del submissions_list, comments_list

#         process_subreddit_conversation_trees(
#             subreddit,
#             all_submissions_data,
#             comments_by_link_id
#         )

#         del all_submissions_data, comments_by_link_id

#     try:
#         if text_nodes_writer: text_nodes_writer.close()
#         if text_community_edges_writer: text_community_edges_writer.close()
#         if text_user_edges_writer: text_user_edges_writer.close()
            
#         if unique_community_nodes and community_nodes_writer:
#             community_nodes_list = [{'id': name, 'node_type': 'community'} for name in sorted(list(unique_community_nodes))]
#             community_nodes_writer.write_table(pa.Table.from_pylist(community_nodes_list, schema=UNIQUE_COMMUNITY_NODES_SCHEMA))
#             community_nodes_writer.close()
#     except Exception as e:
#         print(f"ERROR: Error closing Parquet writers: {e}")

#     print("\n--- SCRIPT 2 COMPLETE ---")

# if __name__ == "__main__":
#     main()