import orjson
import os
import re
from typing import Optional, List, Dict, Set, Tuple, Any
import pandas as pd
import shutil
from collections import defaultdict
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
from combine_utils import (
    combine_nodes_to_tree,
    trim_and_get_size,
    get_flat_nodes_and_edges_from_trimmed_tree,
)

# --- Configuration ---
ROOT_PATH = "../data/pretrain_2017_2_2021_6"
DATA_PATH_RAW = os.path.join(ROOT_PATH, "raw_filtered_data") 
DATA_PATH_GRAPH_OUTPUT = os.path.join(ROOT_PATH, "graph_data")

MAX_TREE_DEPTH = 5
TRIM_BRANCH_FACTOR = 2

# Outputs of this script
TEXT_NODES_PARQUET_PATH = os.path.join(DATA_PATH_GRAPH_OUTPUT, "all_text_nodes.parquet")
TEXT_COMMUNITY_EDGES_PARQUET_PATH = os.path.join(DATA_PATH_GRAPH_OUTPUT, "text_community_edges.parquet")
TEXT_USER_EDGES_PARQUET_PATH = os.path.join(DATA_PATH_GRAPH_OUTPUT, "text_user_edges.parquet")
# NEW: Path for unique community nodes
UNIQUE_COMMUNITY_NODES_PARQUET_PATH = os.path.join(DATA_PATH_GRAPH_OUTPUT, "unique_community_nodes.parquet")

# --- Global Parquet Writers ---
text_nodes_writer: Optional[pq.ParquetWriter] = None
text_community_edges_writer: Optional[pq.ParquetWriter] = None
text_user_edges_writer: Optional[pq.ParquetWriter] = None
# NEW: Writer for community nodes
community_nodes_writer: Optional[pq.ParquetWriter] = None


# --- Global Data Structures ---
# NEW: Set to collect all unique subreddit names (Community Nodes)
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

# NEW: Schema for Community Nodes
UNIQUE_COMMUNITY_NODES_SCHEMA = pa.schema([
    ('id', pa.string()),
    ('node_type', pa.string()), # Will be 'community'
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
    global text_nodes_writer, text_community_edges_writer, text_user_edges_writer, unique_community_nodes

    # NEW: Add the subreddit name to the global set immediately
    # Assuming the input 'subreddit' is the correct, lowercased name from the directory structure
    unique_community_nodes.add(subreddit)


    current_subreddit_text_nodes = []
    current_subreddit_text_community_edges = []
    current_subreddit_text_user_edges = []

    print(f"\n--- Processing conversation trees for subreddit: {subreddit} ---")

    submissions_for_this_subreddit = {
        sub_id: data for sub_id, data in all_submissions_data.items()
        if data.get('subreddit', '').lower() == subreddit
    }


    for submission_id, submission_data in tqdm(
        submissions_for_this_subreddit.items(),
        desc=f"Building/Trimming trees for {subreddit}"
    ):
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
                    'target_id': submission_data['subreddit'],
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
        print(f"  --> Appended {len(current_subreddit_text_nodes)} text nodes for {subreddit}.")
    
    if current_subreddit_text_community_edges:
        table = pa.Table.from_pylist(current_subreddit_text_community_edges, schema=TEXT_COMMUNITY_EDGES_SCHEMA)
        text_community_edges_writer.write_table(table)
        print(f"  --> Appended {len(current_subreddit_text_community_edges)} text-community edges for {subreddit}.")

    if current_subreddit_text_user_edges:
        table = pa.Table.from_pylist(current_subreddit_text_user_edges, schema=TEXT_USER_EDGES_SCHEMA)
        text_user_edges_writer.write_table(table)
        print(f"  --> Appended {len(current_subreddit_text_user_edges)} text-user/reply edges for {subreddit}.")


# --- Main Function ---

def main():
    """Entry point to build and trim trees and generate graph Parquet files."""

    global text_nodes_writer, text_community_edges_writer, text_user_edges_writer, community_nodes_writer

    if not os.path.isdir(DATA_PATH_GRAPH_OUTPUT):
        os.makedirs(DATA_PATH_GRAPH_OUTPUT, exist_ok=True)
        
    # Clean up old edge files to ensure a fresh run
    # UPDATED: Added UNIQUE_COMMUNITY_NODES_PARQUET_PATH to the cleanup list
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
        # NEW: Initialize writer for community nodes
        community_nodes_writer = pq.ParquetWriter(UNIQUE_COMMUNITY_NODES_PARQUET_PATH, UNIQUE_COMMUNITY_NODES_SCHEMA)
        print("INFO: Initialized Parquet writers for text_nodes, text_community_edges, text_user_edges, and unique_community_nodes.")
    except Exception as e:
        print(f"ERROR: Could not initialize Parquet writers: {e}. Exiting.")
        return

    # Discover subreddits by listing directories in DATA_PATH_RAW
    subreddit_dirs = [d for d in os.listdir(DATA_PATH_RAW) if os.path.isdir(os.path.join(DATA_PATH_RAW, d))]
    if not subreddit_dirs:
        print(f"WARNING: No subreddit directories found in {DATA_PATH_RAW}. Exiting.")
        return

    print(f"INFO: Found {len(subreddit_dirs)} subreddits to process from {DATA_PATH_RAW}.")

    # Process each subreddit
    for subreddit in tqdm(sorted(subreddit_dirs), desc="Overall subreddit processing"):
        subreddit_path = os.path.join(DATA_PATH_RAW, subreddit)
        post_file = os.path.join(subreddit_path, "POST.txt")
        rc_file = os.path.join(subreddit_path, "RC.txt")

        if not os.path.exists(post_file):
            print(f"WARNING: No POST.txt found for subreddit {subreddit}. Skipping.")
            continue

        submissions_list = load_jsonl_data(post_file)
        all_submissions_data = {s['id']: s for s in submissions_list}

        comments_list = load_jsonl_data(rc_file)
        comments_by_link_id = defaultdict(list)
        for comment in comments_list:
            # The link_id in the raw comment data from Script 1 is already stripped of 't3_'
            if comment.get('link_id'):
                comments_by_link_id[comment['link_id']].append(comment)
        
        # Free up memory
        del submissions_list
        del comments_list

        process_subreddit_conversation_trees(
            subreddit,
            all_submissions_data,
            comments_by_link_id
        )

        # Free up memory for the current subreddit
        del all_submissions_data
        del comments_by_link_id


    # --- Finalize Parquet Writers ---
    try:
        if text_nodes_writer:
            text_nodes_writer.close()
            print(f"Text nodes finalized at {TEXT_NODES_PARQUET_PATH}.")
        if text_community_edges_writer:
            text_community_edges_writer.close()
            print(f"Text-Community edges finalized at {TEXT_COMMUNITY_EDGES_PARQUET_PATH}.")
        if text_user_edges_writer:
            text_user_edges_writer.close()
            print(f"Text-User/Reply edges finalized at {TEXT_USER_EDGES_PARQUET_PATH}.")
            
        # NEW: Write and finalize unique community nodes
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

    print("\n--- SCRIPT 2 COMPLETE ---")
    print("Final graph data (trimmed text nodes and edges) is now organized in Parquet files under:", DATA_PATH_GRAPH_OUTPUT)


if __name__ == "__main__":
    main()