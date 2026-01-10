import os
import shutil
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import re
import numpy as np
from sklearn.model_selection import train_test_split 
from typing import List, Tuple, Set, Optional

# --- Configuration ---
# Set the root path for the Hateful dataset's graph output
HATEFUL_ROOT_PATH = "../data/hateful_2/"
# Retained original output folder naming for consistency
DATA_PATH_GRAPH_OUTPUT = os.path.join(HATEFUL_ROOT_PATH, "graph_data_split") 
TRAIN_DIR = os.path.join(DATA_PATH_GRAPH_OUTPUT, "train")
TEST_DIR = os.path.join(DATA_PATH_GRAPH_OUTPUT, "test")

# --- INPUT DATA PATH ---
HATEFUL_path = '../data/hateful/all_comments.parquet'

# Hyperparameter for split
TRAIN_SPLIT_RATIO = 0.8
RANDOM_SEED = 42 # For reproducible operations

# Standardized Output File Names
TEXT_NODES_FILENAME = "all_text_nodes.parquet"
USER_NODES_FILENAME = "unique_user_nodes.parquet"
COMMUNITY_NODES_FILENAME = "unique_community_nodes.parquet"
TEXT_COMMUNITY_EDGES_FILENAME = "text_community_edges.parquet"
TEXT_USER_EDGES_FILENAME = "text_user_edges.parquet"
USER_COMMUNITY_ACTIVE_EDGES_FILENAME = "user_community_active_edges.parquet"


# --- Standardized PyArrow Schemas (ADJUSTED FOR BINARY LABEL) ---

# Schemas for Nodes (Uses int64 label for binary 0/1 classification)
TEXT_NODES_SCHEMA = pa.schema([
    ('post_id', pa.string()),
    ('content', pa.string()),
    ('title', pa.string()),
    ('community_id', pa.string()),
    ('label', pa.int64()), # Hateful label is binary (0 or 1)
])

USER_NODES_SCHEMA = pa.schema([
    ('user_id', pa.string())
])

COMMUNITY_NODES_SCHEMA = pa.schema([
    ('community_id', pa.string())
])

# Schemas for Edges (Standardized: source_id, target_id, edge_type)
TEXT_COMMUNITY_EDGES_SCHEMA = pa.schema([
    ('source_id', pa.string()), # post_id
    ('target_id', pa.string()), # community_id
    ('edge_type', pa.string())  # 'posts_in'
])

TEXT_USER_EDGES_SCHEMA = pa.schema([
    ('source_id', pa.string()), # post_id
    ('target_id', pa.string()), # user_id
    ('edge_type', pa.string())  # 'posted_by'
])

USER_COMMUNITY_ACTIVE_EDGES_SCHEMA = pa.schema([
    ('source_id', pa.string()), # user_id
    ('target_id', pa.string()), # community_id
    ('edge_type', pa.string())  # 'active_in'
])

def map_to_binary_label(original_label: str) -> int:
    """
    Maps labels to 1 (Hate), 0 (No Hate), or -1 (Drop/Ignore).
    
    Returns:
        1:  Hate
        0:  Valid No Hate
        -1: Invalid/Unlabelled (NA, None, etc.)
    """
    # Explicit Hate Categories
    HATE_CATEGORIES = [
        'DEG', 'NDG', 'True', 'HOM', 'IdentityDirectedAbuse', 
        'AffiliationDirectedAbuse', 'APR', 'PersonDirectedAbuse', 
        'CMP', 'Slur'
    ]
    
    # Explicit Valid No-Hate Categories
    # 'False' implies the 'Hate' boolean check was False.
    VALID_NO_HATE_CATEGORIES = [
        'False', 'Neutral', 'CounterSpeech'
    ]
    
    # Convert to string to handle potential non-string types safely
    label_str = str(original_label)
    
    if label_str in HATE_CATEGORIES:
        return 1  # Hate
    elif label_str in VALID_NO_HATE_CATEGORIES:
        return 0  # No Hate
    else:
        # Includes NA, None, 'nan', or any other unlisted label
        return -1  # To be dropped

# --- Standardized Clean Text Function ---
def clean_text(text: str) -> str:
    """Standardized text cleaning, including URL and user/subreddit replacement."""
    if not isinstance(text, str): return ""
    # Replace URLs with [URL]
    text = re.sub(r'http\S+|www\S+', '[URL]', text)
    # Replace u/ or r/ followed by non-space characters with [USER] or [SUBREDDIT]
    text = re.sub(r'u/\S+', '[USER]', text)
    text = re.sub(r'r/\S+', '[SUBREDDIT]', text) 
    # Replace @[user] with [USER]
    text = re.sub(r'@[a-zA-Z0-9_]+', '[USER]', text)
    # Remove special characters, keeping basic punctuation
    text = re.sub(r'[^\w\s.,!?;:\'"()-]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_hateful_data(file_path: str):
    """
    Processes the Hateful Parquet file, extracts and cleans data, 
    and standardizes column names. Drops rows with undefined labels.
    """
    try:
        # Load the data
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None, None, None

    # --- Data Standardization and Cleaning ---

    # 1. Standardize Author
    if 'author' in df.columns:
        df['author'] = df['author'].replace(['[deleted]', '[removed]', 'null', None], '__missing_user__') 
        df['author'] = df['author'].fillna('__missing_user__') 
    else:
        print("Error: 'author' column not found.")
        return None, None, None, None

    # 2. Deduplication Logic
    original_len = len(df)
    df.drop_duplicates(subset=['id'], keep='first', inplace=True)
    if len(df) < original_len:
        print(f"INFO: Dropped {original_len - len(df)} duplicates based on 'id'.")
    
    # 3. Content Cleaning
    df['body'] = df['body'].apply(clean_text)

    # 4. Title Handling
    df['title'] = ''
        
    # 5. Label Conversion & Filtering (Crucial Change)
    df['original_label'] = df['label'].astype(str) 
    
    # Apply mapping: 1 (Hate), 0 (No Hate), -1 (Drop)
    df['temp_label_check'] = df['original_label'].apply(map_to_binary_label)
    
    # Filter out the -1s (NA/None)
    pre_filter_count = len(df)
    df = df[df['temp_label_check'] != -1].copy()
    post_filter_count = len(df)
    dropped_count = pre_filter_count - post_filter_count
    
    if dropped_count > 0:
        print(f"INFO: Dropped {dropped_count} rows containing 'NA', 'None', or invalid labels.")
    
    # Finalize the label column
    df['label'] = df['temp_label_check'].astype('int64')
    df.drop(columns=['temp_label_check'], inplace=True)

    # --- Edge and Node Component Creation ---
    
    # These DataFrames contain all data before balancing, which is critical 
    # for correctly establishing the full universe of users/communities before splitting.
    
    # 1. User-community active edges (user_id, community_id)
    user_community_edges_df = df[['author', 'subreddit']].rename(columns={
        'author': 'user_id', 
        'subreddit': 'community_id'
    })

    # 2. Text-user edges (post_id, user_id)
    post_user_edges_df = df[['id', 'author']].rename(columns={
        'id': 'post_id', 
        'author': 'user_id'
    })

    # 3. Text-community edges (post_id, community_id)
    post_community_edges_df = df[['id', 'subreddit']].rename(columns={
        'id': 'post_id', 
        'subreddit': 'community_id'
    })

    # 4. All post nodes (The main text/node DataFrame)
    post_nodes_df = df[['id', 'body', 'label', 'author', 'subreddit', 'title']].rename(columns={
        'id': 'post_id', 
        'body': 'content', 
        'author': 'user_id', 
        'subreddit': 'community_id'
    })

    return user_community_edges_df, post_user_edges_df, post_community_edges_df, post_nodes_df

def write_split_files(data_df: pd.DataFrame, edge_df_list: list, unique_users: set, path: str):
    """
    Writes the split DataFrames and unique users to the specified directory, 
    enforcing the six-file output and schemas.
    """
    
    os.makedirs(path, exist_ok=True)
    folder_name = os.path.basename(path) 
    
    # --- 1. Node File: all_text_nodes.parquet ---
    data_df_to_save = data_df.copy()
    
    # Drop temporary/linking columns: 'user_id' (used for edges)
    data_df_to_save.drop(columns=['user_id'], errors='ignore', inplace=True) 
    
    # Enforce schema order and column list: post_id, content, title, community_id, label
    data_df_to_save = data_df_to_save[['post_id', 'content', 'title', 'community_id', 'label']]
    
    # Write nodes with schema
    pq.write_table(pa.Table.from_pandas(data_df_to_save, schema=TEXT_NODES_SCHEMA), os.path.join(path, TEXT_NODES_FILENAME))
    
    # Calculate and print stats for the saved file
    total_count = len(data_df_to_save)
    hateful_count = data_df_to_save['label'].sum() # Since label is 1 for hateful
    
    print(f"  {folder_name}/{TEXT_NODES_FILENAME} created with {total_count} posts.")
    print(f"  -> Hateful (label=1): {hateful_count} | Not Hateful (label=0): {total_count - hateful_count}")

    # --- 2. Edge Files ---
    for df_in, name in edge_df_list:
        df = df_in.copy()
        df.drop_duplicates(inplace=True)
        
        # Standardize edge columns and schema based on name
        if name == TEXT_COMMUNITY_EDGES_FILENAME.replace(".parquet", ""):
            df.rename(columns={'post_id': 'source_id', 'community_id': 'target_id'}, inplace=True)
            df['edge_type'] = 'posts_in'
            schema = TEXT_COMMUNITY_EDGES_SCHEMA
        elif name == TEXT_USER_EDGES_FILENAME.replace(".parquet", ""):
            df.rename(columns={'post_id': 'source_id', 'user_id': 'target_id'}, inplace=True)
            df['edge_type'] = 'posted_by'
            schema = TEXT_USER_EDGES_SCHEMA
        elif name == USER_COMMUNITY_ACTIVE_EDGES_FILENAME.replace(".parquet", ""):
            df.rename(columns={'user_id': 'source_id', 'community_id': 'target_id'}, inplace=True)
            df['edge_type'] = 'active_in'
            schema = USER_COMMUNITY_ACTIVE_EDGES_SCHEMA
        else:
            continue

        pq.write_table(pa.Table.from_pandas(df, schema=schema), os.path.join(path, f"{name}.parquet"))
        print(f"  {folder_name}/{name}.parquet created with {len(df)} rows.")

    # --- 3. Node File: unique_user_nodes.parquet ---
    user_nodes_df = pd.DataFrame(list(unique_users), columns=['user_id'])
    pq.write_table(pa.Table.from_pandas(user_nodes_df, schema=USER_NODES_SCHEMA), os.path.join(path, USER_NODES_FILENAME))
    print(f"  {folder_name}/{USER_NODES_FILENAME} created with {len(user_nodes_df)} rows.")
    
    # --- 4. Node File: unique_community_nodes.parquet ---
    # Retrieve 'community_id' from the original data_df
    community_nodes_df = pd.DataFrame(data_df['community_id'].unique(), columns=['community_id'])
    pq.write_table(pa.Table.from_pandas(community_nodes_df, schema=COMMUNITY_NODES_SCHEMA), os.path.join(path, COMMUNITY_NODES_FILENAME))
    print(f"  {folder_name}/{COMMUNITY_NODES_FILENAME} created with {len(community_nodes_df)} rows.")
    
    print("-" * 40)


def main():
    """
    Entry point to process the Hateful Parquet file, perform balancing, 
    stratified split, and save resulting graph components.
    """

    # --- Initial Setup ---
    print("--- Starting Hateful Graph Parquet Generation (Balanced & Split) ---")
    if os.path.exists(DATA_PATH_GRAPH_OUTPUT):
        print(f"INFO: Removing existing graph data directory: {DATA_PATH_GRAPH_OUTPUT}")
        shutil.rmtree(DATA_PATH_GRAPH_OUTPUT)
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    # --- PHASE 1: Load and Pre-process Data (Includes Binary Mapping) ---
    print(f"\n--- PHASE 1: Processing Hateful Parquet file & Applying Binary Label Mapping ---")
    
    all_user_community_edges, all_post_user_edges, all_post_community_edges, all_post_nodes_full = process_hateful_data(HATEFUL_path)

    if all_post_nodes_full is None or all_post_nodes_full.empty:
        print("ERROR: No data accumulated. Exiting.")
        return
        
    # Print Initial Label Distribution
    initial_counts = all_post_nodes_full['label'].value_counts()
    print("\n--- Initial Binary Label Distribution (After Cleaning NAs) ---")
    print(f"Hate (1):     {initial_counts.get(1, 0)}")
    print(f"No Hate (0):  {initial_counts.get(0, 0)}")
    print("------------------------------------------")

    # --- PHASE 2: Balancing/Undersampling ---
    print("\n--- PHASE 2: Undersampling 'No Hate' (0) Class for Perfect Balance ---")
    
    hate_nodes = all_post_nodes_full[all_post_nodes_full['label'] == 1]
    nohate_nodes = all_post_nodes_full[all_post_nodes_full['label'] == 0]
    
    # Determine the size of the minority class (Hate)
    minority_size = len(hate_nodes)
    
    if minority_size == 0:
        print("ERROR: No Hateful examples found (label=1). Cannot balance. Exiting.")
        return

    # Randomly sample the majority class (No Hate) to match the minority size
    # Ensure we don't try to sample more than exists
    sample_n = min(minority_size, len(nohate_nodes))
    
    nohate_nodes_sampled = nohate_nodes.sample(
        n=sample_n, 
        random_state=RANDOM_SEED,
        replace=False # Sample without replacement
    )

    # Combine the balanced datasets
    all_post_nodes = pd.concat([hate_nodes, nohate_nodes_sampled], ignore_index=True)
    
    print(f"✅ Data Balanced: Total Balanced Size = {len(all_post_nodes)} posts.")
    print(f"   (Hate: {len(hate_nodes)} | No Hate: {len(nohate_nodes_sampled)})")

    # --- PHASE 3: Stratified Split of BALANCED Post Nodes ---
    print("\n--- PHASE 3: Performing 80/20 Stratified Split on Balanced Texts ---")
    
    # Stratified Split on the BALANCED posts using the binary 'label' column
    train_nodes, test_nodes = train_test_split(
        all_post_nodes,
        test_size=1.0 - TRAIN_SPLIT_RATIO,
        stratify=all_post_nodes['label'],
        random_state=RANDOM_SEED
    )

    # Print Split Statistics for Verification (Should show perfect balance in each split)
    print("\n--- Split Statistics (Label Balance) ---")
    
    def print_split_stats(name, df):
        total_count = len(df)
        hateful_count = df['label'].sum() 
        not_hateful_count = total_count - hateful_count
        
        print(f"  {name} Total Posts: {total_count}")
        print(f"    -> Hateful (label=1): {hateful_count} | Not Hateful (label=0): {not_hateful_count}")

    print_split_stats("Train", train_nodes)
    print_split_stats("Test", test_nodes)
    print("------------------------------------------")


    # --- PHASE 4: Splitting Edges and Users (Filtered by BALANCED Post IDs) ---
    print("\n--- PHASE 4: Deriving Train/Test Edges and Users from Balanced Node Splits ---")

    train_post_ids = set(train_nodes['post_id'])
    test_post_ids = set(test_nodes['post_id'])
    
    # The edges must be filtered based on the post IDs that survived the balancing/split
    
    # 1. Edges: Filter based on post_id 
    all_post_community_edges.drop_duplicates(inplace=True)
    all_post_user_edges.drop_duplicates(inplace=True)
    all_user_community_edges.drop_duplicates(inplace=True)
    
    train_post_community_edges = all_post_community_edges[all_post_community_edges['post_id'].isin(train_post_ids)]
    test_post_community_edges = all_post_community_edges[all_post_community_edges['post_id'].isin(test_post_ids)]
    
    train_post_user_edges = all_post_user_edges[all_post_user_edges['post_id'].isin(train_post_ids)]
    test_post_user_edges = all_post_user_edges[all_post_user_edges['post_id'].isin(test_post_ids)]

    # 2. Users: Unique users in each split
    train_users = set(train_nodes['user_id'].unique())
    test_users = set(test_nodes['user_id'].unique())
    
    # 3. User-Community Edges: Filter based on user_id 
    train_user_community_edges = all_user_community_edges[all_user_community_edges['user_id'].isin(train_users)]
    test_user_community_edges = all_user_community_edges[all_user_community_edges['user_id'].isin(test_users)]
    
    print(f"Total Train Users: {len(train_users)}")
    print(f"Total Test Users: {len(test_users)}")
    print(f"Overlapping Users (for robustness check): {len(train_users.intersection(test_users))}")


    # --- PHASE 5: Writing Split Parquet Files ---
    print("\n--- PHASE 5: Writing Train and Test Graph Components to Disk ---")

    # Train Split components
    train_edges_list = [
        (train_post_community_edges, TEXT_COMMUNITY_EDGES_FILENAME.replace(".parquet", "")),
        (train_post_user_edges, TEXT_USER_EDGES_FILENAME.replace(".parquet", "")),
        (train_user_community_edges, USER_COMMUNITY_ACTIVE_EDGES_FILENAME.replace(".parquet", ""))
    ]
    write_split_files(train_nodes, train_edges_list, train_users, TRAIN_DIR)

    # Test Split components
    test_edges_list = [
        (test_post_community_edges, TEXT_COMMUNITY_EDGES_FILENAME.replace(".parquet", "")),
        (test_post_user_edges, TEXT_USER_EDGES_FILENAME.replace(".parquet", "")),
        (test_user_community_edges, USER_COMMUNITY_ACTIVE_EDGES_FILENAME.replace(".parquet", ""))
    ]
    write_split_files(test_nodes, test_edges_list, test_users, TEST_DIR)

    print("\n--- SCRIPT COMPLETE ---")
    print(f"All graph files are located in: {DATA_PATH_GRAPH_OUTPUT}")

if __name__ == "__main__":
    main()