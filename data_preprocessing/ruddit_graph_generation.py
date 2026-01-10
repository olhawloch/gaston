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
ROOT_PATH = "../data/ruddit/"
# Retained original output folder naming for consistency with the last script you provided
DATA_PATH_GRAPH_OUTPUT = os.path.join(ROOT_PATH, "graph_data_split") 
TRAIN_DIR = os.path.join(DATA_PATH_GRAPH_OUTPUT, "train")
TEST_DIR = os.path.join(DATA_PATH_GRAPH_OUTPUT, "test")

# --- INPUT DATA PATH ---
CSV_FILES_PATH = "../data/Ruddit/Dataset/Ruddit_full_expansion.csv"

# Hyperparameter for split
TRAIN_SPLIT_RATIO = 0.8
RANDOM_SEED = 42 # For reproducible train/test split

# Standardized Output File Names
TEXT_NODES_FILENAME = "all_text_nodes.parquet"
USER_NODES_FILENAME = "unique_user_nodes.parquet"
COMMUNITY_NODES_FILENAME = "unique_community_nodes.parquet"
TEXT_COMMUNITY_EDGES_FILENAME = "text_community_edges.parquet"
TEXT_USER_EDGES_FILENAME = "text_user_edges.parquet"
USER_COMMUNITY_ACTIVE_EDGES_FILENAME = "user_community_active_edges.parquet"


# --- Standardized PyArrow Schemas ---

# Schemas for Nodes (No timestamp, uses float64 label for continuous score)
TEXT_NODES_SCHEMA = pa.schema([
    ('post_id', pa.string()),
    ('content', pa.string()),
    ('title', pa.string()),
    ('community_id', pa.string()),
    ('label', pa.float64()), # Ruddit label is a continuous score
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

def process_csv_file(file_path: str):
    """
    Processes a single CSV file, extracts and cleans data.
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None, None, None, None

    # Filter out rows with missing data in essential columns, EXCLUDING 'author'
    df.dropna(subset=['comment_id', 'subreddit', 'comment_text'], inplace=True)
    
    # --- Standardized Author Modification (Fixed inplace=True warnings) ---
    if 'author' in df.columns:
        # Map deleted/removed/null/NaN authors to '__missing_user__'
        df['author'] = df['author'].replace(['[deleted]', '[removed]', 'null', None], '__missing_user__') # FIX: Avoid chained assignment
        df['author'] = df['author'].fillna('__missing_user__') # FIX: Avoid chained assignment
    else:
        print(f"Error: 'author' column not found in {file_path}. Skipping user-related edges.")
        return None, None, None, None
    
    # --- Deduplication Logic (CRITICAL for Score Priority) ---
    # Temporarily replace NaN scores with a unique value to prioritize scored entries
    df['offensiveness_score'] = df['offensiveness_score'].fillna(-999) 
    
    # Sort by 'offensiveness_score' (highest score/non-NaN entry first)
    df = df.sort_values(by='offensiveness_score', ascending=False)
    
    # Deduplicate based on 'comment_id', keeping the highest-priority/highest-score entry
    df.drop_duplicates(subset=['comment_id'], keep='first', inplace=True)

    # Restore the temporary value to NaN for the final output
    df['offensiveness_score'] = df['offensiveness_score'].replace(-999, np.nan)
    # --- End Deduplication Logic ---
    
    # --- Data Standardization and Cleaning ---
    
    # 1. Title Handling: Ensure 'title' column exists (renaming from submission_title)
    if 'submission_title' in df.columns:
        df.rename(columns={'submission_title': 'title'}, inplace=True)
    if 'title' not in df.columns:
        df['title'] = ''
        
    df['title'] = df['title'].fillna('') # FIX: Avoid chained assignment
    df['title'] = df['title'].apply(clean_text)

    # 2. Content Cleaning
    df['comment_text'] = df['comment_text'].apply(clean_text)

    # 1. User-community active edges
    user_community_edges_df = df[['author', 'subreddit']].rename(columns={'author': 'user_id', 'subreddit': 'community_id'})

    # 2. Text-user edges
    post_user_edges_df = df[['comment_id', 'author']].rename(columns={'comment_id': 'post_id', 'author': 'user_id'})

    # 3. Text-community edges
    post_community_edges_df = df[['comment_id', 'subreddit']].rename(columns={'comment_id': 'post_id', 'subreddit': 'community_id'})

    # 4. All post nodes (Contains continuous label and temporary IDs)
    post_nodes_df = df[['comment_id', 'comment_text', 'offensiveness_score', 'author', 'subreddit', 'title']].rename(columns={
        'comment_id': 'post_id', 
        'comment_text': 'content', 
        'offensiveness_score': 'label', # 'label' now holds the continuous score or NaN
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
    
    # Get the name of the folder (train or test)
    folder_name = os.path.basename(path) 
    
    # --- 1. Node File: all_text_nodes.parquet ---
    data_df_to_save = data_df.copy()
    
    # FIX: DO NOT drop 'community_id'. Drop temporary/linking columns: 'user_id', 'binary_label', 'timestamp'.
    data_df_to_save.drop(columns=['user_id', 'binary_label', 'timestamp'], errors='ignore', inplace=True)
    
    # Enforce schema order and column list
    data_df_to_save = data_df_to_save[['post_id', 'content', 'title', 'community_id', 'label']]
    
    # Write nodes with schema
    pq.write_table(pa.Table.from_pandas(data_df_to_save, schema=TEXT_NODES_SCHEMA), os.path.join(path, TEXT_NODES_FILENAME))
    
    # Calculate and print stats for the saved file
    labeled_count = data_df_to_save['label'].notna().sum()
    pos_count = (data_df_to_save['label'] > 0).sum()
    
    print(f"  {folder_name}/{TEXT_NODES_FILENAME} created with {len(data_df)} posts.")
    print(f"  -> Labeled posts: {labeled_count} (Score > 0: {pos_count}, Score <= 0/Neutral: {labeled_count - pos_count})")

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
            continue # Should not happen

        pq.write_table(pa.Table.from_pandas(df, schema=schema), os.path.join(path, f"{name}.parquet"))
        print(f"  {folder_name}/{name}.parquet created with {len(df)} rows.")

    # --- 3. Node File: unique_user_nodes.parquet ---
    user_nodes_df = pd.DataFrame(list(unique_users), columns=['user_id'])
    pq.write_table(pa.Table.from_pandas(user_nodes_df, schema=USER_NODES_SCHEMA), os.path.join(path, USER_NODES_FILENAME))
    print(f"  {folder_name}/{USER_NODES_FILENAME} created with {len(user_nodes_df)} rows.")
    
    # --- 4. Node File: unique_community_nodes.parquet (ADDED for 6-file consistency) ---
    # Need to retrieve 'community_id' from the original data_df, as it was dropped from data_df_to_save
    community_nodes_df = pd.DataFrame(data_df['community_id'].unique(), columns=['community_id'])
    pq.write_table(pa.Table.from_pandas(community_nodes_df, schema=COMMUNITY_NODES_SCHEMA), os.path.join(path, COMMUNITY_NODES_FILENAME))
    print(f"  {folder_name}/{COMMUNITY_NODES_FILENAME} created with {len(community_nodes_df)} rows.")
    
    print("-" * 40)


def main():
    """
    Entry point to process all CSVs, perform stratified split on labeled text nodes, 
    and save resulting graph components to 'train' and 'test' folders.
    """

    # --- Initial Setup ---
    print("--- Starting Ruddit Graph Parquet Generation (Train & Test) ---")
    if os.path.exists(DATA_PATH_GRAPH_OUTPUT):
        print(f"INFO: Removing existing graph data directory: {DATA_PATH_GRAPH_OUTPUT}")
        shutil.rmtree(DATA_PATH_GRAPH_OUTPUT)
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    # --- File Discovery Logic ---
    csv_files_to_process = []
    if os.path.isdir(CSV_FILES_PATH):
        for dirpath, _, filenames in os.walk(CSV_FILES_PATH):
            for filename in filenames:
                if filename.endswith(".csv"):
                    csv_files_to_process.append(os.path.join(dirpath, filename))
    elif os.path.isfile(CSV_FILES_PATH) and CSV_FILES_PATH.endswith(".csv"):
        csv_files_to_process.append(CSV_FILES_PATH)
    else:
        print(f"ERROR: The provided path is not a valid file or directory: {CSV_FILES_PATH}")
        return

    if not csv_files_to_process:
        print(f"WARNING: No .csv files found at the specified path: {CSV_FILES_PATH}")
        return


    # --- PHASE 1: Accumulate All Data and Deduplicate ---
    all_user_community_edges = pd.DataFrame()
    all_post_user_edges = pd.DataFrame()
    all_post_community_edges = pd.DataFrame()
    all_post_nodes = pd.DataFrame() 
    
    print(f"\n--- PHASE 1: Processing {len(csv_files_to_process)} CSV file(s) and accumulating data ---")
    
    for file_path in tqdm(csv_files_to_process, desc="Processing CSV files"):
        user_community_edges_df, post_user_edges_df, post_community_edges_df, post_nodes_df = process_csv_file(file_path)

        if user_community_edges_df is None:
            continue

        # Accumulate all components
        all_user_community_edges = pd.concat([all_user_community_edges, user_community_edges_df], ignore_index=True)
        all_post_user_edges = pd.concat([all_post_user_edges, post_user_edges_df], ignore_index=True)
        all_post_community_edges = pd.concat([all_post_community_edges, post_community_edges_df], ignore_index=True)
        all_post_nodes = pd.concat([all_post_nodes, post_nodes_df], ignore_index=True)

    # Final Deduplication on Post Nodes
    if all_post_nodes.empty:
        print("ERROR: No data accumulated. Exiting.")
        return
        
    # Sort by label (lowest score, including NaN, first) to prioritize the post with a score.
    # This deduplication logic is critical for the Ruddit dataset structure.
    all_post_nodes['label_temp'] = all_post_nodes['label'].fillna(np.inf)
    all_post_nodes = all_post_nodes.sort_values(by='label_temp', ascending=True).copy() # Use .copy() after sorting/filtering
    all_post_nodes.drop_duplicates(subset=['post_id'], keep='first', inplace=True)
    all_post_nodes.drop(columns=['label_temp'], inplace=True)
    
    # Filter Edges to only include those belonging to the final, unique set of Post Nodes
    final_post_ids = set(all_post_nodes['post_id'])
    all_post_community_edges = all_post_community_edges[all_post_community_edges['post_id'].isin(final_post_ids)].drop_duplicates()
    all_post_user_edges = all_post_user_edges[all_post_user_edges['post_id'].isin(final_post_ids)].drop_duplicates()
    all_user_community_edges.drop_duplicates(inplace=True)


    # --- PHASE 2: Stratified Split of Labeled Post Nodes ---
    print("\n--- PHASE 2: Performing 80/20 Stratified Split on Labeled Texts ---")
    
    # 1. Separate Labeled and Unlabeled Posts
    labeled_posts = all_post_nodes[all_post_nodes['label'].notna()]
    unlabeled_posts = all_post_nodes[all_post_nodes['label'].isna()]

    # 2. Create Binary Label for Stratification ONLY (Used for T-S Split)
    # This column is TEMPORARY and dropped later.
    labeled_posts = labeled_posts.copy() # Avoid SettingWithCopyWarning
    labeled_posts['binary_label'] = (labeled_posts['label'] > 0).astype(int)
    
    if len(labeled_posts) == 0:
        print("ERROR: No labeled posts found for splitting. Exiting.")
        return
    
    # 3. Perform Stratified Split on LABELED posts
    train_labeled, test_labeled = train_test_split(
        labeled_posts,
        test_size=1.0 - TRAIN_SPLIT_RATIO,
        stratify=labeled_posts['binary_label'],
        random_state=RANDOM_SEED
    )

    # 4. Recombine with Unlabeled Posts
    train_nodes = pd.concat([train_labeled, unlabeled_posts], ignore_index=True)
    test_nodes = test_labeled.copy()
    
    # Print Split Statistics for Verification
    print("\n--- Split Statistics (Label Balance based on Score > 0) ---")
    
    def print_split_stats(name, df_labeled, df_unlabeled=None):
        labeled_count = len(df_labeled)
        # Use the temporary binary_label for printing stats
        pos_count = df_labeled['binary_label'].sum()
        neg_count = labeled_count - pos_count
        unlabeled_count = len(df_unlabeled) if df_unlabeled is not None else 0
        
        print(f"  {name} Total Posts: {labeled_count + unlabeled_count}")
        print(f"    -> Labeled: {labeled_count} | Positive (Score > 0): {pos_count} | Negative (Score <= 0): {neg_count}")
        if df_unlabeled is not None:
            print(f"    -> Unlabeled (NA Score): {unlabeled_count}")

    print_split_stats("Train", train_labeled, unlabeled_posts)
    print_split_stats("Test", test_labeled)
    print("---------------------------------------------------------")


    # --- PHASE 3: Splitting Edges and Users ---
    print("\n--- PHASE 3: Deriving Train/Test Edges and Users from Node Splits ---")

    train_post_ids = set(train_nodes['post_id'])
    test_post_ids = set(test_nodes['post_id'])
    
    # 1. Edges: Filter based on post_id 
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


    # --- PHASE 4: Writing Split Parquet Files ---
    print("\n--- PHASE 4: Writing Train and Test Graph Components to Disk ---")

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