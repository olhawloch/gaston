import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
from tqdm import tqdm
from typing import Dict, List, Set, Optional, Any
import shutil
import re 

# --- Configuration ---
ROOT_PATH = "../data/normvio/"
# Input files list
INPUT_FILES = {
    'train': os.path.join(ROOT_PATH, "normvio_data_train_with_authors.parquet"),
    'test': os.path.join(ROOT_PATH, "normvio_data_test_with_authors.parquet")
}
# Output root directory
DATA_PATH_GRAPH_OUTPUT = "../data/normvio/graph_data"

# Standardized Output File Names
TEXT_NODES_FILENAME = "all_text_nodes.parquet"
USER_NODES_FILENAME = "unique_user_nodes.parquet"
COMMUNITY_NODES_FILENAME = "unique_community_nodes.parquet"
TEXT_COMMUNITY_EDGES_FILENAME = "text_community_edges.parquet"
TEXT_USER_EDGES_FILENAME = "text_user_edges.parquet"
USER_COMMUNITY_ACTIVE_EDGES_FILENAME = "user_community_active_edges.parquet"


# --- Standardized PyArrow Schemas ---

# Schemas for Nodes (No timestamp, includes title, uses boolean label for NormVio)
TEXT_NODES_SCHEMA = pa.schema([
    ('post_id', pa.string()),
    ('content', pa.string()),
    ('title', pa.string()),
    ('community_id', pa.string()),
    ('label', pa.bool_()), # NormVio's label ('bool_derail') is boolean
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


# --- Utility Function for Analysis ---

def analyze_output_files():
    """Reads the generated graph Parquet files, prints their sizes, and provides label breakdowns."""
    print("\n" + "="*70)
    print("📈 POST-PROCESSING ANALYSIS: FILE SIZES AND LABEL BREAKDOWN")
    print("="*70)

    for data_split in INPUT_FILES.keys():
        OUTPUT_SUBDIR = os.path.join(DATA_PATH_GRAPH_OUTPUT, data_split)
        TEXT_NODES_PATH = os.path.join(OUTPUT_SUBDIR, TEXT_NODES_FILENAME)
        USER_NODES_PATH = os.path.join(OUTPUT_SUBDIR, USER_NODES_FILENAME)
        COMMUNITY_NODES_PATH = os.path.join(OUTPUT_SUBDIR, COMMUNITY_NODES_FILENAME)
        
        print(f"\n--- Analysis for {data_split.upper()} Split ---")

        # 1. Print File Sizes and Row Counts
        print("\nNode and Edge File Sizes & Counts:")
        
        # Read the main node file to get the label breakdown
        if os.path.exists(TEXT_NODES_PATH):
            try:
                # Read only the necessary columns for efficiency
                text_nodes_df = pd.read_parquet(TEXT_NODES_PATH, columns=['post_id', 'community_id', 'label'])
                file_size_bytes = os.path.getsize(TEXT_NODES_PATH)
                file_size_mb = file_size_bytes / (1024 * 1024)
                print(f"  📜 {TEXT_NODES_FILENAME}: {len(text_nodes_df):<10,} rows | {file_size_mb:.2f} MB")
                
                # Print unique user and community counts from the other files (if they exist)
                if os.path.exists(USER_NODES_PATH):
                    user_nodes_df = pd.read_parquet(USER_NODES_PATH, columns=['user_id'])
                    user_size_bytes = os.path.getsize(USER_NODES_PATH)
                    user_size_mb = user_size_bytes / (1024 * 1024)
                    print(f"  👤 {USER_NODES_FILENAME}: {len(user_nodes_df):<10,} unique users | {user_size_mb:.2f} MB")

                if os.path.exists(COMMUNITY_NODES_PATH):
                    community_nodes_df = pd.read_parquet(COMMUNITY_NODES_PATH, columns=['community_id'])
                    comm_size_bytes = os.path.getsize(COMMUNITY_NODES_PATH)
                    comm_size_mb = comm_size_bytes / (1024 * 1024)
                    print(f"  🏘️ {COMMUNITY_NODES_FILENAME}: {len(community_nodes_df):<10,} unique communities | {comm_size_mb:.2f} MB")

            except Exception as e:
                print(f"⚠️ WARNING: Could not read graph files for analysis: {e}")
                continue # Skip label analysis if reading fails
            
            # 2. Label Breakdown (from all_text_nodes.parquet)
            print("\nLabel Breakdown ('bool_derail' value):")
            label_counts = text_nodes_df['label'].value_counts()
            label_perc = text_nodes_df['label'].value_counts(normalize=True).mul(100).round(2)
            
            label_summary = pd.DataFrame({
                'Count': label_counts,
                'Percentage': label_perc
            })
            
            # The 'bool_derail' column is boolean (True/False)
            label_summary.index = ['Violates Norm (True)' if i else 'No Violation (False)' for i in label_summary.index]
            print(label_summary.to_markdown(numalign="left", stralign="left"))
            
            # 3. Community Breakdown (Top 5)
            print("\nTop 5 Communities (Subreddits):")
            top_communities = text_nodes_df['community_id'].value_counts(normalize=True).head(5).mul(100).round(2)
            print(top_communities.to_markdown(numalign="left", stralign="left"))
            
        else:
            print(f"❌ Graph files for {data_split.upper()} not found. Skipping analysis.")


# --- Processing Function (Modified) ---

def process_file(data_split: str, input_path: str):
    """Generates all required graph Parquet files for a single input file into a dedicated subdirectory."""
    print(f"\n--- Starting Graph Parquet Generation for: {data_split.upper()} ---")

    # --- Define and create the output subdirectory ---
    OUTPUT_SUBDIR = os.path.join(DATA_PATH_GRAPH_OUTPUT, data_split)
    
    if os.path.exists(OUTPUT_SUBDIR):
        print(f"INFO: Removing existing output directory: {OUTPUT_SUBDIR}")
        shutil.rmtree(OUTPUT_SUBDIR)
    os.makedirs(OUTPUT_SUBDIR, exist_ok=True)
    
    # --- Define standardized output file paths (omitted for brevity) ---
    TEXT_NODES_PARQUET_PATH = os.path.join(OUTPUT_SUBDIR, TEXT_NODES_FILENAME)
    UNIQUE_USER_NODES_PARQUET_PATH = os.path.join(OUTPUT_SUBDIR, USER_NODES_FILENAME)
    UNIQUE_COMMUNITY_NODES_PARQUET_PATH = os.path.join(OUTPUT_SUBDIR, COMMUNITY_NODES_FILENAME) 
    TEXT_COMMUNITY_EDGES_PARQUET_PATH = os.path.join(OUTPUT_SUBDIR, TEXT_COMMUNITY_EDGES_FILENAME)
    TEXT_USER_EDGES_PARQUET_PATH = os.path.join(OUTPUT_SUBDIR, TEXT_USER_EDGES_FILENAME)
    USER_COMMUNITY_ACTIVE_EDGES_PARQUET_PATH = os.path.join(OUTPUT_SUBDIR, USER_COMMUNITY_ACTIVE_EDGES_FILENAME)
    
    # --- Step 1: Load and Prepare Data ---
    if not os.path.exists(input_path):
        print(f"❌ ERROR: Input file not found at {input_path}. Skipping {data_split}.")
        return

    try:
        df = pd.read_parquet(input_path)
        print(f"INFO: Loaded {len(df):,} records for {data_split}.")
    except Exception as e:
        print(f"❌ ERROR reading input Parquet for {data_split}: {e}. Skipping.")
        return

    # --- Pre-processing/Cleaning ---
    # Rename columns to match the output logic
    df = df.rename(columns={'comment_id': 'post_id', 
                            'final_comment_text': 'text', 
                            'subreddit': 'community_id',
                            'author': 'user_id'})
    
    # Filter out records where key columns (excluding user_id) are missing
    df_clean = df.dropna(subset=['post_id', 'community_id', 'text']).copy() # Use .copy() to avoid SettingWithCopyWarning
    
    if len(df_clean) < len(df):
        missing_count = len(df) - len(df_clean)
        print(f"⚠️ WARNING: Dropping {missing_count:,} records due to missing key columns (post_id, community_id, or text).")
        
    # --- Standardized Author Modification (For fine-tuning scripts 2-6) ---
    if 'user_id' in df_clean.columns:
        # Map deleted/removed/null/NaN authors to '__missing_user__' and fill any remaining NaNs
        df_clean['user_id'] = df_clean['user_id'].replace(['[deleted]', '[removed]', 'null', None], '__missing_user__').fillna('__missing_user__')

    # --- Step 2: Generate All Post Nodes (all_text_nodes.parquet) ---
    print("2. Generating Post Nodes...")
    
    # 2a. Content and Text Cleaning
    post_nodes_df = df_clean[['post_id', 'text', 'community_id', 'bool_derail']].copy()
    post_nodes_df = post_nodes_df.rename(columns={'text': 'content', 'bool_derail': 'label'})
    post_nodes_df['content'] = post_nodes_df['content'].apply(clean_text)

    # 2b. Add mandatory 'title' column (empty string since NormVio lacks it)
    if 'title' not in df_clean.columns:
        df_clean['title'] = ''
        
    post_nodes_df['title'] = df_clean['title'].astype(str).fillna('')
    post_nodes_df['title'] = post_nodes_df['title'].apply(clean_text)
    
    # 2c. Final Column Selection (REMOVED 'timestamp')
    post_nodes_df = post_nodes_df[['post_id', 'content', 'title', 'community_id', 'label']]
    
    # Use explicit schema when writing
    pq.write_table(pa.Table.from_pandas(post_nodes_df, schema=TEXT_NODES_SCHEMA), TEXT_NODES_PARQUET_PATH)
    print(f"✅ {TEXT_NODES_FILENAME} created with {len(post_nodes_df):,} rows.")

    # --- Step 3: Generate Text-Community Edges (text_community_edges.parquet) ---
    print("3. Generating Text-Community Edges...")
    text_community_edges_df = df_clean[['post_id', 'community_id']].copy().drop_duplicates()
    
    # Standardize edge columns
    text_community_edges_df = text_community_edges_df.rename(columns={'post_id': 'source_id', 'community_id': 'target_id'})
    text_community_edges_df['edge_type'] = 'posts_in'
    
    pq.write_table(pa.Table.from_pandas(text_community_edges_df, schema=TEXT_COMMUNITY_EDGES_SCHEMA), TEXT_COMMUNITY_EDGES_PARQUET_PATH)
    print(f"✅ {TEXT_COMMUNITY_EDGES_FILENAME} created with {len(text_community_edges_df):,} rows.")

    # --- Step 4: Generate Text-User Edges (text_user_edges.parquet) ---
    print("4. Generating Text-User Edges...")
    text_user_edges_df = df_clean[['post_id', 'user_id']].copy().drop_duplicates()
    
    # Standardize edge columns
    text_user_edges_df = text_user_edges_df.rename(columns={'post_id': 'source_id', 'user_id': 'target_id'})
    text_user_edges_df['edge_type'] = 'posted_by'
    
    pq.write_table(pa.Table.from_pandas(text_user_edges_df, schema=TEXT_USER_EDGES_SCHEMA), TEXT_USER_EDGES_PARQUET_PATH)
    print(f"✅ {TEXT_USER_EDGES_FILENAME} created with {len(text_user_edges_df):,} rows.")

    # ----------------------------------------------------------------------
    # --- Step 5: Generate Unique User Nodes (unique_user_nodes.parquet) ---
    # ----------------------------------------------------------------------
    print("5. Generating Unique User Nodes...")
    user_nodes_df = pd.DataFrame(df_clean['user_id'].unique(), columns=['user_id'])
    
    pq.write_table(pa.Table.from_pandas(user_nodes_df, schema=USER_NODES_SCHEMA), UNIQUE_USER_NODES_PARQUET_PATH)
    print(f"✅ {USER_NODES_FILENAME} created with {len(user_nodes_df):,} rows.")
    
    # -----------------------------------------------------------------------------
    # --- Step 6: Generate Unique Community Nodes (unique_community_nodes.parquet) ---
    # -----------------------------------------------------------------------------
    print("6. Generating Unique Community Nodes...")
    community_nodes_df = pd.DataFrame(df_clean['community_id'].unique(), columns=['community_id'])
    
    pq.write_table(pa.Table.from_pandas(community_nodes_df, schema=COMMUNITY_NODES_SCHEMA), UNIQUE_COMMUNITY_NODES_PARQUET_PATH)
    print(f"✅ {COMMUNITY_NODES_FILENAME} created with {len(community_nodes_df):,} rows.")
    
    # --- Step 7: Generate User-Community Active Edges (user_community_active_edges.parquet) ---
    print("7. Generating User-Community Active Edges...")
    user_community_edges_df = df_clean[['user_id', 'community_id']].copy().drop_duplicates()
    
    # Standardize edge columns
    user_community_edges_df = user_community_edges_df.rename(columns={'user_id': 'source_id', 'community_id': 'target_id'})
    user_community_edges_df['edge_type'] = 'active_in'
    
    pq.write_table(pa.Table.from_pandas(user_community_edges_df, schema=USER_COMMUNITY_ACTIVE_EDGES_SCHEMA), USER_COMMUNITY_ACTIVE_EDGES_PARQUET_PATH)
    print(f"✅ {USER_COMMUNITY_ACTIVE_EDGES_FILENAME} created with {len(user_community_edges_df):,} rows.")

    print(f"\n--- ✅ Graph files for {data_split.upper()} generated successfully in {OUTPUT_SUBDIR}. ---")

def main():
    """Main function to loop over all input files."""
    print("--- Starting NormVio Graph Parquet Generation (Train & Test) ---")
    
    # Ensure the root output directory exists
    os.makedirs(DATA_PATH_GRAPH_OUTPUT, exist_ok=True)

    for data_split, input_path in INPUT_FILES.items():
        process_file(data_split, input_path)

    print("\n\n--- ✅ ALL Graph Structure Files Generated. ---")
    
    # --- Added Analysis Step ---
    analyze_output_files()


if __name__ == "__main__":
    main()