import os
import shutil
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import re
from typing import List, Tuple, Set, Optional, Dict

# --- Configuration ---
ROOT_PATH = "../data/dreaddit_2/"
# Define Input Files
INPUT_FILES = {
    'train': "../data/dreaddit/dreaddit-train-expansion.csv",
    'test': "../data/dreaddit/dreaddit-test-expansion.csv"
}
# Output root directory
DATA_PATH_GRAPH_OUTPUT = os.path.join(ROOT_PATH, "graph_data")

# Standardized Output File Names
TEXT_NODES_FILENAME = "all_text_nodes.parquet"
USER_NODES_FILENAME = "unique_user_nodes.parquet"
COMMUNITY_NODES_FILENAME = "unique_community_nodes.parquet"
TEXT_COMMUNITY_EDGES_FILENAME = "text_community_edges.parquet"
TEXT_USER_EDGES_FILENAME = "text_user_edges.parquet"
USER_COMMUNITY_ACTIVE_EDGES_FILENAME = "user_community_active_edges.parquet"


# --- Standardized PyArrow Schemas ---
# (Schemas remain the same)
TEXT_NODES_SCHEMA = pa.schema([
    ('post_id', pa.string()),
    ('content', pa.string()),
    ('title', pa.string()),
    ('community_id', pa.string()),
    ('label', pa.bool_()), # Ensure this aligns with the converted column
])

USER_NODES_SCHEMA = pa.schema([
    ('user_id', pa.string())
])

COMMUNITY_NODES_SCHEMA = pa.schema([
    ('community_id', pa.string())
])

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


def process_csv_file(file_path: str) -> Optional[pd.DataFrame]:
    """
    Loads, cleans, and deduplicates the Dreaddit CSV file.
    Returns the cleaned DataFrame ready for component extraction.
    """
    try:
        # Use a copy immediately after reading to avoid SettingWithCopyWarning from the start
        df = pd.read_csv(file_path).copy() 
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    # Filter out rows with missing or empty values in key columns
    df.dropna(subset=['post_id', 'subreddit', 'text'], inplace=True)

    # --- Standardized Author Modification ---
    if 'author' in df.columns:
        missing_authors = ['[deleted]', '[removed]', 'null', None]
        # First, replace the explicit strings/None
        df['author'] = df['author'].replace(missing_authors, '__missing_user__')
        # Then, fill any remaining NaNs (which would be float NaNs)
        df['author'].fillna('__missing_user__', inplace=True) # This inplace is fine on the full series
    else:
        print(f"Error: 'author' column not found in {file_path}. Cannot generate user-related edges.")
        return None
    
    # Sort by 'label' in descending order (1 > 0) to prepare for deduplication, 
    # ensuring the labeled post is kept if duplicates exist.
    df = df.sort_values(by='label', ascending=False)
    # Deduplicate based on 'post_id', keeping the highest-label entry.
    df.drop_duplicates(subset=['post_id'], keep='first', inplace=True)

    # Ensure each final post has a community.
    df.dropna(subset=['subreddit'], inplace=True)

    # --- Data Standardization and Cleaning ---
    
    if 'label' in df.columns:
        # Assume 1 is True (Distress) and 0 is False (No Distress)
        df['label'] = df['label'].astype(bool) 
    
    # 1. Title Handling: Ensure 'title' column exists and handle NaNs/dtypes
    if 'title' not in df.columns:
        df['title'] = ''
        
    df['title'] = df['title'].astype(str).fillna('')
    df['title'] = df['title'].apply(clean_text)

    # 2. Content Cleaning
    df['text'] = df['text'].astype(str).apply(clean_text)

    # Final DF for component extraction
    return df

def generate_graph_components(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    # 1. User-community active edges
    user_community_edges_df = df[['author', 'subreddit']].rename(columns={'author': 'user_id', 'subreddit': 'community_id'})

    # 2. Text-user edges
    post_user_edges_df = df[['post_id', 'author']].rename(columns={'author': 'user_id'})

    # 3. Text-community edges
    post_community_edges_df = df[['post_id', 'subreddit']].rename(columns={'subreddit': 'community_id'})

    # 4. All text nodes
    post_nodes_df = df[['post_id', 'text', 'title', 'subreddit', 'label']].rename(columns={
        'text': 'content', 
        'subreddit': 'community_id'
    })
    
    # 5. Unique User Nodes
    user_nodes_df = pd.DataFrame(df['author'].unique(), columns=['user_id'])
    
    # 6. Unique Community Nodes
    community_nodes_df = pd.DataFrame(df['subreddit'].unique(), columns=['community_id'])

    return {
        'all_text_nodes': post_nodes_df,
        'unique_user_nodes': user_nodes_df,
        'unique_community_nodes': community_nodes_df,
        'text_community_edges': post_community_edges_df,
        'text_user_edges': post_user_edges_df,
        'user_community_active_edges': user_community_edges_df
    }


def write_components_to_parquet(components: Dict[str, pd.DataFrame], data_split: str):
    """Writes the extracted components to the standardized Parquet files for the split."""
    
    OUTPUT_SUBDIR = os.path.join(DATA_PATH_GRAPH_OUTPUT, data_split)
    os.makedirs(OUTPUT_SUBDIR, exist_ok=True)
    
    print(f"\n--- Writing Graph Components for {data_split.upper()} ---")

    # Mapping of component names to file names and schemas
    file_map = {
        'all_text_nodes': (TEXT_NODES_FILENAME, TEXT_NODES_SCHEMA),
        'unique_user_nodes': (USER_NODES_FILENAME, USER_NODES_SCHEMA),
        'unique_community_nodes': (COMMUNITY_NODES_FILENAME, COMMUNITY_NODES_SCHEMA),
        'text_community_edges': (TEXT_COMMUNITY_EDGES_FILENAME, TEXT_COMMUNITY_EDGES_SCHEMA),
        'text_user_edges': (TEXT_USER_EDGES_FILENAME, TEXT_USER_EDGES_SCHEMA),
        'user_community_active_edges': (USER_COMMUNITY_ACTIVE_EDGES_FILENAME, USER_COMMUNITY_ACTIVE_EDGES_SCHEMA),
    }

    for component_name, (file_name, schema) in file_map.items():
        df = components[component_name].copy()
        
        # Standardize edge columns before writing
        if 'edges' in component_name:
            if 'post_id' in df.columns:
                # Use a dictionary mapping for cleaner renames
                df.rename(columns={'post_id': 'source_id'}, inplace=True)
                if 'community_id' in df.columns and 'user_id' not in df.columns:
                    df.rename(columns={'community_id': 'target_id'}, inplace=True)
                elif 'user_id' in df.columns and 'community_id' not in df.columns:
                    df.rename(columns={'user_id': 'target_id'}, inplace=True)
            elif 'user_id' in df.columns and 'community_id' in df.columns:
                 df.rename(columns={'user_id': 'source_id', 'community_id': 'target_id'}, inplace=True)
            
            if 'text_community_edges' in component_name:
                df['edge_type'] = 'posts_in'
            elif 'text_user_edges' in component_name:
                df['edge_type'] = 'posted_by'
            elif 'user_community_active_edges' in component_name:
                df['edge_type'] = 'active_in'
            
            # Drop duplicates one last time on standardized (source, target) pair
            df.drop_duplicates(subset=['source_id', 'target_id'], inplace=True)
        
        # Write to Parquet with explicit schema
        file_path = os.path.join(OUTPUT_SUBDIR, file_name)
        pq.write_table(pa.Table.from_pandas(df, schema=schema), file_path)
        print(f"{file_name} created with {len(df):,} rows.")
        
        # Print label breakdown for the main nodes file
        if component_name == 'all_text_nodes':
            # This should now correctly count True/False because it was converted to bool earlier
            label_counts = df['label'].value_counts()
            print("  -> Label Breakdown (Boolean):")
            print(f"     - True (1, Distress): {label_counts.get(True, 0)}")
            print(f"     - False (0, No Distress): {label_counts.get(False, 0)}")


def process_split(data_split: str, input_path: str):
    """Coordinates the processing of a single train/test split."""
    print(f"\n--- Starting Graph Parquet Generation for: {data_split.upper()} ---")

    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found at {input_path}. Skipping {data_split}.")
        return

    # 1. Load, Clean, and Deduplicate Data
    df_cleaned = process_csv_file(input_path)
    
    if df_cleaned is None or df_cleaned.empty:
        print(f"WARNING: No valid data generated for {data_split}.")
        return

    print(f"INFO: Processed {len(df_cleaned):,} records for {data_split}.")

    # 2. Generate Graph Components
    components = generate_graph_components(df_cleaned)

    # 3. Write Components to Parquet
    write_components_to_parquet(components, data_split)
    
    print(f"--- Graph files for {data_split.upper()} generated successfully. ---")


def main():
    """Main function to loop over the train and test input files."""
    print("--- Starting Dreaddit Graph Parquet Generation (Train & Test) ---")
    
    # Ensure the root output directory exists and is clean
    if os.path.exists(DATA_PATH_GRAPH_OUTPUT):
        print(f"INFO: Removing existing graph data directory: {DATA_PATH_GRAPH_OUTPUT}")
        shutil.rmtree(DATA_PATH_GRAPH_OUTPUT)
    os.makedirs(DATA_PATH_GRAPH_OUTPUT, exist_ok=True)

    for data_split, input_path in INPUT_FILES.items():
        process_split(data_split, input_path)

    print("\n\n--- ALL Graph Structure Files Generated. ---")


if __name__ == "__main__":
    main()