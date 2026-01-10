import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import os
from typing import Optional

# --- Configuration ---
DATA_PATH = "../data/recommendation_2/graph_data/"

# --- File Paths ---
TEXT_NODES_PARQUET_PATH = os.path.join(DATA_PATH, "all_text_nodes.parquet")
USER_NODES_PARQUET_PATH = os.path.join(DATA_PATH, "unique_user_nodes.parquet")
COMMUNITY_NODES_PARQUET_PATH = os.path.join(DATA_PATH, "unique_community_nodes_2.parquet") 
TEXT_COMMUNITY_EDGES_PARQUET_PATH = os.path.join(DATA_PATH, "text_community_edges.parquet")
TEXT_USER_EDGES_PARQUET_PATH = os.path.join(DATA_PATH, "text_user_edges.parquet")
USER_COMMUNITY_ACTIVE_EDGES_PARQUET_PATH = os.path.join(DATA_PATH, "user_community_active_edges.parquet")

def inspect_parquet_file(file_path: str, sample_rows: int = 5):
    """
    Opens a Parquet file, prints its stats, schema, and a sample of its data.
    """
    print(f"\n--- Inspecting File: {os.path.basename(file_path)} ---")
    print(f"Full Path: {file_path}")

    if not os.path.exists(file_path):
        print(f"ERROR: File not found at {file_path}")
        return
    
    if os.path.getsize(file_path) == 0:
        print(f"WARNING: File is empty.")
        return

    try:
        parquet_file = pq.ParquetFile(file_path)
        metadata = parquet_file.metadata
        schema = parquet_file.schema.to_arrow_schema()

        print("\n--- PyArrow Schema ---")
        print(schema)

        print(f"\n--- Basic Statistics ---")
        print(f"Number of rows: {metadata.num_rows:,}")
        print(f"Number of columns: {metadata.num_columns}")
        
        print("\n--- Columns and Types ---")
        for field in schema:
            print(f"  - {field.name}: {field.type}")

        print(f"\n--- Sample Data (first {sample_rows} rows) ---")
        if metadata.num_rows > 0:
            head_table = parquet_file.read_row_group(0).slice(0, min(sample_rows, metadata.num_rows))
            head_df = head_table.to_pandas()

            long_string_columns = ['text_content', 'title', 'preview']
            for col in long_string_columns:
                if col in head_df.columns:
                    head_df[col] = head_df[col].astype(str).str.replace('\n|\r', ' ', regex=True)
                    head_df[col] = head_df[col].str.slice(0, 150).apply(lambda x: x + '...' if len(x) == 150 else x)

            print(head_df.to_markdown(index=False))
        else:
            print("No rows to display.")

    except pa.ArrowInvalid as e:
        print(f"ERROR: File {file_path} is not a valid Parquet file or is corrupted. Reason: {e}")
    except Exception as e:
        print(f"ERROR: Could not read or process {file_path}. Reason: {e}")

def get_unique_column_count(file_path: str, column_name: str) -> Optional[int]:
    """
    Reads a specific column from a Parquet file and returns the number of unique values.
    Returns None if the file or column doesn't exist or an error occurs.
    """
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        print(f"INFO: Skipping unique count for '{os.path.basename(file_path)}' (File not found or empty).")
        return None
    try:
        parquet_file = pq.ParquetFile(file_path)
        if column_name not in parquet_file.schema.to_arrow_schema().names:
            print(f"ERROR: Column '{column_name}' not found in {os.path.basename(file_path)}.")
            return None
        
        table = pq.read_table(file_path, columns=[column_name])
        return table.to_pandas()[column_name].nunique()
    except Exception as e:
        print(f"ERROR: Could not get unique count from {os.path.basename(file_path)}. Reason: {e}")
        return None

def main():
    """
    Main function to inspect all generated Parquet files and perform validation checks.
    """
    parquet_files_to_inspect = [
        # Node Files
        TEXT_NODES_PARQUET_PATH,
        USER_NODES_PARQUET_PATH,
        COMMUNITY_NODES_PARQUET_PATH,
        # Edge Files
        TEXT_COMMUNITY_EDGES_PARQUET_PATH,
        TEXT_USER_EDGES_PARQUET_PATH,
        USER_COMMUNITY_ACTIVE_EDGES_PARQUET_PATH,
    ]

    print("="*80)
    print("--- GENERAL FILE INSPECTION ---")
    print("="*80)
    
    for file_path in parquet_files_to_inspect:
        inspect_parquet_file(file_path)

    print("\n" + "="*80)
    print("--- DATA VALIDATION: COMMUNITY COUNT CONSISTENCY CHECK ---")
    print("="*80)

    # 1. Get total communities from the main node file
    print(f"\n1. Checking the primary community node file...")
    community_nodes_count = None
    if os.path.exists(COMMUNITY_NODES_PARQUET_PATH) and os.path.getsize(COMMUNITY_NODES_PARQUET_PATH) > 0:
        try:
            meta = pq.ParquetFile(COMMUNITY_NODES_PARQUET_PATH).metadata
            community_nodes_count = meta.num_rows
            print(f"Total communities in **{os.path.basename(COMMUNITY_NODES_PARQUET_PATH)}**: **{community_nodes_count:,}**")
        except Exception as e:
            print(f"ERROR: Could not read row count from {os.path.basename(COMMUNITY_NODES_PARQUET_PATH)}. Reason: {e}")
    else:
        print(f"INFO: Skipping '{os.path.basename(COMMUNITY_NODES_PARQUET_PATH)}' (File not found or empty).")

    # 2. Get unique communities from the text->community edge file
    print(f"\n2. Checking the text-to-community edge file...")
    text_comm_edge_count = get_unique_column_count(TEXT_COMMUNITY_EDGES_PARQUET_PATH, "target_id")
    if text_comm_edge_count is not None:
        print(f"Unique communities in **{os.path.basename(TEXT_COMMUNITY_EDGES_PARQUET_PATH)}**: **{text_comm_edge_count:,}** (from 'target_id' column)")

    # 3. Get unique communities from the user->community edge file
    print(f"\n3. Checking the user-to-community edge file...")
    user_comm_edge_count = get_unique_column_count(USER_COMMUNITY_ACTIVE_EDGES_PARQUET_PATH, "community_id")
    if user_comm_edge_count is not None:
        print(f"Unique communities in **{os.path.basename(USER_COMMUNITY_ACTIVE_EDGES_PARQUET_PATH)}**: **{user_comm_edge_count:,}** (from 'community_id' column)")

    # 4. Final Comparison
    print("\n--- Comparison Summary ---")
    counts = {
        'Community Node File': community_nodes_count,
        'Text->Comm Edge File': text_comm_edge_count,
        'User->Comm Edge File': user_comm_edge_count
    }
    
    valid_counts = [v for v in counts.values() if v is not None]

    if not valid_counts:
        print("Could not retrieve any community counts for comparison.")
    elif len(set(valid_counts)) == 1:
        print(f"**SUCCESS**: All available community counts match: **{valid_counts[0]:,}**")
    else:
        print("**WARNING**: Community counts do not match across files!")
        for name, count in counts.items():
            print(f"  - {name}: {count:,}" if count is not None else f"  - {name}: Not available")
        print("\nThis may indicate an issue in the data generation pipeline.")

    print("\n--- Inspection and Validation Complete ---")

if __name__ == "__main__":
    main()