import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import re
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import numpy as np
import argparse

# --- Configuration ---
EMBEDDING_MODEL_NAME = 'google/embeddinggemma-300m'
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_ID_COLUMN = 'post_id' #'post_id' for dst, 'id' for pretrain + rec
INPUT_TEXT_COLUMN = 'content' #'content' for dst, 'text_content' for the pretrain + rec
INPUT_TITLE_COLUMN = 'title'
INPUT_LABEL_COLUMN = 'label'

EMBEDDING_MODEL_NAME = 'google/embeddinggemma-300m'
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Helper functions ---
def get_total_rows(file_path: str) -> int:
    try:
        parquet_file = pq.ParquetFile(file_path)
        return parquet_file.metadata.num_rows
    except Exception as e:
        print(f"Error reading parquet metadata: {e}")
        exit(1)

model = None
def initialize_model():
    global model
    if model is None:
        print(f"Initializing model on device: {DEVICE}")
        model = SentenceTransformer(EMBEDDING_MODEL_NAME, trust_remote_code=True).to(DEVICE)
        model.eval()

def clean_text(text: str) -> str:
    if not isinstance(text, str): return ""
    text = re.sub(r'[^\w\s.,!?;:\'"()-]', '', text)
    text = re.sub(r'http\S+|www\S+', '[URL]', text)
    text = re.sub(r'u/\S+', '[USER]', text)
    text = re.sub(r'@[a-zA-Z0-9_]+', '[USER]', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_gemma_embeddings(texts: list[str]) -> np.ndarray:
    embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False, convert_to_numpy=True, prompt_name="Retrieval-document")
    return embeddings

def compute_node_embeddings_chunk(start_row: int, end_row: int, job_id: int, text_nodes_path: str, temp_dir: str):
    initialize_model() # Make sure model is loaded
    print(f"--- Starting Node Embedding Computation for job {job_id} (rows {start_row}-{end_row}) ---")
    os.makedirs(temp_dir, exist_ok=True)
    output_path = os.path.join(temp_dir, f"chunk_{job_id}.parquet")
    
    parquet_file = pq.ParquetFile(text_nodes_path)
    input_schema = parquet_file.schema.to_arrow_schema()
    has_label = INPUT_LABEL_COLUMN in input_schema.names
    
    output_fields = [pa.field('id', pa.string()), pa.field('embedding', pa.list_(pa.float32()))]
    if has_label:
        output_fields.append(pa.field('label', input_schema.field(INPUT_LABEL_COLUMN).type))
    output_schema = pa.schema(output_fields)
    
    columns_to_read = [INPUT_ID_COLUMN, INPUT_TEXT_COLUMN, INPUT_TITLE_COLUMN] + ([INPUT_LABEL_COLUMN] if has_label else [])
    
    with pq.ParquetWriter(output_path, output_schema, compression='snappy') as writer:
        current_row = 0
        total_rows_in_job = end_row - start_row
        with tqdm(total=total_rows_in_job, desc=f"Job {job_id}", unit="rows") as pbar:
            for batch in parquet_file.iter_batches(batch_size=10000, columns=columns_to_read):
                df_full = batch.to_pandas()
                batch_start_row, batch_end_row = current_row, current_row + len(df_full)
                slice_start, slice_end = max(0, start_row - batch_start_row), min(len(df_full), end_row - batch_start_row)
                
                if slice_start >= slice_end:
                    current_row = batch_end_row
                    if current_row >= end_row: break
                    continue

                df_process = df_full.iloc[slice_start:slice_end].copy()
                rows_in_slice = len(df_process)
                
                df_process.rename(columns={
                    INPUT_ID_COLUMN: 'id', 
                    INPUT_TEXT_COLUMN: 'text_content', 
                    INPUT_TITLE_COLUMN: 'title'
                }, inplace=True)
                
                if has_label:
                    df_process.rename(columns={INPUT_LABEL_COLUMN: 'label'}, inplace=True)
                
                df_process['title'] = df_process['title'].fillna('')
                df_process['text_content'] = df_process['text_content'].fillna('')
                
                df_process['combined_text'] = df_process['title'] + ' ' + df_process['text_content']
                
                df_process['combined_text'] = df_process['combined_text'].apply(clean_text)
                
                df_process.dropna(subset=['combined_text'], inplace=True)
                df_process = df_process[df_process['combined_text'] != '']
                
                if not df_process.empty:
                    texts = df_process['combined_text'].tolist()
                    embeddings = get_gemma_embeddings(texts)
                    
                    records = {'id': df_process['id'].tolist(), 'embedding': [emb.tolist() for emb in embeddings]}
                    if has_label: records['label'] = df_process['label'].tolist()
                    
                    table = pa.Table.from_pydict(records, schema=output_schema)
                    writer.write_table(table)
                    
                pbar.update(rows_in_slice)
                current_row = batch_end_row
                if current_row >= end_row: break
                
    print(f"Job {job_id} Complete: Saved results to {output_path}")

def combine_parquet_files(input_dir: str, output_path: str):
    print(f"\n--- Combining temporary Parquet files from {input_dir} ---")
    temp_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.startswith('chunk_') and f.endswith('.parquet')]
    if not temp_files:
        print("No temporary files found to combine. Exiting.")
        return
    master_schema = pq.read_schema(temp_files[0])
    print(f"Writing combined file to: {output_path}")
    with pq.ParquetWriter(output_path, master_schema, compression='snappy') as writer:
        for file_path in tqdm(temp_files, desc="Combining chunks"):
            writer.write_table(pq.read_table(file_path))
    print(f"Successfully combined {len(temp_files)} files into {output_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate or combine embeddings for text nodes using EmbeddingGemma.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the graph_data directory for the task.")
    parser.add_argument("--mode", type=str, required=True, choices=['compute', 'combine', 'count'], help="Operation to perform.")
    parser.add_argument("--start_row", type=int, help="Starting row index for 'compute' mode.")
    parser.add_argument("--end_row", type=int, help="Ending row index for 'compute' mode.")
    parser.add_argument("--job_id", type=int, help="Unique ID for this job/chunk in 'compute' mode.")
    args = parser.parse_args()

    DATA_PATH = args.data_path
    TEMP_EMBEDDINGS_DIR = os.path.join(DATA_PATH, "temp_text_embeddings_gemma") 
    TEXT_NODES_PARQUET_PATH = os.path.join(DATA_PATH, "all_text_nodes.parquet")
    TEXT_EMBEDDINGS_FINAL_PATH = os.path.join(DATA_PATH, "text_node_embeddings_gemma.parquet")

    if args.mode == 'count':
        total_rows = get_total_rows(TEXT_NODES_PARQUET_PATH)
        print(total_rows)
    elif args.mode == 'compute':
        if args.start_row is None or args.end_row is None or args.job_id is None:
            parser.error("--start_row, --end_row, and --job_id are required for 'compute' mode.")
        compute_node_embeddings_chunk(args.start_row, args.end_row, args.job_id, TEXT_NODES_PARQUET_PATH, TEMP_EMBEDDINGS_DIR)
    elif args.mode == 'combine':
        combine_parquet_files(TEMP_EMBEDDINGS_DIR, TEXT_EMBEDDINGS_FINAL_PATH)