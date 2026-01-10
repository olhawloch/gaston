import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import re
import torch
from transformers import BertTokenizer, BertModel
from tqdm.auto import tqdm
import numpy as np
import argparse
from typing import List

# --- Configuration ---
INPUT_ID_COLUMN = 'post_id' #'post_id', id for pretrain
INPUT_TEXT_COLUMN = 'content' # text_content for the pretrain dataset
INPUT_TITLE_COLUMN = 'title'
INPUT_LABEL_COLUMN = 'label'

# Model parameters
MODEL_NAME = 'bert-base-uncased'
MAX_LENGTH = 512    # Max sequence length (BERT and XLNet are same)
INTERNAL_GPU_BATCH_SIZE = 32     # Batch size for the model's forward pass
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Global Model Variables (for lazy loading) ---
tokenizer = None
model = None

# --- Helper functions ---
def get_total_rows(file_path: str) -> int:
    """Gets the total number of rows from Parquet metadata."""
    try:
        parquet_file = pq.ParquetFile(file_path)
        return parquet_file.metadata.num_rows
    except Exception as e:
        print(f"Error reading parquet metadata: {e}")
        exit(1)

def initialize_model():
    """Initializes the tokenizer and model, moving model to DEVICE."""
    global tokenizer, model
    if model is None:
        print(f"Initializing model '{MODEL_NAME}' on device: {DEVICE}")
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        model = BertModel.from_pretrained(MODEL_NAME).to(DEVICE)
        model.eval()

def clean_text(text: str) -> str:
    """Applies basic text cleaning."""
    if not isinstance(text, str): return ""
    text = re.sub(r'[^\w\s.,!?;:\'"()-]', '', text)
    text = re.sub(r'http\S+|www\S+', '[URL]', text)
    text = re.sub(r'u/\S+', '[USER]', text)
    text = re.sub(r'@[a-zA-Z0-9_]+', '[USER]', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_bert_embeddings(texts: list[str]) -> np.ndarray:
    """
    Generates BERT embeddings for a list of texts using
    internal batching for GPU efficiency.
    """
    all_embeddings: List[np.ndarray] = []
    
    # Internal batch processing (using INTERNAL_GPU_BATCH_SIZE)
    for i in range(0, len(texts), INTERNAL_GPU_BATCH_SIZE):
        batch_texts = texts[i:i + INTERNAL_GPU_BATCH_SIZE]

        inputs = tokenizer(
            batch_texts,
            return_tensors='pt',
            max_length=MAX_LENGTH,
            truncation=True,
            padding=True,
        ).to(DEVICE)
        
        with torch.no_grad():
            # Use **inputs syntax for BertModel
            outputs = model(**inputs)
            
        # Use the [CLS] token embedding (at index 0)
        # This is the standard for sentence-level tasks with BERT.
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        
        all_embeddings.extend(cls_embeddings.cpu().numpy())
    
    return np.array(all_embeddings)

def compute_node_embeddings_chunk(start_row: int, end_row: int, job_id: int, text_nodes_path: str, temp_dir: str):
    """
    Processes a specific 'chunk' (row slice) of the input Parquet file,
    as assigned by a SLURM worker.
    """
    initialize_model() # Make sure model is loaded
    print(f"--- Starting BERT Embedding Computation for job {job_id} (rows {start_row}-{end_row}) ---")
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
                
                slice_start = max(0, start_row - batch_start_row)
                slice_end = min(len(df_full), end_row - batch_start_row)
                
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
                    embeddings = get_bert_embeddings(texts)
                    
                    records = {'id': df_process['id'].tolist(), 'embedding': [emb.tolist() for emb in embeddings]}
                    if has_label: records['label'] = df_process['label'].tolist()
                    
                    table = pa.Table.from_pydict(records, schema=output_schema)
                    writer.write_table(table)
                    
                pbar.update(rows_in_slice)
                current_row = batch_end_row
                if current_row >= end_row: break
                
    print(f"Job {job_id} Complete: Saved results to {output_path}")

def combine_parquet_files(input_dir: str, output_path: str):
    """Combines all temporary chunk_*.parquet files into a single file."""
    print(f"\n--- Combining temporary Parquet files from {input_dir} ---")
    temp_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.startswith('chunk_') and f.endswith('.parquet')]
    if not temp_files:
        print("No temporary files found to combine. Exiting.")
        return
        
    temp_files.sort(key=lambda f: int(re.search(r'chunk_(\d+)\.parquet', f).group(1)))

    master_schema = pq.read_schema(temp_files[0])
    print(f"Writing combined file to: {output_path}")
    with pq.ParquetWriter(output_path, master_schema, compression='snappy') as writer:
        for file_path in tqdm(temp_files, desc="Combining chunks"):
            writer.write_table(pq.read_table(file_path))
    print(f"Successfully combined {len(temp_files)} files into {output_path}.")
    print(f"You can now safely delete the temporary directory: {input_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate or combine embeddings for text nodes using BERT.") # Changed desc
    parser.add_argument("--data_path", type=str, required=True, help="Path to the graph_data directory for the task.")
    parser.add_argument("--mode", type=str, required=True, choices=['compute', 'combine', 'count'], help="Operation to perform.")
    parser.add_argument("--start_row", type=int, help="Starting row index for 'compute' mode.")
    parser.add_argument("--end_row", type=int, help="Ending row index for 'compute' mode.")
    parser.add_argument("--job_id", type=int, help="Unique ID for this job/chunk in 'compute' mode.")
    args = parser.parse_args()

    # --- Derive paths based on the --data_path argument ---
    DATA_PATH = args.data_path
    TEMP_EMBEDDINGS_DIR = os.path.join(DATA_PATH, "temp_text_embeddings_bert") 
    
    # Input file is the same
    TEXT_NODES_PARQUET_PATH = os.path.join(DATA_PATH, "all_text_nodes.parquet")
    
    TEXT_EMBEDDINGS_FINAL_PATH = os.path.join(DATA_PATH, "text_node_embeddings_bert.parquet")

    if not os.path.exists(TEXT_NODES_PARQUET_PATH) and args.mode != 'combine':
        print(f"Error: Input file not found at {TEXT_NODES_PARQUET_PATH}")
        exit(1)

    if args.mode == 'count':
        total_rows = get_total_rows(TEXT_NODES_PARQUET_PATH)
        print(total_rows) # This stdout is captured by the launcher script
        
    elif args.mode == 'compute':
        if args.start_row is None or args.end_row is None or args.job_id is None:
            parser.error("--start_row, --end_row, and --job_id are required for 'compute' mode.")
        compute_node_embeddings_chunk(args.start_row, args.end_row, args.job_id, TEXT_NODES_PARQUET_PATH, TEMP_EMBEDDINGS_DIR)
        
    elif args.mode == 'combine':
        combine_parquet_files(TEMP_EMBEDDINGS_DIR, TEXT_EMBEDDINGS_FINAL_PATH)