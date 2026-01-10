import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import numpy as np
from tqdm import tqdm
import sys
from collections import defaultdict
import argparse

# --- Configuration ---
EMBEDDING_DIM = None 

# Column Names
TEXT_SOURCE_ID_COLUMN = 'source_id'
COMMUNITY_TARGET_ID_COLUMN = 'target_id'
USER_TARGET_ID_COLUMN = 'target_id'

def compute_community_embeddings(text_emb_path: str, text_comm_path: str, community_nodes_path: str, output_path: str) -> dict:
    print("--- Starting Phase 1: Compute Community Embeddings ---")
    global EMBEDDING_DIM
    try:
        text_community_edges_df = pq.read_table(text_comm_path).to_pandas()
        text_community_edges_df[COMMUNITY_TARGET_ID_COLUMN] = text_community_edges_df[COMMUNITY_TARGET_ID_COLUMN].astype(str).str.lower()
        text_community_edges_df[TEXT_SOURCE_ID_COLUMN] = text_community_edges_df[TEXT_SOURCE_ID_COLUMN].astype(str)

        parquet_file = pq.ParquetFile(text_emb_path)
        first_embedding = parquet_file.read_row_group(0, columns=['embedding']).to_pydict()['embedding'][0]
        EMBEDDING_DIM = len(first_embedding)
        print(f"Automatically detected Embedding Dimension: {EMBEDDING_DIM}")

        community_aggs = defaultdict(lambda: {'sum': np.zeros(EMBEDDING_DIM, dtype=np.float32), 'count': 0})
        
        required_text_ids = set(text_community_edges_df[TEXT_SOURCE_ID_COLUMN])
        
        text_embeddings_for_comms = {}
        with tqdm(total=parquet_file.metadata.num_rows, desc="Loading relevant text embeddings") as pbar:
            for batch in parquet_file.iter_batches(batch_size=100_000, columns=['id', 'embedding']):
                chunk_df = batch.to_pandas()
                chunk_df['id'] = chunk_df['id'].astype(str)
                
                relevant_rows = chunk_df[chunk_df['id'].isin(required_text_ids)]
                for _, row in relevant_rows.iterrows():
                    text_embeddings_for_comms[row['id']] = np.array(row['embedding'], dtype=np.float32)
                pbar.update(len(chunk_df))

        print("Aggregating embeddings for communities...")
        for _, edge in tqdm(text_community_edges_df.iterrows(), total=len(text_community_edges_df), desc="Aggregating communities"):
            text_id = edge[TEXT_SOURCE_ID_COLUMN]
            comm_id = edge[COMMUNITY_TARGET_ID_COLUMN]
            if text_id in text_embeddings_for_comms:
                community_aggs[comm_id]['sum'] += text_embeddings_for_comms[text_id]
                community_aggs[comm_id]['count'] += 1
        
        final_community_embeddings = {cid: agg['sum'] / agg['count'] for cid, agg in community_aggs.items() if agg['count'] > 0}
        
        # community IDs (cid) are already lowercase
        new_embeddings_df = pd.DataFrame(final_community_embeddings.items(), columns=['id', 'embedding_calculated'])
        
        original_community_nodes_df = pq.read_table(community_nodes_path).to_pandas()
        if 'community_id' in original_community_nodes_df.columns:
            original_community_nodes_df.rename(columns={'community_id': 'id'}, inplace=True)

        original_community_nodes_df['id'] = original_community_nodes_df['id'].astype(str).str.lower()

        merged_df = pd.merge(original_community_nodes_df, new_embeddings_df, on='id', how='left')
        
        def create_final_embedding(row):
            if isinstance(row['embedding_calculated'], np.ndarray):
                return row['embedding_calculated'].tolist()
            else:
                return np.zeros(EMBEDDING_DIM, dtype=np.float32).tolist()
        merged_df['embedding'] = merged_df.apply(create_final_embedding, axis=1)
        
        final_columns = ['id', 'embedding']
        if 'label' in merged_df.columns: final_columns.append('label')
        if 'label_2' in merged_df.columns: final_columns.append('label_2')
        final_df_to_save = merged_df[final_columns]

        final_schema = pa.Table.from_pandas(final_df_to_save.head(1)).schema
        pq.write_table(pa.Table.from_pandas(final_df_to_save, schema=final_schema), output_path)
        
        print(f"Community embeddings saved. Total: {len(final_df_to_save):,}")
        
        return {row['id']: np.array(row['embedding']) for _, row in final_df_to_save.iterrows()}
    
    except Exception as e:
        print(f"An error occurred in Phase 1: {e}", file=sys.stderr)
        sys.exit(1)

def process_users_in_chunks(final_community_embeddings: dict, all_users_path, text_user_edges_path, user_comm_edges_path, text_emb_path, output_path):
    """
    Phase 2: Processes all users in manageable chunks to calculate and save their
    average embeddings, preventing out-of-memory errors.
    """
    print("\n--- Starting Phase 2: Process User Embeddings in Chunks ---")
    
    try:
        all_users_df = pq.read_table(all_users_path).to_pandas()
        all_user_ids = all_users_df.iloc[:, 0].astype(str).tolist()

        USER_CHUNK_SIZE = 100_000
        USER_EMBEDDINGS_SCHEMA = pa.schema([('id', pa.string()), ('embedding', pa.list_(pa.float32()))])
        
        if os.path.exists(output_path):
            os.remove(output_path)

        with pq.ParquetWriter(output_path, USER_EMBEDDINGS_SCHEMA, compression='snappy') as writer:
            for i in tqdm(range(0, len(all_user_ids), USER_CHUNK_SIZE), desc="Processing user chunks"):
                user_chunk_ids = set(all_user_ids[i:i + USER_CHUNK_SIZE])
                
                user_to_texts_chunk = defaultdict(list)
                user_to_communities_chunk = defaultdict(list)
                
                # Stream through text-user edges to build mapping for the chunk
                for batch in pq.ParquetFile(text_user_edges_path).iter_batches(batch_size=500_000, columns=[TEXT_SOURCE_ID_COLUMN, USER_TARGET_ID_COLUMN]):
                    df = batch.to_pandas()
                    df[USER_TARGET_ID_COLUMN] = df[USER_TARGET_ID_COLUMN].astype(str)
                    df[TEXT_SOURCE_ID_COLUMN] = df[TEXT_SOURCE_ID_COLUMN].astype(str)
                    
                    relevant_edges = df[df[USER_TARGET_ID_COLUMN].isin(user_chunk_ids)]
                    for _, edge in relevant_edges.iterrows():
                        user_to_texts_chunk[edge[USER_TARGET_ID_COLUMN]].append(edge[TEXT_SOURCE_ID_COLUMN])

                # Stream through user-community edges to build mapping for the chunk
                for batch in pq.ParquetFile(user_comm_edges_path).iter_batches(batch_size=500_000, columns=['source_id', 'target_id']):
                    df = batch.to_pandas()
                    df['source_id'] = df['source_id'].astype(str)
                    df['target_id'] = df['target_id'].astype(str).str.lower()
                    
                    relevant_edges = df[df['source_id'].isin(user_chunk_ids)]
                    for _, edge in relevant_edges.iterrows():
                        user_to_communities_chunk[edge['source_id']].append(edge['target_id'])
                
                chunk_text_ids_needed = {text_id for user_texts in user_to_texts_chunk.values() for text_id in user_texts}
                
                text_embeddings_for_chunk = {}
                if chunk_text_ids_needed:
                    for batch in pq.ParquetFile(text_emb_path).iter_batches(batch_size=100_000, columns=['id', 'embedding']):
                        df = batch.to_pandas()
                        df['id'] = df['id'].astype(str)
                        
                        relevant_rows = df[df['id'].isin(chunk_text_ids_needed)]
                        for _, row in relevant_rows.iterrows():
                            text_embeddings_for_chunk[row['id']] = np.array(row['embedding'], dtype=np.float32)

                chunk_final_embeddings = {}
                for user_id in user_chunk_ids:
                    if user_id in user_to_texts_chunk:
                        embeddings_to_avg = [text_embeddings_for_chunk[tid] for tid in user_to_texts_chunk[user_id] if tid in text_embeddings_for_chunk]
                        if embeddings_to_avg:
                            chunk_final_embeddings[user_id] = np.mean(embeddings_to_avg, axis=0)
                        else:
                            chunk_final_embeddings[user_id] = np.zeros(EMBEDDING_DIM, dtype=np.float32)
                    else:
                        connected_comms = user_to_communities_chunk.get(user_id, [])
                        # This comparison is now case-insensitive
                        comm_embeddings_to_avg = [final_community_embeddings[cid] for cid in connected_comms if cid in final_community_embeddings]
                        if comm_embeddings_to_avg:
                            chunk_final_embeddings[user_id] = np.mean(comm_embeddings_to_avg, axis=0)
                        else:
                            chunk_final_embeddings[user_id] = np.zeros(EMBEDDING_DIM, dtype=np.float32)

                chunk_data = [{'id': uid, 'embedding': emb.tolist()} for uid, emb in chunk_final_embeddings.items()]
                if chunk_data:
                    table = pa.Table.from_pylist(chunk_data, schema=USER_EMBEDDINGS_SCHEMA)
                    writer.write_table(table)
        
        print(f"\nFinal user embeddings saved to {output_path}")

    except Exception as e:
        print(f"An error occurred in Phase 2: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Generate avg_init embeddings for a given dataset.")
    parser.add_argument("--data_path", required=True, help="The path to the graph_data directory for the target dataset.")
    args = parser.parse_args()

    DATA_PATH = args.data_path
    
    TEXT_EMBEDDINGS_PARQUET_PATH = os.path.join(DATA_PATH, "text_node_embeddings_gemma.parquet")
    TEXT_COMMUNITY_EDGES_PARQUET_PATH = os.path.join(DATA_PATH, "text_community_edges.parquet")
    TEXT_USER_EDGES_PARQUET_PATH = os.path.join(DATA_PATH, "text_user_edges.parquet")
    USER_COMMUNITY_ACTIVE_EDGES_PARQUET_PATH = os.path.join(DATA_PATH, "user_community_active_edges.parquet")
    ALL_USERS_PARQUET_PATH = os.path.join(DATA_PATH, "unique_user_nodes.parquet") 
    COMMUNITY_EMBEDDINGS_PARQUET_PATH = os.path.join(DATA_PATH, "avg_community_node_embeddings_gemma.parquet")
    USER_EMBEDDINGS_PARQUET_PATH = os.path.join(DATA_PATH, "avg_user_node_embeddings_gemma.parquet")
    ORIGINAL_COMMUNITY_NODES_PATH = os.path.join(DATA_PATH, "unique_community_nodes.parquet")

    final_community_embeddings = compute_community_embeddings(
        TEXT_EMBEDDINGS_PARQUET_PATH, 
        TEXT_COMMUNITY_EDGES_PARQUET_PATH,
        ORIGINAL_COMMUNITY_NODES_PATH,
        COMMUNITY_EMBEDDINGS_PARQUET_PATH
    )
    process_users_in_chunks(
        final_community_embeddings,
        ALL_USERS_PARQUET_PATH,
        TEXT_USER_EDGES_PARQUET_PATH,
        USER_COMMUNITY_ACTIVE_EDGES_PARQUET_PATH,
        TEXT_EMBEDDINGS_PARQUET_PATH,
        USER_EMBEDDINGS_PARQUET_PATH
    )
    print("\n--- GEMMA-BASED AVG EMBEDDING GENERATION COMPLETE ---")

if __name__ == "__main__":
    main()