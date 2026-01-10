import pandas as pd
import pyarrow.parquet as pq
import os
import torch
from torch_geometric.data import HeteroData, Dataset
import numpy as np
from typing import Dict, List
import random

# --- Configuration ---
DATA_PATH = "../GASTON/pretraining/graph_data/"
PROCESSED_GRAPH_PATH = "contrastive_init_heterogenous_graph_Gemma.pt"
CONTR_INIT_SUFFIX = ""
SAMPLED_SUFFIX = ""
TEXT_EMBEDDING_FILENAME = "text_node_embeddings_gemma.parquet"

RANDOM_SEED = 42

# --- Define specific input files based on the sampling ---
MASTER_USER_IDS_FILE = f"contr_init_user_ids_master{CONTR_INIT_SUFFIX}.parquet"
MASTER_COMM_IDS_FILE = f"contr_init_community_ids_master{CONTR_INIT_SUFFIX}.parquet"
CONTR_INIT_COMM_EMBEDS_FILE = f"community_node_embeddings_contrastive{CONTR_INIT_SUFFIX}.pt"
USER_COMM_EDGES_FILE = f"user_community_active_edges{SAMPLED_SUFFIX}.parquet"

COLUMN_NAMES = {
    'text_nodes': {'id': 'id'},
    'user_nodes': {'id': 'id'},
    'comm_nodes': {'id': 'id'},
    'text_user_edges': {'source': 'source_id', 'target': 'target_id', 'type': 'edge_type'},
    'text_comm_edges': {'source': 'source_id', 'target': 'target_id'},
    'user_comm_edges': {'source': 'source_id', 'target': 'target_id'}
    #'user_comm_edges': {'source': 'user_id', 'target': 'community_id'}
}

# Dynamic Suffix Detection
suffix = "_single"
if os.path.exists(os.path.join(DATA_PATH, "contr_init_user_ids_master_combined.parquet")):
    suffix = "_combined"

class ContrastivePretrainGraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
    
    @property
    def raw_file_names(self):
        return [TEXT_EMBEDDING_FILENAME, f"contr_init_user_ids_master{suffix}.parquet", f"contr_init_community_ids_master{suffix}.parquet", 
                f"community_node_embeddings_contrastive{suffix}.pt", "text_user_edges.parquet", "text_community_edges.parquet", "user_community_active_edges.parquet"]

    @property
    def processed_file_names(self):
        return [PROCESSED_GRAPH_PATH]

    def download(self):
        pass

    def process(self):
        print("--- Starting Contrastive Pre-training Graph Construction ---")

        # --- Pass 1: Load files and create mappings ---
        print("Pass 1: Loading files and building ID mappings...")
        
        text_nodes_df = pq.read_table(os.path.join(self.root, TEXT_EMBEDDING_FILENAME)).to_pandas()
        
        user_nodes_df = pq.read_table(os.path.join(self.root, MASTER_USER_IDS_FILE)).to_pandas()
        comm_nodes_df = pq.read_table(os.path.join(self.root, MASTER_COMM_IDS_FILE)).to_pandas()
        text_user_edges_df = pq.read_table(os.path.join(self.root, "text_user_edges.parquet")).to_pandas()
        text_comm_edges_df = pq.read_table(os.path.join(self.root, "text_community_edges.parquet")).to_pandas()

        print("[DEBUG] Normalizing IDs to lowercase...")
        
        # Normalize User IDs
        user_id_col = COLUMN_NAMES['user_nodes']['id']
        user_nodes_df[user_id_col] = user_nodes_df[user_id_col].astype(str).str.lower()
        user_nodes_df = user_nodes_df.drop_duplicates(subset=[user_id_col]) # Keep first
        print(f"[DEBUG] Normalized master user list. New count: {len(user_nodes_df):,}")

        text_user_target_col = COLUMN_NAMES['text_user_edges']['target']
        text_user_edges_df[text_user_target_col] = text_user_edges_df[text_user_target_col].astype(str).str.lower()

        # Normalize Community IDs
        comm_id_col = COLUMN_NAMES['comm_nodes']['id']
        comm_nodes_df[comm_id_col] = comm_nodes_df[comm_id_col].astype(str).str.lower()
        # IMPORTANT: We cannot drop duplicates here if it misaligns with the feature tensor.
        # We build the map assuming the order is tied to the .pt file.
        # Duplicates will just map to the *last* index, which is fine.
        print(f"[DEBUG] Normalizing master community list (preserving count for features): {len(comm_nodes_df):,}")
        
        text_comm_target_col = COLUMN_NAMES['text_comm_edges']['target']
        text_comm_edges_df[text_comm_target_col] = text_comm_edges_df[text_comm_target_col].astype(str).str.lower()

        valid_text_ids = set(text_user_edges_df[COLUMN_NAMES['text_user_edges']['source']]).union(
                         set(text_comm_edges_df[COLUMN_NAMES['text_comm_edges']['source']]))
        print(f"[DEBUG] Found {len(valid_text_ids):,} unique text IDs with edges.")
        
        print(f"[DEBUG] Filtering text nodes. Initial shape: {text_nodes_df.shape}")
        text_nodes_df = text_nodes_df[text_nodes_df[COLUMN_NAMES['text_nodes']['id']].isin(valid_text_ids)]
        print(f"[DEBUG] Filtered text nodes shape: {text_nodes_df.shape}")
        if text_nodes_df.empty:
            print("CRITICAL WARNING: No text nodes remain after filtering. Graph will be empty.")

        text_id_to_idx = {id_: i for i, id_ in enumerate(text_nodes_df[COLUMN_NAMES['text_nodes']['id']])}
        user_id_to_idx = {id_: i for i, id_ in enumerate(user_nodes_df[user_id_col])}
        comm_id_to_idx = {id_: i for i, id_ in enumerate(comm_nodes_df[comm_id_col])}
        
        print(f"Nodes retained: Text={len(text_id_to_idx):,}, User={len(user_id_to_idx):,}, Community={len(comm_id_to_idx):,}")
        
        # --- Pass 2: Construct HeteroData object ---
        print("\nPass 2: Building HeteroData object...")
        data = HeteroData()
        data['text'].x = torch.tensor(np.vstack(text_nodes_df['embedding'].apply(np.array).values), dtype=torch.float32)
        print(f"[DEBUG] data['text'].x shape: {data['text'].x.shape}")
        
        comm_features_path = os.path.join(self.root, CONTR_INIT_COMM_EMBEDS_FILE)
        print(f"[DEBUG] Loading community features from: {comm_features_path}")
        comm_features = torch.load(comm_features_path, weights_only=False).cpu()
        data['community'].x = comm_features
        print(f"[DEBUG] data['community'].x shape: {data['community'].x.shape}")
        
        # Check for feature/ID misalignment after normalization
        if len(comm_nodes_df) != data['community'].x.shape[0]:
             print(f"\n\nCRITICAL WARNING: Mismatch in community data!")
             print(f"  Master community ID list has {len(comm_nodes_df):,} entries.")
             print(f"  Community feature tensor has {data['community'].x.shape[0]:,} rows.")
             print(f"  This will likely cause a crash or silent errors.")
             print(f"  This can happen if lowercasing {MASTER_COMM_IDS_FILE} and dropping duplicates")
             print(f"  creates a different number of rows than in {CONTR_INIT_COMM_EMBEDS_FILE}.")
             # We proceed, but this is a major warning.
             # Let's adjust num_nodes to match the ID list, though this is risky.
             data['community'].num_nodes = len(comm_id_to_idx)
        
        data['user'].num_nodes = len(user_id_to_idx)
        # Ensure user features match community features dim
        feat_dim = data['community'].x.size(1) if data['community'].x is not None else 768
        data['user'].x = torch.zeros((len(user_id_to_idx), feat_dim), dtype=torch.float32)
        print(f"[DEBUG] data['user'].x shape (initialized as zeros): {data['user'].x.shape}")

        def load_edges(df, src_map, dst_map, src_col, dst_col, edge_name=""):
            print(f"[DEBUG] load_edges ({edge_name}): Mapping {len(df):,} raw edges...")
            
            initial_count = len(df)
            if initial_count == 0:
                print(f"WARNING: No edges found for {edge_name} in the input DataFrame.")
                return torch.empty((2, 0), dtype=torch.long)
                
            src_idx = df[src_col].map(src_map)
            dst_idx = df[dst_col].map(dst_map)
            
            src_na_mask = src_idx.isna()
            dst_na_mask = dst_idx.isna()
            
            num_src_na = src_na_mask.sum()
            num_dst_na = dst_na_mask.sum()
            
            # This mask finds rows where *either* mapping failed
            combined_na_mask = src_na_mask | dst_na_mask
            num_total_invalid = combined_na_mask.sum()
            
            # Only print if there are invalid edges
            if num_total_invalid > 0:
                print(f"[DEBUG] load_edges ({edge_name}): Breakdown of invalid edges:")
                print(f"  -> {num_src_na:10,} edges dropped due to (source_id: '{src_col}') not in master list.")
                print(f"  -> {num_dst_na:10,} edges dropped due to (target_id: '{dst_col}') not in master list.")
                print(f"  -> {num_total_invalid:10,} total rows dropped (at least one ID was invalid).")

            mask = ~combined_na_mask # Invert the NA mask to get *valid* edges
            num_valid = mask.sum()
            
            print(f"[DEBUG] load_edges ({edge_name}): Found {num_valid:,} valid edges (out of {initial_count:,}).")
            
            if num_valid == 0:
                print(f"WARNING: No valid edges found for {edge_name} after mapping.")
                return torch.empty((2, 0), dtype=torch.long)
                
            return torch.tensor([src_idx[mask].astype(int).values, dst_idx[mask].astype(int).values], dtype=torch.long)
        
        if COLUMN_NAMES['text_user_edges']['type'] in text_user_edges_df.columns:
            print(f"[DEBUG] Filtering 'replies_to' from text_user_edges. Initial shape: {text_user_edges_df.shape}")
            text_user_edges_df = text_user_edges_df[text_user_edges_df[COLUMN_NAMES['text_user_edges']['type']] != 'replies_to']
            print(f"[DEBUG] Filtered text_user_edges shape: {text_user_edges_df.shape}")
        
        data['text', 'post_in', 'community'].edge_index = load_edges(
            text_comm_edges_df, text_id_to_idx, comm_id_to_idx, 
            COLUMN_NAMES['text_comm_edges']['source'], COLUMN_NAMES['text_comm_edges']['target'],
            edge_name="text->comm")
        
        data['text', 'post_by', 'user'].edge_index = load_edges(
            text_user_edges_df, text_id_to_idx, user_id_to_idx, 
            COLUMN_NAMES['text_user_edges']['source'], COLUMN_NAMES['text_user_edges']['target'],
            edge_name="text->user")
        
        user_comm_edges_df = pq.read_table(os.path.join(self.root, USER_COMM_EDGES_FILE)).to_pandas()
        
        user_comm_src_col = COLUMN_NAMES['user_comm_edges']['source']
        user_comm_tgt_col = COLUMN_NAMES['user_comm_edges']['target']
        
        print(f"[DEBUG] Normalizing {len(user_comm_edges_df):,} user->comm edges to lowercase...")
        user_comm_edges_df[user_comm_src_col] = user_comm_edges_df[user_comm_src_col].astype(str).str.lower()
        user_comm_edges_df[user_comm_tgt_col] = user_comm_edges_df[user_comm_tgt_col].astype(str).str.lower()

        data['user', 'active', 'community'].edge_index = load_edges(
            user_comm_edges_df, user_id_to_idx, comm_id_to_idx, 
            user_comm_src_col, user_comm_tgt_col,
            edge_name="user->comm")

        # --- Pass 3: Generate aggregation index map ---
        print("\nPass 3: Generating aggregation index map...")
        # We use the already-loaded and normalized user_comm_edges_df from Pass 2
        user_indices = user_comm_edges_df[user_comm_src_col].map(user_id_to_idx)
        comm_indices = user_comm_edges_df[user_comm_tgt_col].map(comm_id_to_idx)
        print(f"[DEBUG] Agg map: Total user-comm pairs processed: {len(user_comm_edges_df)}")
        mask = user_indices.notna() & comm_indices.notna()
        print(f"[DEBUG] Agg map: Valid pairs (in all maps): {mask.sum():,}")
        agg_map = {'user_indices': torch.tensor(user_indices[mask].astype(int).values), 'community_indices': torch.tensor(comm_indices[mask].astype(int).values)}
        torch.save(agg_map, os.path.join(self.root, "user_community_agg_index_Gemma.pt"))
        print("Aggregation index map saved.")
                
        print("\n--- [DEBUG CHECK] Verifying User-Community Edge Coverage ---")
        all_user_indices_set = set(range(data['user'].num_nodes))
        
        users_with_comm_edges = torch.empty(0, dtype=torch.long)
        if data['user', 'active', 'community'].edge_index.numel() > 0:
            users_with_comm_edges = torch.unique(data['user', 'active', 'community'].edge_index[0])
            
        users_with_comm_edges_set = set(users_with_comm_edges.cpu().numpy())
        
        num_users_total = len(all_user_indices_set)
        num_users_with_edges = len(users_with_comm_edges_set)
        
        missing_user_indices = all_user_indices_set - users_with_comm_edges_set
        num_missing = len(missing_user_indices)

        print(f"[DEBUG CHECK] Total users in graph: {num_users_total:,}")
        print(f"[DEBUG CHECK] Users with at least one community edge: {num_users_with_edges:,}")
        
        if num_missing == 0:
            print("  -> \033[92mSUCCESS:\033[0m All users have at least one community edge.")
        else:
            print(f"  -> \033[91mCRITICAL WARNING:\033[0m {num_missing:,} users have NO community edges and will not be initialized.")
            if num_missing < 10:
                print(f"     Missing user indices: {list(missing_user_indices)}")
            else:
                print(f"     First 10 missing user indices: {list(missing_user_indices)[:10]}")
        print("---------------------------------------------------------")

        print("\n--- [DEBUG CHECK] Verifying Text-Community Edge Coverage ---")
        
        # Get all text nodes
        num_text_total = data['text'].num_nodes
        all_text_indices_set = set(range(num_text_total))
        
        text_with_comm_edges = torch.empty(0, dtype=torch.long)
        
        # Find all text nodes that have a 'post_in' edge
        if ('text', 'post_in', 'community') in data.edge_types and data['text', 'post_in', 'community'].edge_index.numel() > 0:
            text_with_comm_edges = torch.unique(data['text', 'post_in', 'community'].edge_index[0]) # [0] is the text node
        else:
            print("[DEBUG CHECK] No ('text', 'post_in', 'community') edges found.")
            
        text_with_comm_edges_set = set(text_with_comm_edges.cpu().numpy())
        num_text_with_edges = len(text_with_comm_edges_set)
        
        # Find the difference
        missing_text_indices = all_text_indices_set - text_with_comm_edges_set
        num_missing = len(missing_text_indices)

        print(f"[DEBUG CHECK] Total text nodes in graph: {num_text_total:,}")
        print(f"[DEBUG CHECK] Text nodes with at least one community edge: {num_text_with_edges:,}")
        
        if num_missing == 0:
            print("  -> \033[92mSUCCESS:\033[0m All text nodes have at least one community edge.")
        else:
            print(f"  -> \033[93mWARNING:\033[0m {num_missing:,} text nodes have NO community edges.")
            if num_missing < 10:
                print(f"     Missing text indices (examples): {list(missing_text_indices)}")
            else:
                print(f"     First 10 missing text indices (examples): {list(missing_text_indices)[:10]}")
        print("---------------------------------------------------------")

        print("\n--- Final Graph Structure ---")
        print(data)
        for node_type in data.node_types:
            if hasattr(data[node_type], 'x') and data[node_type].x is not None:
                print(f"  {node_type} features shape: {data[node_type].x.shape}")
            else:
                print(f"  {node_type} has no .x features or is None.")
        for edge_type in data.edge_types:
            print(f"  {edge_type} edges: {data[edge_type].edge_index.shape[1]:,}")
        print("---------------------------------")
        
        print("\n--- Graph Construction Complete ---")
        try:
            torch.save(data, self.processed_paths[0])
            print(f"[DEBUG] Graph saved successfully to: {self.processed_paths[0]}")
        except Exception as e:
             print(f"CRITICAL ERROR: Failed to save final graph to {self.processed_paths[0]}. Error: {e}")

# Main execution
if __name__ == '__main__':
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    print(f"--- Random seed set to: {RANDOM_SEED} ---")
    dataset = ContrastivePretrainGraphDataset(root=DATA_PATH)