import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import torch
from torch_geometric.data import HeteroData, Dataset
import numpy as np
import collections
from typing import List, Dict, Tuple
from tqdm import tqdm
import argparse
from collections import defaultdict

# --- Configuration ---
MISSING_USER_PLACEHOLDERS = ["__missing_user__", "__missing__"]
UNLABELED_SENTINEL = -999.0

# --- Helper Functions ---
def get_path(root: str, filename: str) -> str:
    return os.path.join(root, filename)

def determine_node_columns(df: pd.DataFrame) -> Tuple[str, bool]:
    file_columns = df.columns.tolist()
    if 'id' in file_columns: id_col = 'id'
    elif 'post_id' in file_columns: id_col = 'post_id'
    else: raise ValueError(f"Node feature file must contain an ID column. Columns found: {file_columns}")
    return id_col, 'label' in file_columns

def print_label_distribution(data: HeteroData, task: str, unlabeled_sentinel: float):
    """Prints a sanity check summary of the node labels based on the task."""
    print(f"\n--- Label Sanity Check (TASK: {task.upper()}) ---")

    if task == 'rspct':
        if 'community' in data.node_types and hasattr(data['community'], 'y'):
            labels = data['community'].y
            counts = collections.Counter(labels)
            print("Community-Level Label Counts:")
            for label, count in counts.most_common(10): # Show top 10 for brevity
                print(f"  - {label}: {count:,}")
        else:
            print("No labels found on community nodes for RSPCT task.")
        return

    if 'text' in data.node_types and hasattr(data['text'], 'y'):
        labels = data['text'].y
        
        if task in ["dreaddit", "normvio", "hateful"]: 
            if isinstance(labels, torch.Tensor) and labels.dtype == torch.long:
                label_counts = torch.bincount(labels)
                print("Classification Label Counts:")
                for i, count in enumerate(label_counts.tolist()):
                    print(f"  Class {i}: {count:,}")
            else:
                print(f"Label data type found but not compatible for classification check: {type(labels)}.")
                
        elif task == "RMHD":
            if isinstance(labels, list):
                counts = collections.Counter(labels)
                print(f"Multi-Class/String Label Counts (Top {min(5, len(counts))}/Total {len(counts)}):")
                for label, count in counts.most_common(5):
                    print(f"  - '{label}': {count:,}")
            else:
                print(f"Label data type found but not compatible for RMHD check: {type(labels)}.")

        elif task == "ruddit":
            if isinstance(labels, torch.Tensor) and (labels.dtype == torch.float32 or labels.dtype == torch.float64):
                unlabeled_count = (labels == unlabeled_sentinel).sum().item()
                labeled_data = labels[labels != unlabeled_sentinel]
                
                print("Regression Label Summary:")
                print(f"  Labeled Samples: {labeled_data.size(0):,}")
                print(f"  Unlabeled Samples (Sentinel): {unlabeled_count:,}")
                if labeled_data.size(0) > 0:
                    print(f"  Mean: {labeled_data.mean().item():.4f}")
                    print(f"  Std Dev: {labeled_data.std().item():.4f}")
                    print(f"  Min/Max: {labeled_data.min().item():.4f} / {labeled_data.max().item():.4f}")
            else:
                print(f"Label data type found but not compatible for regression check: {type(labels)}.")
        
        else:
             print(f"Labels found, but TASK='{task}' is unrecognized. Skipping detailed distribution check.")
    else:
        print("data['text'].y or data['community'].y attribute not found. No labels were loaded for the graph.")

# --- Heterogeneous Graph Dataset Class ---
class ContrastiveDownstreamGraphDataset(Dataset):
    def __init__(self, root, graph_type: str, task: str, output_filename: str, pretrain_data_path: str, pretrain_model_path: str, transform=None, pre_transform=None):
        self.pretrain_data_path = pretrain_data_path
        self.graph_type = graph_type
        self.task = task
        self.output_filename = output_filename
        self.unlabeled_sentinel = UNLABELED_SENTINEL
        self.pretrain_model_path = pretrain_model_path
        super().__init__(root, transform, pre_transform)
    
    @property
    def raw_dir(self) -> str: return self.root
    @property
    def processed_dir(self) -> str: return os.path.join(self.root, 'processed')
    @property
    def raw_file_names(self): return []
    @property
    def processed_file_names(self): return [self.output_filename]
    def download(self): pass

    def process(self):
        print(f"--- Starting Contr-Init Downstream Graph Construction (TASK: {self.task.upper()}) ---")
        
        # --- PASS 1: Load, Filter, and Map ---
        print("Pass 1: Loading data, filtering by pre-trained communities, and building ID mappings...")
        
        master_comm_ids_df = pq.read_table(get_path(self.pretrain_data_path, "contr_init_community_ids_master.parquet")).to_pandas()
        valid_pretrain_communities = set(master_comm_ids_df.iloc[:, 0].str.lower())
        print(f"Loaded {len(valid_pretrain_communities):,} master communities from pre-training.")

        text_comm_edges_df = pq.read_table(get_path(self.root, "text_community_edges.parquet")).to_pandas()
        text_comm_edges_df['source_id'] = text_comm_edges_df['source_id'].astype(str).str.lower()
        user_comm_edges_df = pq.read_table(get_path(self.root, "user_community_active_edges.parquet")).to_pandas()
        user_comm_edges_df['source_id'] = user_comm_edges_df['source_id'].astype(str).str.lower()

        if self.task == 'rspct':
            print("[INFO RSPCT] Defining downstream communities *only* from 'unique_community_nodes.parquet' (label file).")
            label_file_path = get_path(self.root, "unique_community_nodes.parquet")
            label_comm_df = pq.read_table(label_file_path, columns=['community_id']).to_pandas()
            active_downstream_communities = set(label_comm_df['community_id'].str.lower())
            print(f"Found {len(active_downstream_communities):,} total labeled communities for RSPCT task (normalized to lowercase).")
            text_comm_edges_df['target_id'] = text_comm_edges_df['target_id'].str.lower()
            user_comm_edges_df['target_id'] = user_comm_edges_df['target_id'].str.lower()
        else:
            print("[INFO] Defining downstream communities from text and user edge files.")
            text_comm_edges_df['target_id'] = text_comm_edges_df['target_id'].str.lower()
            user_comm_edges_df['target_id'] = user_comm_edges_df['target_id'].str.lower()
            active_downstream_communities = set(text_comm_edges_df['target_id']).union(set(user_comm_edges_df['target_id']))
            print(f"Found {len(active_downstream_communities):,} active communities in the downstream task (normalized to lowercase).")

        final_community_set = active_downstream_communities.intersection(valid_pretrain_communities)
        print(f"Found {len(final_community_set):,} communities in both the task and pre-trained list.")

        text_comm_edges_df = text_comm_edges_df[text_comm_edges_df['target_id'].isin(final_community_set)]
        user_comm_edges_df = user_comm_edges_df[user_comm_edges_df['target_id'].isin(final_community_set)]
        valid_text_ids_from_edges = set(text_comm_edges_df['source_id'])
        print(f"Found {len(valid_text_ids_from_edges):,} valid text nodes from filtered text-community edges.")

        # Conditionally load text labels only if the task is NOT rspct
        text_parquet_file = get_path(self.root, "text_node_embeddings_xlnet.parquet")
        try:
            text_schema = pq.ParquetFile(text_parquet_file).schema.names
        except Exception as e:
            print(f"FATAL: Could not read schema from {text_parquet_file}: {e}")
            raise
            
        text_columns_to_read = ['id', 'embedding']
        
        if self.task == 'rspct':
            print("[INFO] Task is RSPCT. Will not look for 'label' column in text node file.")
        elif 'label' in text_schema:
            print(f"[INFO] Task is {self.task}. Found 'label' column in text node file, will load it.")
            text_columns_to_read.append('label')
        else:
            print(f"[WARN] Task is {self.task}, but 'label' column not found in {text_parquet_file}. Proceeding without text labels.")

        print(f"Loading all text nodes from {text_parquet_file}...")
        text_nodes_df = pq.read_table(text_parquet_file, columns=text_columns_to_read).to_pandas()
        text_nodes_df['id'] = text_nodes_df['id'].astype(str).str.lower()
        print(f"Loaded {len(text_nodes_df):,} total text nodes. Filtering by valid edge set...")

        # Filter text nodes to only include those that have a valid community edge
        text_nodes_df = text_nodes_df[text_nodes_df['id'].isin(valid_text_ids_from_edges)]
        text_nodes_df['id'] = text_nodes_df['id'].astype(str).str.lower()
        text_nodes_df = text_nodes_df.drop_duplicates(subset=['id'], keep='first')

        # --- User Definition and Filtering Logic ---
        user_nodes_df = pq.read_table(get_path(self.root, "user_node_embeddings_from_contr_init.parquet")).to_pandas()
        user_nodes_df['id'] = user_nodes_df['id'].astype(str).str.lower()
        user_nodes_df = user_nodes_df.drop_duplicates(subset=['id'], keep='first')
        print(f"Found {len(user_nodes_df):,} initial user embeddings for the downstream task.")

        # 1. Define the "universe" of users: all users with embeddings, minus placeholders
        all_potential_user_ids = set(user_nodes_df['id'])
        for placeholder in MISSING_USER_PLACEHOLDERS:
            if placeholder in all_potential_user_ids: all_potential_user_ids.remove(placeholder)
        print(f"Found {len(all_potential_user_ids):,} non-placeholder users with embeddings.")
        
        # 2. Load text-user edges and filter them by valid texts (as before)
        text_user_edges_df = pq.read_table(get_path(self.root, "text_user_edges.parquet")).to_pandas()
        text_user_edges_df['source_id'] = text_user_edges_df['source_id'].astype(str).str.lower()
        text_user_edges_df['target_id'] = text_user_edges_df['target_id'].astype(str).str.lower()
        valid_text_ids = set(text_nodes_df['id']) 
        text_user_edges_df = text_user_edges_df[text_user_edges_df['source_id'].isin(valid_text_ids)]

        # 3. Apply task-specific user filtering
        if self.task == 'recommendation':
            print("\n--- Applying User Sampling for Recommendation Task ---")
            
            # 3a. Identify users with text edges (who are also in our embedding universe)
            users_with_text_edges_all = set(text_user_edges_df['target_id'])
            users_with_text_edges = all_potential_user_ids.intersection(users_with_text_edges_all)
            num_users_with_text = len(users_with_text_edges)
            print(f"Found {num_users_with_text:,} users with text edges (who also have embeddings).")

            # 3b. Identify users without text edges (from our embedding universe)
            users_without_text_edges = all_potential_user_ids - users_with_text_edges
            num_users_without_text = len(users_without_text_edges)
            print(f"Found {num_users_without_text:,} users without text edges (who also have embeddings).")
            
            # 3c. Perform sampling if imbalanced
            if num_users_without_text > num_users_with_text and num_users_with_text > 0:
                print(f"Sampling {num_users_with_text:,} users from the {num_users_without_text:,} users without text edges...")
                users_without_text_list = list(users_without_text_edges)
                sampled_indices = np.random.choice(len(users_without_text_list), num_users_with_text, replace=False)
                sampled_users_without_text = {users_without_text_list[i] for i in sampled_indices}
                print("Sampling complete.")
                
                # 4. Combine sets
                final_user_set = users_with_text_edges.union(sampled_users_without_text)
                print(f"Final user set size after sampling: {len(final_user_set):,}")
            else:
                print("Number of users without text edges is <= users with text edges (or no users with text found). Keeping all potential users.")
                final_user_set = all_potential_user_ids

        else:
            # --- Original Logic (for non-recommendation tasks) ---
            print(f"[INFO] Using original active user logic for task '{self.task}'.")
            active_text_users = set(text_user_edges_df['target_id'])
            active_community_users = set(user_comm_edges_df['source_id'])
            active_users = active_text_users.union(active_community_users)
            
            # Intersect active users with the users who actually have embeddings
            final_user_set = all_potential_user_ids.intersection(active_users)
            print(f"Found {len(final_user_set):,} active users with embeddings.")
        
        # --- Finalize ID sets and create mappings ---
        all_text_ids = set(text_nodes_df['id']) 
        # Filter the user DataFrame based on the final set
        user_nodes_df = user_nodes_df[user_nodes_df['id'].isin(final_user_set)]
        # The final user IDs are from the *filtered* DataFrame
        all_user_ids = set(user_nodes_df['id']) 
        
        text_id_to_idx = {id_: i for i, id_ in enumerate(sorted(list(all_text_ids)))}
        user_id_to_idx = {id_: i for i, id_ in enumerate(sorted(list(all_user_ids)))}
        community_id_to_idx = {id_: i for i, id_ in enumerate(sorted(list(final_community_set)))}
        
        print(f"Final nodes for graph: Text={len(text_id_to_idx):,}, User={len(user_id_to_idx):,}, Community={len(community_id_to_idx):,}")
        
        data = HeteroData()
        
        # --- PASS 2: Load Node Features ---
        print("\nPass 2: Loading node features...")
        
        def stream_load_features(df_full, id_col, id_map, node_type):
            df = df_full[df_full[id_col].isin(id_map)].copy()
            df['idx'] = df[id_col].map(id_map)
            df.set_index('idx', inplace=True)
            df.sort_index(inplace=True)
            
            data[node_type].x = torch.from_numpy(np.vstack(df['embedding'].apply(np.array).values))
            print(f"Loaded {data[node_type].x.shape[0]} features for {node_type}.")
            
            if self.graph_type == 'downstream' and 'label' in df.columns:
                if self.task == 'rspct' and node_type == 'community':
                    print("--> Attaching labels to COMMUNITY nodes for RSPCT task.")
                    data[node_type].y = df['label'].tolist()
                    if 'label_2' in df.columns: data[node_type].y_2 = df['label_2'].tolist()
                elif node_type == 'text':
                    print(f"--> Attaching labels to TEXT nodes for {self.task.upper()} task.")
                    if self.task in ["normvio", "dreaddit", "hateful"]:
                        data[node_type].y = torch.tensor(df['label'].astype(int).values, dtype=torch.long)
                    elif self.task == "ruddit":
                        continuous_labels = df['label'].to_numpy(dtype=np.float64)
                        continuous_labels[np.isnan(continuous_labels)] = self.unlabeled_sentinel
                        data[node_type].y = torch.tensor(continuous_labels, dtype=torch.float32)
                    elif self.task == "RMHD":
                        data[node_type].y = df['label'].tolist()

        stream_load_features(text_nodes_df, "id", text_id_to_idx, "text")
        stream_load_features(user_nodes_df, "id", user_id_to_idx, "user")
        
        # --- Community Embedding Subsetting Logic ---
        state_dict = torch.load(self.pretrain_model_path, map_location='cpu')
        community_embeddings_master = state_dict['community_embedding.weight']
        print(f"Successfully loaded FINAL trained community embeddings with shape: {community_embeddings_master.shape}")
        master_comm_id_to_master_idx = {id_.lower(): i for i, id_ in enumerate(master_comm_ids_df.iloc[:, 0])}  

        ordered_master_indices = []
        for comm_id, _ in sorted(community_id_to_idx.items(), key=lambda item: item[1]):
            master_idx = master_comm_id_to_master_idx.get(comm_id)
            if master_idx is not None: ordered_master_indices.append(master_idx)
        
        data['community'].x = community_embeddings_master[torch.tensor(ordered_master_indices, dtype=torch.long)]
        print(f"Loaded {data['community'].x.size(0):,} community features from pre-trained tensor.")

        # We must load community labels separately, as they are not in the state_dict
        if self.task == 'rspct':
            print("[INFO RSPCT] Task is rspct. Loading community labels from 'unique_community_nodes.parquet'...")
            try:
                community_label_file = get_path(self.root, "unique_community_nodes.parquet")           
                label_schema = pq.ParquetFile(community_label_file).schema.names
                if 'community_id' not in label_schema:
                    raise ValueError(f"'community_id' column not found in unique_community_nodes.parquet. Found: {label_schema}")
                label_cols_to_read = ['community_id', 'label']
                if 'label' not in label_schema:
                    raise ValueError(f"'label' column not found in unique_community_nodes.parquet. Found: {label_schema}")
                if 'label_2' in label_schema:
                    print("[INFO RSPCT] 'label_2' found, will load it.")
                    label_cols_to_read.append('label_2')
                else:
                    print("[INFO RSPCT] 'label_2' not found, will only load 'label'.")

                label_df = pq.read_table(community_label_file, columns=label_cols_to_read).to_pandas()
                
                # Filter labels for communities that are in our final graph
                # (community_id_to_idx contains the final, filtered set of IDs)
                # Use 'community_id' as the key
                label_df['community_id'] = label_df['community_id'].str.lower()
                label_df = label_df[label_df['community_id'].isin(community_id_to_idx)]
                
                # Map community IDs to their graph indices
                # Use 'community_id' as the key
                label_df['idx'] = label_df['community_id'].map(community_id_to_idx)
                
                # Sort by index to align with the data['community'].x tensor
                label_df.set_index('idx', inplace=True)
                label_df.sort_index(inplace=True)
                
                # Sanity check alignment
                if label_df.shape[0] != data['community'].x.size(0):
                    print(f"[WARN RSPCT] Mismatch in label count ({label_df.shape[0]}) and feature count ({data['community'].x.size(0)}). Check community ID mapping.")
                
                # Attach labels
                data['community'].y = label_df['label'].tolist()
                if 'label_2' in label_df.columns:
                    data['community'].y_2 = label_df['label_2'].tolist()
                
                print(f"[INFO RSPCT] Successfully attached {len(data['community'].y)} labels to {label_df.shape[0]} community nodes from 'unique_community_nodes.parquet'.")
            
            except Exception as e:
                print(f"[ERROR RSPCT] Failed to load community labels for RSPCT task: {e}")
                print("Proceeding without community labels. This will likely fail downstream.")
        
        # --- PASS 3: Load Edges ---
        print("\nPass 3: Loading edges...")
        def load_edges(df, src_map, dst_map, src_col, dst_col, filter_replies=False):
            if filter_replies and 'edge_type' in df.columns:
                df = df[df['edge_type'] != 'replies_to']
            src_idx = df[src_col].map(src_map)
            dst_idx = df[dst_col].map(dst_map)
            mask = src_idx.notna() & dst_idx.notna()
            return torch.tensor([src_idx[mask].astype(int).values, dst_idx[mask].astype(int).values], dtype=torch.long)

        data['text', 'post_in', 'community'].edge_index = load_edges(text_comm_edges_df, text_id_to_idx, community_id_to_idx, 'source_id', 'target_id')
        data['text', 'post_by', 'user'].edge_index = load_edges(text_user_edges_df, text_id_to_idx, user_id_to_idx, 'source_id', 'target_id', filter_replies=True)
        data['user', 'active', 'community'].edge_index = load_edges(user_comm_edges_df, user_id_to_idx, community_id_to_idx, 'source_id', 'target_id')
        
        print("\n--- Saving ID Mappings for Chapter 5 Qualitative Analysis ---")
        import json
        
        # Invert the dictionaries so we can look up ID by Index
        # {Raw_ID: 0} -> {0: Raw_ID}
        idx_to_text_id = {v: k for k, v in text_id_to_idx.items()}
        idx_to_user_id = {v: k for k, v in user_id_to_idx.items()}
        idx_to_comm_id = {v: k for k, v in community_id_to_idx.items()}
        
        mapping_dict = {
            'text_nodes': idx_to_text_id,
            'user_nodes': idx_to_user_id,
            'community_nodes': idx_to_comm_id
        }
        
        # Save as a JSON file next to the .pt file
        mapping_filename = self.output_filename.replace('.pt', '_mappings.json')
        mapping_path = os.path.join(self.processed_dir, mapping_filename)
        
        with open(mapping_path, 'w') as f:
            json.dump(mapping_dict, f)
            
        print(f"Mappings saved to {mapping_path}")

        print("\n--- Graph Construction Complete ---")
        print(data)
        torch.save(data, self.processed_paths[0])
        print(f"Graph saved successfully to {self.processed_paths[0]}")
    
    def len(self): return 1
    def get(self, idx): return torch.load(self.processed_paths[0], weights_only=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a HeteroData graph for downstream tasks using contrastive-init features.")
    parser.add_argument("--graph_type", default='downstream')
    parser.add_argument("--task", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--pretrain_data_path", required=True)
    parser.add_argument("--pretrain_model_path", required=True)
    parser.add_argument("--output_filename", required=True)
    args = parser.parse_args()

    dataset = ContrastiveDownstreamGraphDataset(
        root=args.data_path,
        graph_type=args.graph_type,
        task=args.task,
        output_filename=args.output_filename,
        pretrain_data_path=args.pretrain_data_path,
        pretrain_model_path=args.pretrain_model_path
    )
    hetero_graph = dataset[0]
    
    print("\nGraph loaded/processed successfully:")
    print(hetero_graph)
    
    if args.graph_type == 'downstream':
        print_label_distribution(hetero_graph, args.task, UNLABELED_SENTINEL)