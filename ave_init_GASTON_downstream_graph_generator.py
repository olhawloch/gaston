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
import json # Added for mapping export

# --- Configuration ---
MISSING_USER_PLACEHOLDERS = ["__missing_user__", "__missing__"]
UNLABELED_SENTINEL = -999.0

# --- Helper Functions ---
def get_path(root: str, filename: str) -> str:
    return os.path.join(root, filename)

def print_label_distribution(data: HeteroData, task: str, unlabeled_sentinel: float):
    """Prints a sanity check summary of the node labels based on the task."""
    print(f"\n--- Label Sanity Check (TASK: {task.upper()}) ---")

    if task == 'rspct':
        if 'community' in data.node_types and hasattr(data['community'], 'y'):
            labels = data['community'].y
            counts = collections.Counter(labels)
            print("Community-Level Label Counts:")
            for label, count in counts.most_common(10): 
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
class AveInitDownstreamGraphDataset(Dataset):
    def __init__(self, root, task: str, output_filename: str, pretrain_model_path: str, 
                 avg_user_file: str, avg_comm_file: str, transform=None, pre_transform=None):
        
        self.task = task
        self.output_filename = output_filename
        self.pretrain_model_path = pretrain_model_path
        self.avg_user_file = avg_user_file
        self.avg_comm_file = avg_comm_file
        self.unlabeled_sentinel = UNLABELED_SENTINEL
        
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
        print(f"--- Starting Ave-Init Downstream Graph Construction (TASK: {self.task.upper()}) ---")
        
        # --- PASS 1: Reconstruct Master List and Filter ---
        print("Pass 1: Establishing Pre-training Universe and Downstream Filtering...")
        
        # 1. Reconstruct the MASTER community list from the pre-training files (Sort by ID)
        master_comm_df = pq.read_table(get_path(self.root, self.avg_comm_file), columns=['id']).to_pandas()
        master_comm_df['id'] = master_comm_df['id'].astype(str).str.lower()
        
        # THIS SORT IS CRITICAL: It replicates the logic of the pre-training graph construction
        valid_pretrain_communities = sorted(list(set(master_comm_df['id'])))
        print(f"Loaded and sorted {len(valid_pretrain_communities):,} master communities from {self.avg_comm_file}.")

        # 2. Load Downstream Edges
        text_comm_edges_df = pq.read_table(get_path(self.root, "text_community_edges.parquet")).to_pandas()
        text_comm_edges_df['source_id'] = text_comm_edges_df['source_id'].astype(str).str.lower()
        user_comm_edges_df = pq.read_table(get_path(self.root, "user_community_active_edges.parquet")).to_pandas()
        user_comm_edges_df['source_id'] = user_comm_edges_df['source_id'].astype(str).str.lower()

        # 3. Define Active Communities (Task Dependent)
        if self.task == 'rspct':
            print("[INFO RSPCT] Defining downstream communities *only* from 'unique_community_nodes.parquet' (label file).")
            label_file_path = get_path(self.root, "unique_community_nodes.parquet")
            label_comm_df = pq.read_table(label_file_path, columns=['community_id']).to_pandas()
            active_downstream_communities = set(label_comm_df['community_id'].astype(str).str.lower())
            print(f"Found {len(active_downstream_communities):,} labeled communities for RSPCT.")
            
            text_comm_edges_df['target_id'] = text_comm_edges_df['target_id'].astype(str).str.lower()
            user_comm_edges_df['target_id'] = user_comm_edges_df['target_id'].astype(str).str.lower()
        else:
            print("[INFO] Defining downstream communities from text and user edge files.")
            text_comm_edges_df['target_id'] = text_comm_edges_df['target_id'].astype(str).str.lower()
            user_comm_edges_df['target_id'] = user_comm_edges_df['target_id'].astype(str).str.lower()
            active_downstream_communities = set(text_comm_edges_df['target_id']).union(set(user_comm_edges_df['target_id']))
            print(f"Found {len(active_downstream_communities):,} active communities in the downstream task.")

        # 4. Intersection: Task Communities vs Master Pretrain Communities
        final_community_set = active_downstream_communities.intersection(set(valid_pretrain_communities))
        print(f"Found {len(final_community_set):,} communities in both the task and pre-trained list.")

        # 5. Filter Edges by Community
        text_comm_edges_df = text_comm_edges_df[text_comm_edges_df['target_id'].isin(final_community_set)]
        user_comm_edges_df = user_comm_edges_df[user_comm_edges_df['target_id'].isin(final_community_set)]
        valid_text_ids_from_edges = set(text_comm_edges_df['source_id'])

        # 6. Load and Filter Text Nodes
        text_parquet_file = get_path(self.root, "text_node_embeddings_gemma.parquet")
        try:
            text_schema = pq.ParquetFile(text_parquet_file).schema.names
        except Exception as e:
            raise ValueError(f"Could not read schema from {text_parquet_file}: {e}")
            
        text_columns_to_read = ['id', 'embedding']
        if self.task == 'rspct':
            pass # No text labels needed
        elif 'label' in text_schema:
            text_columns_to_read.append('label')

        print(f"Loading text nodes from {text_parquet_file}...")
        text_nodes_df = pq.read_table(text_parquet_file, columns=text_columns_to_read).to_pandas()
        text_nodes_df['id'] = text_nodes_df['id'].astype(str).str.lower()
        
        # Filter text nodes
        text_nodes_df = text_nodes_df[text_nodes_df['id'].isin(valid_text_ids_from_edges)]
        text_nodes_df = text_nodes_df.drop_duplicates(subset=['id'], keep='first')
        print(f"Kept {len(text_nodes_df):,} text nodes active in this task.")

        # 7. Load User Nodes (Using AVERAGE embeddings file)
        print(f"Loading user nodes from {self.avg_user_file}...")
        user_nodes_df = pq.read_table(get_path(self.root, self.avg_user_file)).to_pandas()
        user_nodes_df['id'] = user_nodes_df['id'].astype(str).str.lower()
        
        # Remove placeholders
        normalized_placeholders = {p.lower() for p in MISSING_USER_PLACEHOLDERS}
        user_nodes_df = user_nodes_df[~user_nodes_df['id'].isin(normalized_placeholders)]
        user_nodes_df = user_nodes_df.drop_duplicates(subset=['id'], keep='first')
        
        all_potential_user_ids = set(user_nodes_df['id'])
        print(f"Found {len(all_potential_user_ids):,} potential users with average embeddings.")

        # 8. Filter Users (Task Specific)
        text_user_edges_df = pq.read_table(get_path(self.root, "text_user_edges.parquet")).to_pandas()
        text_user_edges_df['source_id'] = text_user_edges_df['source_id'].astype(str).str.lower()
        text_user_edges_df['target_id'] = text_user_edges_df['target_id'].astype(str).str.lower()
        
        # Filter text-user edges to only include valid text nodes
        valid_text_ids = set(text_nodes_df['id'])
        text_user_edges_df = text_user_edges_df[text_user_edges_df['source_id'].isin(valid_text_ids)]

        if self.task == 'recommendation':
            print("\n--- Applying User Sampling for Recommendation Task ---")
            users_with_text_edges_all = set(text_user_edges_df['target_id'])
            users_with_text_edges = all_potential_user_ids.intersection(users_with_text_edges_all)
            
            users_without_text_edges = all_potential_user_ids - users_with_text_edges
            
            num_with = len(users_with_text_edges)
            num_without = len(users_without_text_edges)
            print(f"Users with text edges: {num_with:,}. Users without: {num_without:,}.")

            if num_without > num_with and num_with > 0:
                print(f"Sampling {num_with:,} users from the 'without' set.")
                users_without_list = list(users_without_text_edges)
                sampled_indices = np.random.choice(len(users_without_list), num_with, replace=False)
                sampled_users_without = {users_without_list[i] for i in sampled_indices}
                final_user_set = users_with_text_edges.union(sampled_users_without)
            else:
                final_user_set = all_potential_user_ids
        else:
            print(f"[INFO] Filtering users based on activity in edges.")
            active_text_users = set(text_user_edges_df['target_id'])
            active_comm_users = set(user_comm_edges_df['source_id'])
            active_users = active_text_users.union(active_comm_users)
            final_user_set = all_potential_user_ids.intersection(active_users)

        # Apply user filter
        user_nodes_df = user_nodes_df[user_nodes_df['id'].isin(final_user_set)]
        print(f"Final user count: {len(user_nodes_df):,}")

        # 9. Create Mappings
        text_id_to_idx = {id_: i for i, id_ in enumerate(sorted(list(set(text_nodes_df['id']))))}
        user_id_to_idx = {id_: i for i, id_ in enumerate(sorted(list(set(user_nodes_df['id']))))}
        community_id_to_idx = {id_: i for i, id_ in enumerate(sorted(list(final_community_set)))}
        
        print(f"Final Graph Nodes: Text={len(text_id_to_idx):,}, User={len(user_id_to_idx):,}, Community={len(community_id_to_idx):,}")

        data = HeteroData()

        # --- PASS 2: Load Features ---
        print("\nPass 2: Loading Node Features...")

        # Helper to load DF into tensor based on map
        def load_features_from_df(df, id_map, node_type):
            df = df[df['id'].isin(id_map)].copy()
            df['idx'] = df['id'].map(id_map)
            df.set_index('idx', inplace=True)
            df.sort_index(inplace=True)
            data[node_type].x = torch.from_numpy(np.vstack(df['embedding'].apply(np.array).values))
            print(f"Loaded {data[node_type].x.shape} features for {node_type}.")
            
            # Text Labels
            if node_type == 'text' and 'label' in df.columns:
                print(f"--> Attaching labels to TEXT nodes for {self.task.upper()}.")
                if self.task in ["normvio", "dreaddit", "hateful"]:
                    data[node_type].y = torch.tensor(df['label'].astype(int).values, dtype=torch.long)
                elif self.task == "ruddit":
                    continuous_labels = df['label'].to_numpy(dtype=np.float64)
                    continuous_labels[np.isnan(continuous_labels)] = self.unlabeled_sentinel
                    data[node_type].y = torch.tensor(continuous_labels, dtype=torch.float32)
                elif self.task == "RMHD":
                    data[node_type].y = df['label'].tolist()

        # Load Text (Parquet)
        load_features_from_df(text_nodes_df, text_id_to_idx, "text")

        # Load User (Parquet - Average Embeddings)
        load_features_from_df(user_nodes_df, user_id_to_idx, "user")

        # Load Community (Trained Checkpoint)
        print("Loading TRAINED community embeddings from checkpoint...")
        if not os.path.exists(self.pretrain_model_path):
            raise FileNotFoundError(f"Checkpoint not found at {self.pretrain_model_path}")
            
        state_dict = torch.load(self.pretrain_model_path, map_location='cpu')
        
        # Extract the trained weights
        if 'community_embedding.weight' in state_dict:
            trained_weights = state_dict['community_embedding.weight']
        elif 'state_dict' in state_dict and 'community_embedding.weight' in state_dict['state_dict']:
             trained_weights = state_dict['state_dict']['community_embedding.weight']
        else:
             print(f"Keys found in checkpoint: {list(state_dict.keys())[:5]}")
             trained_weights = state_dict.get('community_embedding.weight')
        
        if trained_weights is None:
            raise KeyError("Could not find 'community_embedding.weight' in checkpoint.")

        print(f"Checkpoint weights shape: {trained_weights.shape}")
        
        # Map master indices (Sorted Alphabetical) to downstream indices
        master_id_to_idx = {id_: i for i, id_ in enumerate(valid_pretrain_communities)}
        
        # Validation print
        print("\n--- [DEBUG] Community Mapping Verification ---")
        print(f"Total Master Communities: {len(valid_pretrain_communities)}")
        print("First 3 Master Communities (Index 0-2 in Checkpoint):")
        for i in range(min(3, len(valid_pretrain_communities))):
            print(f"  Row {i}: {valid_pretrain_communities[i]}")
            
        indices_to_select = []
        # community_id_to_idx contains the sorted list of downstream communities
        sorted_downstream_comms = sorted(community_id_to_idx.keys(), key=lambda k: community_id_to_idx[k])
        
        for comm_id in sorted_downstream_comms:
            master_idx = master_id_to_idx.get(comm_id)
            if master_idx is not None:
                indices_to_select.append(master_idx)
            else:
                raise ValueError(f"Community {comm_id} found in downstream set but not in master pretrain list. This should be impossible due to filtering.")
        
        data['community'].x = trained_weights[torch.tensor(indices_to_select, dtype=torch.long)]
        print(f"Extracted {data['community'].x.size(0):,} trained community features.")

        # Community Labels (RSPCT)
        if self.task == 'rspct':
            print("[INFO RSPCT] Loading community labels...")
            label_df = pq.read_table(get_path(self.root, "unique_community_nodes.parquet")).to_pandas()
            label_df['community_id'] = label_df['community_id'].astype(str).str.lower()
            label_df = label_df[label_df['community_id'].isin(community_id_to_idx)]
            label_df['idx'] = label_df['community_id'].map(community_id_to_idx)
            label_df.set_index('idx', inplace=True)
            label_df.sort_index(inplace=True)
            
            data['community'].y = label_df['label'].tolist()
            if 'label_2' in label_df.columns:
                data['community'].y_2 = label_df['label_2'].tolist()

        # --- PASS 3: Load Edges ---
        print("\nPass 3: Loading Edges...")
        
        def load_edges(df, src_map, dst_map, src_col, dst_col, filter_replies=False):
            if filter_replies and 'edge_type' in df.columns:
                df = df[df['edge_type'] != 'replies_to']
            src_idx = df[src_col].map(src_map)
            dst_idx = df[dst_col].map(dst_map)
            mask = src_idx.notna() & dst_idx.notna()
            return torch.tensor([src_idx[mask].astype(int).values, dst_idx[mask].astype(int).values], dtype=torch.long)

        data['text', 'post_in', 'community'].edge_index = load_edges(text_comm_edges_df, text_id_to_idx, community_id_to_idx, 'source_id', 'target_id')
        data['text', 'post_by', 'user'].edge_index = load_edges(text_user_edges_df, text_id_to_idx, user_id_to_idx, 'source_id', 'target_id', filter_replies=True)
        data['user', 'active_in', 'community'].edge_index = load_edges(user_comm_edges_df, user_id_to_idx, community_id_to_idx, 'source_id', 'target_id')

        # --- Save Mappings for Quantitative Analysis ---
        print("\n--- Saving ID Mappings for Qualitative Analysis ---")
        
        # Invert the dictionaries to Map {Index -> Original_ID}
        idx_to_text_id = {v: k for k, v in text_id_to_idx.items()}
        idx_to_user_id = {v: k for k, v in user_id_to_idx.items()}
        idx_to_comm_id = {v: k for k, v in community_id_to_idx.items()}
        
        mapping_dict = {
            'text_nodes': idx_to_text_id,
            'user_nodes': idx_to_user_id,
            'community_nodes': idx_to_comm_id
        }
        
        # Save as a JSON file
        mapping_filename = self.output_filename.replace('.pt', '_mappings.json')
        # Ensure processed dir exists
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
            
        mapping_path = os.path.join(self.processed_dir, mapping_filename)
        
        with open(mapping_path, 'w') as f:
            json.dump(mapping_dict, f)
            
        print(f"Mappings saved to {mapping_path}")

        print("\n--- Graph Construction Complete ---")
        print(data)
        torch.save(data, self.processed_paths[0])
        print(f"Graph saved to {self.processed_paths[0]}")

    def len(self): return 1
    def get(self, idx): return torch.load(self.processed_paths[0], weights_only=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate HeteroData for Ave-Init Downstream Ablation.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--pretrain_model_path", required=True, help="Path to the .pt checkpoint file from ave_init ablation.")
    parser.add_argument("--output_filename", required=True)
    parser.add_argument("--avg_user_file", default="avg_user_node_embeddings.parquet", help="File containing average user embeddings.")
    parser.add_argument("--avg_comm_file", default="avg_community_node_embeddings.parquet", help="File used for initial community ordering.")
    args = parser.parse_args()

    dataset = AveInitDownstreamGraphDataset(
        root=args.data_path,
        task=args.task,
        output_filename=args.output_filename,
        pretrain_model_path=args.pretrain_model_path,
        avg_user_file=args.avg_user_file,
        avg_comm_file=args.avg_comm_file
    )
    hetero_graph = dataset[0]
    
    print("\nGraph processed successfully.")
    print_label_distribution(hetero_graph, args.task, UNLABELED_SENTINEL)