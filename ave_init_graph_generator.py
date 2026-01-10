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
import random
import numpy as np

HIDDEN_CHANNELS = 768

UNLABELED_SENTINEL = -999.0
MISSING_USER_PLACEHOLDERS = ["__missing_user__", "__missing__"]

RANDOM_SEED = 42

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
class RobustHeteroGraphDataset(Dataset):
    def __init__(self, root, graph_type: str, task: str, output_filename: str,
                 user_node_file: str, user_edge_file_pretrain: str, user_edge_file_downstream: str,
                 text_node_file: str, comm_node_file: str, text_user_edge_file: str, text_comm_edge_file: str,
                 pretrain_comm_file: str = None,
                 transform=None, pre_transform=None):
        
        self.graph_type = graph_type
        self.task = task
        self.output_filename = output_filename
        
        self.user_node_file = user_node_file
        self.user_edge_file_pretrain = user_edge_file_pretrain
        self.user_edge_file_downstream = user_edge_file_downstream
        self.text_node_file = text_node_file
        self.comm_node_file = comm_node_file
        self.text_user_edge_file = text_user_edge_file
        self.text_comm_edge_file = text_comm_edge_file

        self.pretrain_comm_file = pretrain_comm_file
        
        super().__init__(root, transform, pre_transform)

    @property
    def raw_dir(self) -> str:
        return self.root

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed')

    @property
    def raw_file_names(self):
        # This is now a formality; the process() method defines its own inputs
        return []

    @property
    def processed_file_names(self):
        return [self.output_filename]

    def download(self):
        pass

    def process(self):
        print(f"--- Starting Graph Construction (TYPE: {self.graph_type.upper()}, TASK: {self.task.upper()}) ---")
        
        NORMALIZED_MISSING_USER_PLACEHOLDERS = {p.lower() for p in MISSING_USER_PLACEHOLDERS}
        print(f"[DEBUG] Normalized missing user placeholders: {NORMALIZED_MISSING_USER_PLACEHOLDERS}")

        # --- PASS 1: Define input files and load initial IDs ---
        print("Pass 1: Loading initial node IDs...")

        if self.graph_type == 'pretrain':
            print("INFO: Using SAMPLED user files for pre-training graph.")
            user_node_file = self.user_node_file
            user_edge_file = self.user_edge_file_pretrain
        else:
            print("INFO: Using FULL user files for downstream graph.")
            user_node_file = self.user_node_file
            user_edge_file = self.user_edge_file_downstream

        text_node_emb_file = self.text_node_file
        comm_node_emb_file = self.comm_node_file
        text_user_edge_file = self.text_user_edge_file

        # Read only ID columns initially
        all_text_ids = set(pq.read_table(get_path(self.root, text_node_emb_file), columns=['id']).to_pandas()['id'])
        
        user_df = pq.read_table(get_path(self.root, user_node_file), columns=['id']).to_pandas()
        user_df['id'] = user_df['id'].astype(str).str.lower()
        all_user_ids_initial = set(user_df['id'])
        
        comm_df = pq.read_table(get_path(self.root, comm_node_emb_file), columns=['id']).to_pandas()
        comm_df['id'] = comm_df['id'].astype(str).str.lower()
        all_comm_ids = set(comm_df['id'])
        
        print(f"Initial potential nodes (post-normalization): Text={len(all_text_ids):,}, User={len(all_user_ids_initial):,}, Community={len(all_comm_ids):,}")

        # --- STRICT PRE-TRAIN FILTERING ---
        if self.pretrain_comm_file:
            print(f"  [FILTER] Filtering communities against pre-train file: {self.pretrain_comm_file}")
            
            # Handle absolute vs relative paths
            pt_path = self.pretrain_comm_file
            if not os.path.isabs(pt_path):
                pt_path = get_path(self.root, pt_path)
            
            # Load the valid pre-training universe
            pt_comm_df = pq.read_table(pt_path, columns=['id']).to_pandas()
            valid_pretrain_set = set(pt_comm_df['id'].astype(str).str.lower())
            
            initial_count = len(all_comm_ids)
            
            # INTERSECTION: Only keep communities that appear in the pre-training file
            all_comm_ids = all_comm_ids.intersection(valid_pretrain_set)
            
            dropped_count = initial_count - len(all_comm_ids)
            print(f"  [FILTER] Dropped {dropped_count:,} communities not in pre-training set.")
            print(f"  [FILTER] Final valid community count: {len(all_comm_ids):,}")

        all_user_ids_cleaned = set()
        for user_id in all_user_ids_initial:
            if user_id not in NORMALIZED_MISSING_USER_PLACEHOLDERS: # [MODIFIED] Use normalized set
                all_user_ids_cleaned.add(user_id)
            else:
                 print(f"INFO: Removed placeholder user '{user_id}' from the initial user set.")
        print(f"Total non-placeholder users: {len(all_user_ids_cleaned):,}")

        final_user_ids = all_user_ids_cleaned # Default: keep all cleaned users

        if self.task == 'recommendation':
            print("\n--- Applying User Sampling for Recommendation Task ---")
            try:
                # Load users who have posted text
                text_user_edges_df = pq.read_table(get_path(self.root, text_user_edge_file), columns=['target_id']).to_pandas()
                text_user_edges_df['target_id'] = text_user_edges_df['target_id'].astype(str).str.lower()
                
                # Ensure we only consider users present in the cleaned user node file
                users_with_text_edges = set(text_user_edges_df['target_id']).intersection(all_user_ids_cleaned)
                num_users_with_text = len(users_with_text_edges)
                print(f"Found {num_users_with_text:,} users with text edges (after normalization).")

                # Identify users without text edges
                users_without_text_edges = all_user_ids_cleaned - users_with_text_edges
                num_users_without_text = len(users_without_text_edges)
                print(f"Found {num_users_without_text:,} users without text edges.")

                if num_users_without_text > num_users_with_text:
                    print(f"Sampling {num_users_with_text:,} users from the {num_users_without_text:,} users without text edges...")
                    # Convert set to list for random sampling
                    users_without_text_list = list(users_without_text_edges)
                    # Use numpy for efficient sampling
                    sampled_indices = np.random.choice(len(users_without_text_list), num_users_with_text, replace=False)
                    sampled_users_without_text = {users_without_text_list[i] for i in sampled_indices}
                    print("Sampling complete.")

                    # Combine users with text and sampled users without text
                    final_user_ids = users_with_text_edges.union(sampled_users_without_text)
                    print(f"Final user set size after sampling: {len(final_user_ids):,} ({num_users_with_text} with text + {len(sampled_users_without_text)} sampled without text)")
                else:
                    # If there are fewer 'without_text' users than 'with_text' users, keep all users
                    print("Number of users without text edges is less than or equal to users with text edges. Keeping all users.")
                    final_user_ids = all_user_ids_cleaned # Already set as default

            except FileNotFoundError:
                print(f"WARNING: {text_user_edge_file} not found. Cannot perform user sampling. Keeping all users.")
            except Exception as e:
                 print(f"WARNING: An error occurred during user sampling: {e}. Keeping all users.")

        # Build final ID mappings based on filtered sets
        text_id_to_idx = {id_: i for i, id_ in enumerate(sorted(list(all_text_ids)))}
        # --- Use final_user_ids for the user map ---
        user_id_to_idx = {id_: i for i, id_ in enumerate(sorted(list(final_user_ids)))}
        community_id_to_idx = {id_: i for i, id_ in enumerate(sorted(list(all_comm_ids)))}

        print(f"\nFinal nodes for graph construction: Text={len(text_id_to_idx):,}, User={len(user_id_to_idx):,}, Community={len(community_id_to_idx):,}")

        data = HeteroData()

        # --- PASS 2: Load Node Features and Labels in Chunks ---
        print("\nPass 2: Loading node features and labels in chunks...")

        # --- [MODIFIED] stream_load_features with normalization ---
        def stream_load_features(file_name, id_col, id_map, node_type):
            num_nodes_in_map = len(id_map)
            feature_list = [None] * num_nodes_in_map
            label_data = defaultdict(lambda: [None] * num_nodes_in_map)

            full_file_path = get_path(self.root, file_name)
            print(f"[DEBUG] Attempting to read features for {node_type} from file: {full_file_path}")

            try:
                if not os.path.exists(full_file_path):
                    print(f"ERROR: Feature file not found: {full_file_path}. Skipping node type {node_type}.")
                    data[node_type].num_nodes = num_nodes_in_map
                    data[node_type].x = torch.empty((0, 768), dtype=torch.float32) # Use 0 features, not num_nodes_in_map
                    return

                parquet_file = pq.ParquetFile(full_file_path)
                schema = parquet_file.schema
                schema_names = schema.names
                print(f"[DEBUG] Successfully opened file. Schema names (top-level): {schema_names}")

            except Exception as e:
                 print(f"ERROR: Could not open or read schema from Parquet file {full_file_path}: {e}. Skipping node type {node_type}.")
                 data[node_type].num_nodes = num_nodes_in_map
                 data[node_type].x = torch.empty((0, 768), dtype=torch.float32)
                 return

            columns_to_read = []
            if id_col in schema_names:
                 columns_to_read.append(id_col)
            else:
                 print(f"ERROR: ID column '{id_col}' not found. Cannot proceed for {node_type}.")
                 data[node_type].num_nodes = num_nodes_in_map
                 data[node_type].x = torch.empty((0, 768), dtype=torch.float32)
                 return

            columns_to_read.append('embedding')
            label_cols_present = [col for col in ['label', 'label_2'] if col in schema_names]
            columns_to_read.extend(label_cols_present)
            print(f"[DEBUG] Attempting to read columns: {columns_to_read}")

            nodes_actually_loaded = 0
            try:
                for batch in tqdm(parquet_file.iter_batches(batch_size=50_000, columns=columns_to_read), desc=f"Loading {node_type} features"):
                    df = batch.to_pandas()
                    if 'embedding' not in df.columns:
                        print(f"ERROR: 'embedding' column was NOT loaded into DataFrame. Columns loaded: {df.columns.tolist()}. Skipping batch for {node_type}.")
                        continue 
                    
                    # Only normalize if it's a type that needs it (user or community)
                    if node_type in ('user', 'community'):
                        df[id_col] = df[id_col].astype(str).str.lower()
                    
                    for _, row in df.iterrows():
                        idx = id_map.get(row[id_col]) 
                        if idx is not None:
                            embedding_data = row['embedding']
                            if embedding_data is None or not hasattr(embedding_data, '__len__'):
                                continue
                            try:
                                feature_list[idx] = np.array(embedding_data, dtype=np.float32)
                                nodes_actually_loaded += 1
                                for col in label_cols_present:
                                    label_data[col][idx] = row[col]
                            except ValueError as ve:
                                # print(f"Warning: Could not convert embedding for id {row[id_col]}. Error: {ve}. Skipping.")
                                continue
            except Exception as e:
                 print(f"ERROR reading or processing batches from {file_name}: {e}. Proceeding with potentially incomplete data.")

            filtered_features = [f for f in feature_list if f is not None]

            if not filtered_features:
                print(f"Warning: No valid features found or kept for node type {node_type} (Expected {num_nodes_in_map} based on map).")
                data[node_type].num_nodes = num_nodes_in_map
                emb_dim = 768 # Default dimension
                data[node_type].x = torch.empty((0, emb_dim), dtype=torch.float32)
                if node_type == self.task and node_type == 'text': data[node_type].y = torch.empty(0, dtype=torch.long)
                if node_type == self.task and node_type == 'community': data[node_type].y = []
                return

            try:
                 stacked_features = np.vstack(filtered_features)
                 data[node_type].x = torch.from_numpy(stacked_features)
                 # --- [FIX] Set num_nodes to the size of the *map*, not the loaded features ---
                 data[node_type].num_nodes = num_nodes_in_map
                 print(f"Loaded {data[node_type].x.shape[0]} features for {node_type} nodes (out of {num_nodes_in_map} in map). Feature shape: {data[node_type].x.shape}")
            except ValueError as ve_stack:
                 print(f"CRITICAL ERROR: Could not stack features for {node_type}, likely due to inconsistent embedding dimensions. Error: {ve_stack}")
                 data[node_type].num_nodes = num_nodes_in_map
                 data[node_type].x = torch.empty((0, 768), dtype=torch.float32) # Assign empty on error
                 return

            # --- Robust Label Assignment ---
            if self.graph_type == 'downstream':
                 label_col_to_use = None
                 if self.task == 'rspct' and node_type == 'community' and 'label' in label_data:
                      label_col_to_use = 'label'
                      print("--> Attaching 'label' labels to COMMUNITY nodes for RSPCT task.")
                 elif node_type == 'text' and 'label' in label_data:
                      label_col_to_use = 'label'
                      print(f"--> Attaching 'label' labels to TEXT nodes for {self.task.upper()} task.")

                 if label_col_to_use:
                    loaded_indices = {idx for idx, feat in enumerate(feature_list) if feat is not None}
                    filtered_raw_labels = [label_data[label_col_to_use][idx] for idx in range(num_nodes_in_map) if idx in loaded_indices]

                    if not filtered_raw_labels:
                         print(f"Warning: No labels found for the loaded {node_type} nodes.")
                    elif self.task in ["normvio", "dreaddit", "hateful"] and node_type=='text':
                        processed_labels = [int(l) if l is not None else int(UNLABELED_SENTINEL) for l in filtered_raw_labels]
                        data[node_type].y = torch.tensor(processed_labels, dtype=torch.long)
                    elif self.task == "ruddit" and node_type=='text':
                        processed_labels = np.array([float(l) if l is not None else UNLABELED_SENTINEL for l in filtered_raw_labels], dtype=np.float64)
                        processed_labels[np.isnan(processed_labels)] = UNLABELED_SENTINEL
                        data[node_type].y = torch.tensor(processed_labels, dtype=torch.float32)
                    elif self.task == "RMHD" and node_type=='text':
                        data[node_type].y = [l if l is not None else "UNLABELED" for l in filtered_raw_labels]
                    elif self.task == "rspct" and node_type=='community':
                        data[node_type].y = [l if l is not None else "UNLABELED" for l in filtered_raw_labels]
                        if 'label_2' in label_data:
                             filtered_raw_labels_2 = [label_data['label_2'][idx] for idx in range(num_nodes_in_map) if idx in loaded_indices]
                             data[node_type].y_2 = [l if l is not None else "UNLABELED" for l in filtered_raw_labels_2]

                    if hasattr(data[node_type], 'y') and len(data[node_type].y) != data[node_type].x.shape[0]:
                         print(f"CRITICAL WARNING: Mismatch between feature count ({data[node_type].x.shape[0]}) and label count ({len(data[node_type].y)}) for {node_type}.")


        stream_load_features(text_node_emb_file, "id", text_id_to_idx, "text")
        stream_load_features(user_node_file, "id", user_id_to_idx, "user") # Uses filtered user map
        stream_load_features(comm_node_emb_file, "id", community_id_to_idx, "community")

        # --- PASS 3: Load Edges in Chunks ---
        print("\nPass 3: Loading edges in chunks...")
        
        def stream_load_edges(file_name, src_map, dst_map, src_col, dst_col, edge_type_tuple, filter_replies=False):
            edge_chunks = []
            num_edges_processed = 0
            num_edges_kept = 0
            file_path = get_path(self.root, file_name)

            if not os.path.exists(file_path):
                 print(f"WARNING: Edge file not found: {file_path}. Skipping edge type {edge_type_tuple}.")
                 return torch.empty((2, 0), dtype=torch.long)
            try:
                parquet_file = pq.ParquetFile(file_path)
            except Exception as e:
                 print(f"ERROR: Could not open edge file {file_path}: {e}. Skipping edge type {edge_type_tuple}.")
                 return torch.empty((2, 0), dtype=torch.long)

            columns_to_read = [src_col, dst_col]
            if filter_replies and 'edge_type' in parquet_file.schema.names:
                columns_to_read.append('edge_type')
            elif filter_replies:
                 print(f"WARNING: filter_replies=True but 'edge_type' column not found in {file_name}.")

            desc=f"Loading edges {edge_type_tuple[0]}->{edge_type_tuple[2]}"
            try:
                for batch in tqdm(parquet_file.iter_batches(batch_size=500_000, columns=columns_to_read), desc=desc):
                    df = batch.to_pandas()
                    initial_batch_count = len(df)
                    num_edges_processed += initial_batch_count
                    
                    if initial_batch_count == 0:
                        continue

                    if filter_replies and 'edge_type' in df.columns:
                        df = df[df['edge_type'] != 'replies_to']

                    if edge_type_tuple == ('text', 'post_by', 'user'):
                        df[dst_col] = df[dst_col].astype(str).str.lower() # user_id
                    elif edge_type_tuple == ('text', 'post_in', 'community'):
                        df[dst_col] = df[dst_col].astype(str).str.lower() # community_id
                    elif edge_type_tuple == ('user', 'active_in', 'community'):
                        df[src_col] = df[src_col].astype(str).str.lower() # user_id
                        df[dst_col] = df[dst_col].astype(str).str.lower() # community_id

                    # Map IDs to indices. Unmapped IDs (not in final_user_ids etc.) will become NaN
                    src_idx = df[src_col].map(src_map)
                    dst_idx = df[dst_col].map(dst_map)

                    src_na_mask = src_idx.isna()
                    dst_na_mask = dst_idx.isna()
                    combined_na_mask = src_na_mask | dst_na_mask
                    
                    num_total_invalid = combined_na_mask.sum()
                    if num_total_invalid > 0:
                        print(f"[DEBUG] load_edges ({edge_type_tuple}): Batch dropout breakdown:")
                        print(f"  -> {src_na_mask.sum():8,} edges dropped due to (source_id: '{src_col}') not in master list.")
                        print(f"  -> {dst_na_mask.sum():8,} edges dropped due to (target_id: '{dst_col}') not in master list.")
                        print(f"  -> {num_total_invalid:8,} total rows dropped in batch (at least one ID was invalid).")

                    # Keep only edges where both source and destination nodes are in the final maps
                    mask = ~combined_na_mask
                    if mask.any():
                        kept_src = src_idx[mask].values.astype(int) # Convert to int after filtering NaNs
                        kept_dst = dst_idx[mask].values.astype(int)
                        edge_chunks.append(torch.tensor([kept_src, kept_dst], dtype=torch.long))
                        num_edges_kept += len(kept_src)
            except Exception as e:
                 print(f"ERROR reading batches from edge file {file_name}: {e}. Proceeding with potentially incomplete edges.")

            if not edge_chunks:
                print(f"Warning: No valid edges kept for {edge_type_tuple} from {file_name} (Processed {num_edges_processed} edges).")
                return torch.empty((2, 0), dtype=torch.long)

            print(f"Kept {num_edges_kept:,} / {num_edges_processed:,} edges for {edge_type_tuple}.")
            return torch.cat(edge_chunks, dim=1)


        # Assign edges using the final maps
        data['text', 'post_in', 'community'].edge_index = stream_load_edges(
            self.text_comm_edge_file, text_id_to_idx, community_id_to_idx,
            'source_id', 'target_id', ('text', 'post_in', 'community'))

        data['text', 'post_by', 'user'].edge_index = stream_load_edges(
            self.text_user_edge_file, text_id_to_idx, user_id_to_idx, # user_id_to_idx is filtered
            'source_id', 'target_id', ('text', 'post_by', 'user'), filter_replies=True)

        data['user', 'active_in', 'community'].edge_index = stream_load_edges(
            user_edge_file, user_id_to_idx, community_id_to_idx, # user_id_to_idx is filtered
            'source_id', 'target_id', ('user', 'active_in', 'community'))


        # This is critical for the contrastive init (pre-training) graph.
        print("\n--- [DEBUG CHECK] Verifying User-Community Edge Coverage ---")
        
        num_users_total = data['user'].num_nodes
        all_user_indices_set = set(range(num_users_total))
        
        users_with_comm_edges = torch.empty(0, dtype=torch.long)
        # Check if the edge_index exists and is not empty
        if ('user', 'active_in', 'community') in data.edge_types and data['user', 'active_in', 'community'].edge_index.numel() > 0:
            users_with_comm_edges = torch.unique(data['user', 'active_in', 'community'].edge_index[0])
        else:
            print("[DEBUG CHECK] No ('user', 'active_in', 'community') edges found.")
            
        users_with_comm_edges_set = set(users_with_comm_edges.cpu().numpy())
        
        num_users_with_edges = len(users_with_comm_edges_set)
        
        missing_user_indices = all_user_indices_set - users_with_comm_edges_set
        num_missing = len(missing_user_indices)

        print(f"[DEBUG CHECK] Total users in graph: {num_users_total:,}")
        print(f"[DEBUG CHECK] Users with at least one community edge: {num_users_with_edges:,}")
        
        if num_missing == 0:
            print("  -> \033[92mSUCCESS:\033[0m All users have at least one community edge.")
        else:
            print(f"  -> \033[91mCRITICAL WARNING:\033[0m {num_missing:,} users have NO community edges.")
            if self.graph_type == 'pretrain':
                 print("     -> This will cause issues for contrastive initialization!")
            if num_missing < 10:
                print(f"     Missing user indices (examples): {list(missing_user_indices)}")
            else:
                print(f"     First 10 missing user indices (examples): {list(missing_user_indices)[:10]}")
        print("---------------------------------------------------------")

        print("\n--- [DEBUG CHECK] Verifying Text-Community Edge Coverage ---")
        
        num_text_total = data['text'].num_nodes
        all_text_indices_set = set(range(num_text_total))
        
        text_with_comm_edges = torch.empty(0, dtype=torch.long)
        # Check if the edge_index exists and is not empty
        if ('text', 'post_in', 'community') in data.edge_types and data['text', 'post_in', 'community'].edge_index.numel() > 0:
            text_with_comm_edges = torch.unique(data['text', 'post_in', 'community'].edge_index[0]) # [0] is the text node
        else:
            print("[DEBUG CHECK] No ('text', 'post_in', 'community') edges found.")
            
        text_with_comm_edges_set = set(text_with_comm_edges.cpu().numpy())
        
        num_text_with_edges = len(text_with_comm_edges_set)
        
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

        # --- FINAL SANITY CHECK ---
        print("\n--- Final Graph Structure ---")
        print(data)
        # Check node counts vs feature matrix sizes
        for node_type in data.node_types:
             num_nodes_expected = data[node_type].num_nodes
             has_x = hasattr(data[node_type], 'x') and data[node_type].x is not None
             num_nodes_features = data[node_type].x.shape[0] if has_x else -1
             
             feature_shape_str = data[node_type].x.shape if has_x else 'MISSING or None'
             # Handle the case where x is an empty tensor
             if has_x and data[node_type].x.numel() == 0 and num_nodes_features == 0:
                 feature_shape_str = f"EMPTY Tensor {data[node_type].x.shape}"
                 
             print(f"Node Type '{node_type}': Expected Nodes={num_nodes_expected:,}, Features Shape={feature_shape_str}")
             
             if not has_x and num_nodes_expected > 0 : # Check if features are missing when nodes are expected
                 all_features_present = False
                 print(f"  !!! CRITICAL: Features (.x) are missing for node type '{node_type}' before saving!")
             
             # Check for mismatch between expected nodes and loaded features
             if has_x and num_nodes_expected > 0 and num_nodes_features != num_nodes_expected:
                 # This check is now more complex. We allow num_nodes_features < num_nodes_expected
                 # as long as the map is correct. The real error is a mismatch in label/feature count.
                 print(f"  !!! WARNING: Mismatch between expected nodes in map ({num_nodes_expected:,}) and loaded features ({num_nodes_features:,}). This may be OK if some nodes just lack features.")

             if hasattr(data[node_type], 'y'):
                  num_labels = len(data[node_type].y) if isinstance(data[node_type].y, list) else data[node_type].y.shape[0]
                  print(f"  Labels Shape/Length: {num_labels:,}")
                  if num_nodes_features != -1 and num_labels != num_nodes_features:
                       print(f"  !!! MISMATCH between features ({num_nodes_features}) and labels ({num_labels})")

        # Check edge counts
        for edge_type in data.edge_types:
             num_edges = data[edge_type].edge_index.shape[1]
             print(f"Edge Type '{edge_type}': Num Edges={num_edges:,}")

        print("\n--- Saving ID Mappings for Qualitative Analysis ---")
        import json
        
        # Invert the dictionaries 
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
        mapping_path = os.path.join(self.processed_dir, mapping_filename)
        
        with open(mapping_path, 'w') as f:
            json.dump(mapping_dict, f)
            
        print(f"Mappings saved to {mapping_path}")

        torch.save(data, self.processed_paths[0])
        print(f"\nGraph saved successfully to {self.processed_paths[0]}")

    def len(self): return 1
    def get(self, idx): return torch.load(self.processed_paths[0], weights_only=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a HeteroData graph for pre-training or downstream tasks.")
    parser.add_argument("--graph_type", required=True, choices=['pretrain', 'downstream'], help="The purpose of the graph.")
    parser.add_argument("--task", required=True, help="The specific downstream task (e.g., 'normvio') or 'NONE' for pre-training.")
    parser.add_argument("--data_path", required=True, help="The path to the graph_data directory.")
    parser.add_argument("--output_filename", required=True, help="The filename for the final processed graph .pt file.")
    parser.add_argument("--user_node_file", default="avg_user_node_embeddings.parquet", help="Filename for user node features.")
    parser.add_argument("--user_edge_file_pretrain", default="user_community_active_edges.parquet", help="Filename for user-community edges (pre-training).")
    parser.add_argument("--user_edge_file_downstream", default="user_community_active_edges.parquet", help="Filename for user-community edges (downstream).")
    parser.add_argument("--text_node_file", default="text_node_embeddings_xlnet.parquet", help="Filename for text node features.")
    parser.add_argument("--comm_node_file", default="avg_community_node_embeddings.parquet", help="Filename for community node features.")
    parser.add_argument("--text_user_edge_file", default="text_user_edges.parquet", help="Filename for text-user edges.")
    parser.add_argument("--text_comm_edge_file", default="text_community_edges.parquet", help="Filename for text-community edges.")
    parser.add_argument("--pretrain_comm_file", default=None, help="Optional: Path to the pre-training community node file for strict filtering.")
    
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    print(f"--- Random seed set to: {RANDOM_SEED} ---")

    dataset = RobustHeteroGraphDataset(
        root=args.data_path,
        graph_type=args.graph_type,
        task=args.task,
        output_filename=args.output_filename,
        user_node_file=args.user_node_file,
        user_edge_file_pretrain=args.user_edge_file_pretrain,
        user_edge_file_downstream=args.user_edge_file_downstream,
        text_node_file=args.text_node_file,
        comm_node_file=args.comm_node_file,
        text_user_edge_file=args.text_user_edge_file,
        text_comm_edge_file=args.text_comm_edge_file,
        pretrain_comm_file=args.pretrain_comm_file
    )
    hetero_graph = dataset[0]
    
    print("\nGraph loaded/processed successfully:")
    print(hetero_graph)
    
    if args.graph_type == 'downstream':
        print_label_distribution(hetero_graph, args.task, UNLABELED_SENTINEL)