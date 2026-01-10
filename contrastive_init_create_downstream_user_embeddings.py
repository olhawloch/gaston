import torch
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import os
from torch_scatter import scatter_mean
import argparse
from collections import defaultdict
from tqdm import tqdm

def create_user_features(pretrain_path: str, downstream_path: str, model_path: str):
    print("--- Creating Static User Features for Downstream Task ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Define Input and Output Paths ---
    PRETRAINED_MODEL_PATH = model_path
    CONTR_INIT_MASTER_COMM_IDS_PATH = os.path.join(pretrain_path, "contr_init_community_ids_master.parquet")
    USER_EMBEDDINGS_OUTPUT_PATH = os.path.join(downstream_path, "user_node_embeddings_from_contr_init.parquet")

    # --- 2. Load Pre-trained Community Embeddings ---
    print(f"Loading pre-trained model from: {PRETRAINED_MODEL_PATH}")
    state_dict = torch.load(PRETRAINED_MODEL_PATH, map_location=device)
    community_embeddings = state_dict['community_embedding.weight'].detach()
    master_comm_ids_df = pq.read_table(CONTR_INIT_MASTER_COMM_IDS_PATH).to_pandas()
    comm_name_to_pretrain_idx = {name.lower(): i for i, name in enumerate(master_comm_ids_df.iloc[:, 0])}
    print(f"Loaded {len(comm_name_to_pretrain_idx)} master community mappings (normalized to lowercase).")

    # --- 3. Load Downstream Task Data ---
    DOWNSTREAM_USER_IDS_PATH = os.path.join(downstream_path, "unique_user_nodes.parquet")
    DOWNSTREAM_USER_COMM_EDGES_PATH = os.path.join(downstream_path, "user_community_active_edges.parquet")

    print(f"Loading downstream user list from: {os.path.basename(DOWNSTREAM_USER_IDS_PATH)}")
    downstream_users_df = pq.read_table(DOWNSTREAM_USER_IDS_PATH).to_pandas()
    downstream_user_list = downstream_users_df.iloc[:, 0].tolist()
    num_downstream_users = len(downstream_user_list)
    downstream_user_to_idx = {uid: i for i, uid in enumerate(downstream_user_list)}
    print(f"Loaded {num_downstream_users:,} users for the downstream task.")

    print(f"Loading downstream user-community edges from: {os.path.basename(DOWNSTREAM_USER_COMM_EDGES_PATH)}")
    downstream_edges_df = pq.read_table(DOWNSTREAM_USER_COMM_EDGES_PATH).to_pandas()
    downstream_edges_df.rename(columns={'source_id': 'user_id', 'target_id': 'community_id'}, inplace=True)
    print(f"Loaded {len(downstream_edges_df):,} downstream user-community edges.")

    # --- 4. Filter Edges and Build Tensors for scatter_mean ---
    print("Filtering edges and building aggregation tensors...")
    
    total_edges_processed = 0
    edges_kept = 0
    edges_dropped_community_missing = 0
    communities_dropped = set()

    src_comm_indices = [] # Will hold indices for the master embedding tensor
    dst_user_indices = [] # Will hold indices for the new downstream user tensor

    for _, row in tqdm(downstream_edges_df.iterrows(), total=len(downstream_edges_df), desc="Mapping edges"):
        total_edges_processed += 1
        user_id = row['user_id']
        
        community_id = row['community_id'].lower()
        
        downstream_user_idx = downstream_user_to_idx.get(user_id)
        pretrain_comm_idx = comm_name_to_pretrain_idx.get(community_id)
        
        if downstream_user_idx is not None and pretrain_comm_idx is not None:
            src_comm_indices.append(pretrain_comm_idx)
            dst_user_indices.append(downstream_user_idx)
            edges_kept += 1
        
        elif downstream_user_idx is not None and pretrain_comm_idx is None:
            edges_dropped_community_missing += 1
            communities_dropped.add(community_id)

    print(f"Found {len(src_comm_indices):,} valid, mapped edges to use for user feature generation.")
    
    print("\n" + "="*50)
    print("--- Edge Filtering Report ---")
    print(f"  Total edges processed: {total_edges_processed:,}")
    print(f"  Edges kept (user & comm matched): {edges_kept:,}")
    print(f"  Edges dropped (community not in pre-train set): {edges_dropped_community_missing:,}")
    if communities_dropped:
        print(f"  --- Found {len(communities_dropped)} downstream communities that were NOT in the pre-trained set ---")
        for i, comm_name in enumerate(list(communities_dropped)[:10]):
            print(f"    - {comm_name}")
        if len(communities_dropped) > 10:
            print(f"    ... and {len(communities_dropped) - 10} more.")
    else:
        print("All active communities in this task were found in the pre-trained set!")
    print("="*50 + "\n")

    src_tensor = torch.tensor(src_comm_indices, dtype=torch.long, device=device)
    dst_tensor = torch.tensor(dst_user_indices, dtype=torch.long, device=device)

    # --- 5. Perform scatter_mean ONCE to calculate all downstream user features ---
    print("Calculating user features via scatter_mean...")
    user_features = scatter_mean(
        src=community_embeddings[src_tensor], 
        index=dst_tensor, 
        dim=0, 
        dim_size=num_downstream_users
    ).cpu().numpy()
    print(f"Generated user features with shape: {user_features.shape}")

    print("\n" + "="*50)
    print("--- User Feature Generation Report ---")
    num_users_with_features = len(torch.unique(dst_tensor))
    num_users_with_zero_vector = num_downstream_users - num_users_with_features
    print(f"  Total users in this task: {num_downstream_users:,}")
    print(f"  Users with features (from 1+ valid community): {num_users_with_features:,}")
    print(f"  Users with a zero-vector embedding: {num_users_with_zero_vector:,}")
    if num_users_with_zero_vector == 0:
        print("  All users in this task had at least one valid community connection.")
    else:
        print(f"  (Note: {num_users_with_zero_vector} users had no community edges OR all their communities were dropped in the filtering step).")
    print("="*50 + "\n")

    # --- 6. Save to Parquet ---
    output_df = pd.DataFrame({'id': downstream_user_list, 'embedding': list(user_features)})
    pq.write_table(pa.Table.from_pandas(output_df, preserve_index=False), USER_EMBEDDINGS_OUTPUT_PATH, compression='snappy')
    print(f"Successfully saved static user features to: {USER_EMBEDDINGS_OUTPUT_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate static user features from pre-trained community embeddings.")
    parser.add_argument("--pretrain_data_path", required=True, help="Path to the pre-training graph_data directory.")
    parser.add_argument("--downstream_path", required=True, help="Path to the downstream task's graph_data directory.")
    parser.add_argument("--model_path", required=True, help="Full path to the saved contrastive_init_pretrained.pt model file.")
    args = parser.parse_args()
    
    create_user_features(args.pretrain_data_path, args.downstream_path, args.model_path)