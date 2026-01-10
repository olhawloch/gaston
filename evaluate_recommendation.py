import torch
import os
import argparse
from tqdm import tqdm
from collections import defaultdict
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
import gc

# --- Configuration ---
# These must match the main script
FWD_EDGE_TYPE = ('user', 'active_in', 'community')
# Fallback if that doesn't exist
ALT_EDGE_TYPE = ('user', 'active', 'community') 

K_VALUES = [10, 20, 50] # K values for Recall, NDCG, and MRR
EVAL_BATCH_SIZE = 1024  # Number of users to process at once

# --- Helper Functions for Metrics ---

@torch.no_grad()
def get_ground_truth(data_obj):
    """Creates a dictionary mapping user_id -> set of ground_truth community_ids."""
    links_dict = defaultdict(set)
    
    # Determine edge type
    edge_type = FWD_EDGE_TYPE
    if FWD_EDGE_TYPE not in data_obj.edge_types:
        if ALT_EDGE_TYPE in data_obj.edge_types:
            edge_type = ALT_EDGE_TYPE
        else:
            raise ValueError(f"Could not find valid edge type in {data_obj.edge_types}")

    # Get the supervision links from the test data
    # NOTE: Use edge_label_index for test data if available, else edge_index
    if hasattr(data_obj[edge_type], 'edge_label_index'):
        edge_index = data_obj[edge_type].edge_label_index
    else:
        edge_index = data_obj[edge_type].edge_index
    
    for i in range(edge_index.size(1)):
        user_node_idx = edge_index[0, i].item()
        comm_node_idx = edge_index[1, i].item()
        links_dict[user_node_idx].add(comm_node_idx)
        
    return links_dict

@torch.no_grad()
def get_train_links(data_obj):
    """Creates a dictionary mapping user_id -> set of training community_ids for filtering."""
    links_dict = defaultdict(set)
    
    # Determine edge type
    edge_type = FWD_EDGE_TYPE
    if FWD_EDGE_TYPE not in data_obj.edge_types:
         if ALT_EDGE_TYPE in data_obj.edge_types:
            edge_type = ALT_EDGE_TYPE
    
    # Get the message-passing links from the train data
    edge_index = data_obj[edge_type].edge_index
    
    for i in range(edge_index.size(1)):
        user_node_idx = edge_index[0, i].item()
        comm_node_idx = edge_index[1, i].item()
        links_dict[user_node_idx].add(comm_node_idx)
        
    return links_dict

@torch.no_grad()
def calculate_metrics_batched(Z_user, Z_community, test_links, train_links, k_values, batch_size):
    """Calculates metrics in batches to avoid OOM."""
    
    recalls = {k: [] for k in k_values}
    ndcgs = {k: [] for k in k_values}
    mrrs = {k: [] for k in k_values} # Added MRR storage
    
    max_k = max(k_values)

    # Identify test users (users who actually have test labels)
    test_user_ids = list(test_links.keys())
    num_test_users = len(test_user_ids)
    
    print(f"Calculating metrics for {num_test_users} test users in batches of {batch_size}...")

    # Transpose community embeddings once for fast multiplication
    # Shape: [768, num_communities]
    Z_comm_T = Z_community.T 

    for start_idx in tqdm(range(0, num_test_users, batch_size)):
        end_idx = min(start_idx + batch_size, num_test_users)
        batch_user_ids = test_user_ids[start_idx:end_idx]
        
        # 1. Get embeddings for this batch of users
        # Shape: [batch_size, 768]
        batch_user_emb = Z_user[batch_user_ids]
        
        # 2. Compute scores for this batch against ALL communities
        # Shape: [batch_size, num_communities]
        batch_scores = batch_user_emb @ Z_comm_T
        
        # 3. Process each user in the batch
        for i, user_id in enumerate(batch_user_ids):
            user_scores = batch_scores[i] # shape: [num_communities]
            true_communities = test_links[user_id]
            
            # Filter training links (mask them out)
            if user_id in train_links:
                filter_indices = list(train_links[user_id])
                user_scores[filter_indices] = -1e9 
            
            # Get top K recommendations
            _, top_k_indices = torch.topk(user_scores, k=max_k)
            top_k_indices = top_k_indices.tolist()

            # --- Calculate metrics for each K ---
            for k in k_values:
                top_k_at_k = top_k_indices[:k]
                top_k_set = set(top_k_at_k)
                
                # Recall@K
                hits = len(true_communities.intersection(top_k_set))
                if len(true_communities) > 0:
                    recalls[k].append(hits / len(true_communities))
                else:
                    recalls[k].append(0.0)

                # NDCG@K
                dcg = 0.0
                for rank, item_id in enumerate(top_k_at_k):
                    if item_id in true_communities:
                        dcg += 1.0 / torch.log2(torch.tensor(rank + 2.0))
                
                idcg = 0.0
                ideal_hits = min(k, len(true_communities))
                for rank in range(ideal_hits):
                    idcg += 1.0 / torch.log2(torch.tensor(rank + 2.0))
                
                if idcg > 0:
                    ndcgs[k].append((dcg / idcg).item())
                else:
                    ndcgs[k].append(0.0)
                    
                # MRR@K
                # Score is 1 / (rank + 1) of the *first* relevant item found
                mrr_score = 0.0
                for rank, item_id in enumerate(top_k_at_k):
                    if item_id in true_communities:
                        mrr_score = 1.0 / (rank + 1.0)
                        break # Stop at the first relevant item
                mrrs[k].append(mrr_score)

    # Average metrics
    avg_recalls = {k: (sum(v) / len(v) if len(v) > 0 else 0) for k, v in recalls.items()}
    avg_ndcgs = {k: (sum(v) / len(v) if len(v) > 0 else 0) for k, v in ndcgs.items()}
    avg_mrrs = {k: (sum(v) / len(v) if len(v) > 0 else 0) for k, v in mrrs.items()}
    
    return avg_recalls, avg_ndcgs, avg_mrrs

# --- Main Evaluation ---
def main(run_dir):
    print(f"Starting offline evaluation for run: {run_dir}")
    
    embedding_path = os.path.join(run_dir, "final_embeddings.pt")
    train_data_path = os.path.join(run_dir, "train_data_for_eval.pt")
    test_data_path = os.path.join(run_dir, "test_data_for_eval.pt")

    # --- 1. Load Data ---
    print("Loading evaluation data...")
    if not all(os.path.exists(p) for p in [embedding_path, train_data_path, test_data_path]):
        print(f"Error: Missing data files in {run_dir}.")
        return

    # Load embeddings
    final_embeddings = torch.load(embedding_path, map_location='cpu', weights_only=False)
    Z_user = final_embeddings['user'].float() # Ensure float32
    Z_community = final_embeddings['community'].float()
    print(f"Embeddings loaded: Users {Z_user.shape}, Communities {Z_community.shape}")

    # Load graph data
    train_data = torch.load(train_data_path, map_location='cpu', weights_only=False)
    test_data = torch.load(test_data_path, map_location='cpu', weights_only=False)
    
    # --- 2. Get Ground Truth ---
    print("Processing ground truth links...")
    test_links = get_ground_truth(test_data)
    train_links = get_train_links(train_data)
    
    print(f"Found {len(test_links)} users with test links.")
    print(f"Found {len(train_links)} users with train links to filter.")
    
    # Free up memory from graph objects
    del train_data
    del test_data
    gc.collect()

    # --- 3. Calculate Metrics Batched ---
    avg_recalls, avg_ndcgs, avg_mrrs = calculate_metrics_batched(
        Z_user, Z_community, test_links, train_links, K_VALUES, EVAL_BATCH_SIZE
    )
    
    print("\n--- Final Full-Graph Evaluation Metrics ---")
    for k in K_VALUES:
        print(f"K={k}:")
        print(f"  Recall@{k:<2}: {avg_recalls[k]:.4f}")
        print(f"  NDCG@{k:<2}  : {avg_ndcgs[k]:.4f}")
        print(f"  MRR@{k:<2}   : {avg_mrrs[k]:.4f}")
    print("---------------------------------------------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Offline Evaluation for Recommendation")
    parser.add_argument('--run_dir', type=str, required=True, help="Path to the model save directory")
    args = parser.parse_args()
    
    main(args.run_dir)