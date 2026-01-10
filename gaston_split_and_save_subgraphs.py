import os
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
import torch_geometric.transforms as T
from tqdm import tqdm
import shutil
import argparse
import random
import numpy as np

# Hyperparameters
BATCH_SIZE = 128
NUM_WORKERS = 2
TRAIN_SPLIT_RATIO = 0.8
VAL_TEST_SPLIT_RATIO = 0.5
RANDOM_SEED = 42

NUM_NEIGHBORS = {
    ('text', 'post_in', 'community'): [25, 10, 5],
    ('community', 'rev_post_in', 'text'): [25, 10, 5],
    
    ('text', 'post_by', 'user'): [25, 10, 5],
    ('user', 'rev_post_by', 'text'): [25, 10, 5],
    
    ('user', 'active', 'community'): [25, 10, 5],
    ('community', 'rev_active', 'user'): [25, 10, 5],
}

def clear_features(batch):
    """Removes the .x feature matrices from all node types in a batch."""
    for node_store in batch.node_stores:
        if 'x' in node_store:
            del node_store['x']
    return batch

# --- Main Execution ---
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate and save subgraph batches from a heterogeneous graph.")
    parser.add_argument(
        '--data_path', 
        type=str, 
        default="../GASTON/pretraining/graph_data/",
        help="Base directory containing graph data and for saving batches/temp files."
    )
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    DATA_PATH = args.data_path
    PROCESSED_GRAPH_PATH = os.path.join(DATA_PATH, "processed/contrastive_init_heterogenous_graph_Gemma.pt")
    SAVE_DIR = os.path.join(DATA_PATH, "contr_init_pregenerated_batches_Gemma")
    TEMP_SUBGRAPH_DIR = os.path.join(DATA_PATH, "temp_subgraphs_contr_init_Gemma")

    print(f"--- Using Base Data Path: {DATA_PATH} ---")
    print(f"Processed Graph: {PROCESSED_GRAPH_PATH}")
    print(f"Save Directory: {SAVE_DIR}")
    print(f"Temp Subgraph Directory: {TEMP_SUBGRAPH_DIR}")
    print(f"----------------------------------------")

    print("Loading heterogeneous graph...")
    
    if not os.path.exists(PROCESSED_GRAPH_PATH):
        raise FileNotFoundError(f"Missing processed graph file: {PROCESSED_GRAPH_PATH}. Please run the graph generator script first.")
        
    data = torch.load(PROCESSED_GRAPH_PATH, weights_only=False)
    
    # --- Step 0: Cleanup Old Subgraphs ---
    train_subgraph_path = os.path.join(TEMP_SUBGRAPH_DIR, "train_subgraph.pt")
    val_test_subgraph_path = os.path.join(TEMP_SUBGRAPH_DIR, "val_test_subgraph.pt")

    if os.path.exists(train_subgraph_path) or os.path.exists(val_test_subgraph_path):
        print(f"ATTENTION: Deleting previous subgraphs in {TEMP_SUBGRAPH_DIR} to ensure schema consistency.")
        if os.path.exists(train_subgraph_path): os.remove(train_subgraph_path)
        if os.path.exists(val_test_subgraph_path): os.remove(val_test_subgraph_path)
    
    os.makedirs(TEMP_SUBGRAPH_DIR, exist_ok=True)
    
    # --- Step 1: Split and Save Subgraphs ---
    if not os.path.exists(train_subgraph_path) or not os.path.exists(val_test_subgraph_path):
        print("\nCreating training and validation/test subgraphs and saving to disk...")
        user_indices = torch.randperm(data['user'].num_nodes)
        train_split_size_user = int(TRAIN_SPLIT_RATIO * data['user'].num_nodes)
        val_test_user_indices = user_indices[train_split_size_user:]
        train_user_indices = user_indices[:train_split_size_user]

        edge_storage = data['text', 'post_by', 'user']
        if not hasattr(edge_storage, 'edge_index') or edge_storage.num_edges == 0:
            raise AttributeError("Required 'edge_index' is missing or empty in the base graph. Check generator output.")
            
        text_to_user_edge_index = edge_storage.edge_index
        
        # Determine which text nodes belong to the train/val_test sets based on their corresponding user
        source_texts_train = text_to_user_edge_index[0]
        target_users_train = text_to_user_edge_index[1]
        
        train_text_mask = torch.isin(target_users_train, train_user_indices)
        train_text_indices = source_texts_train[train_text_mask].unique() 
        
        val_test_text_mask = torch.isin(target_users_train, val_test_user_indices)
        val_test_text_indices = source_texts_train[val_test_text_mask].unique() 

        all_community_indices = torch.arange(data['community'].num_nodes)
        
        # Create subgraphs with only the split nodes
        train_subgraph = data.subgraph({
            'user': train_user_indices, 
            'text': train_text_indices,
            'community': all_community_indices
        })
        
        val_test_subgraph = data.subgraph({
            'user': val_test_user_indices, 
            'text': val_test_text_indices,
            'community': all_community_indices
        })
        
        torch.save(train_subgraph, train_subgraph_path)
        torch.save(val_test_subgraph, val_test_subgraph_path)
        print("Subgraphs saved successfully.")
        
    # --- Step 2: Load Subgraphs and Apply T.ToUndirected (Creation of reverse edges) ---
    print("\nLoading subgraphs from disk and applying T.ToUndirected...")
    
    train_subgraph = torch.load(train_subgraph_path, weights_only=False)
    val_test_subgraph = torch.load(val_test_subgraph_path, weights_only=False)
    
    train_subgraph = T.ToUndirected(merge=True)(train_subgraph)
    val_test_subgraph = T.ToUndirected(merge=True)(val_test_subgraph)
    
    print(f"Train subgraph edge types (should be 6): {list(train_subgraph.edge_types)}")
    print("Subgraphs loaded and prepared successfully.")

    # Split Val/Test user nodes from the val_test_subgraph
    val_split_size_user = int(VAL_TEST_SPLIT_RATIO * val_test_subgraph['user'].num_nodes)
    val_user_nodes_subgraph = torch.arange(val_test_subgraph['user'].num_nodes)[:val_split_size_user]
    test_user_nodes_subgraph = torch.arange(val_test_subgraph['user'].num_nodes)[val_split_size_user:]

    os.makedirs(os.path.join(SAVE_DIR, 'train'), exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, 'val'), exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, 'test'), exist_ok=True)

    print("\nStarting batch generation and saving...")

    remove_isolated_nodes = T.RemoveIsolatedNodes()

    # --- Generate Training Batches ---
    print("\nProcessing train split...")
    train_loader = NeighborLoader(
        train_subgraph,
        num_neighbors=NUM_NEIGHBORS, 
        batch_size=BATCH_SIZE,
        input_nodes=('user', torch.arange(train_subgraph['user'].num_nodes)),
        shuffle=True,
        num_workers=NUM_WORKERS,
        persistent_workers=False,
        pin_memory=False,
    )
    for i, batch in enumerate(tqdm(train_loader, desc="Generating train batches")):
        save_path = os.path.join(SAVE_DIR, 'train', f'batch_{i}.pt')
        batch = remove_isolated_nodes(batch)
        batch = clear_features(batch)
        torch.save(batch, save_path)
    
    # --- Generate Validation Batches ---
    print("\nProcessing val split...")
    val_loader = NeighborLoader(
        val_test_subgraph,
        num_neighbors=NUM_NEIGHBORS,
        batch_size=BATCH_SIZE,
        input_nodes=('user', val_user_nodes_subgraph),
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=False,
        pin_memory=False,
    )
    for i, batch in enumerate(tqdm(val_loader, desc="Generating val batches")):
        save_path = os.path.join(SAVE_DIR, 'val', f'batch_{i}.pt')
        batch = remove_isolated_nodes(batch)
        batch = clear_features(batch)
        torch.save(batch, save_path)
    
    # --- Generate Test Batches ---
    print("\nProcessing test split...")
    test_loader = NeighborLoader(
        val_test_subgraph,
        num_neighbors=NUM_NEIGHBORS,
        batch_size=BATCH_SIZE,
        input_nodes=('user', test_user_nodes_subgraph),
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=False,
        pin_memory=False,
    )
    for i, batch in enumerate(tqdm(test_loader, desc="Generating test batches")):
        save_path = os.path.join(SAVE_DIR, 'test', f'batch_{i}.pt')
        batch = remove_isolated_nodes(batch)
        batch = clear_features(batch)
        torch.save(batch, save_path)
            
    print("\nBatch generation complete! All batches saved to:", SAVE_DIR)