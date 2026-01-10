import torch
from torch_geometric.data import HeteroData
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def get_degree_stats(edge_index, num_nodes, direction='source'):
    """
    Calculates degree statistics for nodes.
    direction: 'source' (out-degree) or 'target' (in-degree)
    """
    row_idx = 0 if direction == 'source' else 1
    indices = edge_index[row_idx]
    
    # Calculate degrees
    degrees = torch.bincount(indices, minlength=num_nodes).float()
    
    # Convert to numpy for easy stats
    deg_np = degrees.cpu().numpy()
    
    stats = {
        'min': np.min(deg_np),
        'max': np.max(deg_np),
        'mean': np.mean(deg_np),
        'median': np.median(deg_np),
        'std': np.std(deg_np),
        'p90': np.percentile(deg_np, 90),
        'p99': np.percentile(deg_np, 99),
        'zero_degree_count': np.sum(deg_np == 0),
        'non_zero_count': np.sum(deg_np > 0)
    }
    
    return stats, deg_np

def plot_degree_distribution(degrees, title, save_path):
    plt.figure(figsize=(10, 6))
    plt.hist(degrees, bins=50, log=True, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel("Degree (Log Scale)")
    plt.ylabel("Frequency (Log Scale)")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(save_path)
    plt.close()
    print(f"  -> Plot saved to {save_path}")

def main(args):
    print(f"--- Analyzing Graph: {args.graph_path} ---")
    
    if not os.path.exists(args.graph_path):
        print("Error: File not found.")
        return

    # Load Graph
    data = torch.load(args.graph_path, weights_only=False)
    print("Graph loaded successfully.\n")

    # --- 1. Basic Node Counts ---
    print("=== Node Counts ===")
    total_nodes = 0
    for node_type in data.node_types:
        count = data[node_type].num_nodes
        total_nodes += count
        feat_dim = data[node_type].x.shape[1] if hasattr(data[node_type], 'x') and data[node_type].x is not None else "None"
        print(f"  {node_type:<15}: {count:10,} nodes | Feature Dim: {feat_dim}")
    print("-" * 30)

    # --- 2. Basic Edge Counts ---
    print("\n=== Edge Counts ===")
    for edge_type in data.edge_types:
        count = data[edge_type].edge_index.size(1)
        print(f"  {str(edge_type):<40}: {count:10,} edges")
    print("-" * 30)

    # --- 3. User-Community Interaction Analysis ---
    print("\n=== User <-> Community Interaction Stats ===")
    
    # Identify the interaction edge
    target_edge = None
    possible_edges = [('user', 'active_in', 'community'), ('user', 'active', 'community')]
    
    for et in possible_edges:
        if et in data.edge_types:
            target_edge = et
            break
            
    if target_edge:
        print(f"Analyzing edge type: {target_edge}")
        edge_index = data[target_edge].edge_index
        num_users = data['user'].num_nodes
        num_comms = data['community'].num_nodes
        
        # A. User Degrees (How many communities does a user interact with?)
        user_stats, user_degrees = get_degree_stats(edge_index, num_users, direction='source')
        
        print("\n[User Activity Stats]")
        print(f"  Avg communities per user : {user_stats['mean']:.4f}")
        print(f"  Median communities       : {user_stats['median']:.1f}")
        print(f"  Max communities          : {user_stats['max']:.0f}")
        print(f"  Top 10% users joined >   : {user_stats['p90']:.1f} communities")
        print(f"  Top 1% users joined >    : {user_stats['p99']:.1f} communities")
        print(f"  Users with 0 communities : {user_stats['zero_degree_count']:,} ({user_stats['zero_degree_count']/num_users*100:.2f}%)")

        if args.plot:
            plot_degree_distribution(user_degrees, f"User Degree Distribution ({args.task_name})", "user_degree_dist.png")

        # B. Community Degrees (How many users are in a community?)
        comm_stats, comm_degrees = get_degree_stats(edge_index, num_comms, direction='target')
        
        print("\n[Community Popularity Stats]")
        print(f"  Avg users per community  : {comm_stats['mean']:.4f}")
        print(f"  Median users             : {comm_stats['median']:.1f}")
        print(f"  Max users                : {comm_stats['max']:.0f}")
        print(f"  Communities with 0 users : {comm_stats['zero_degree_count']:,}")

        if args.plot:
            plot_degree_distribution(comm_degrees, f"Community Popularity Distribution ({args.task_name})", "comm_degree_dist.png")

        # C. Density / Sparsity
        num_interactions = edge_index.size(1)
        possible_interactions = num_users * num_comms
        density = num_interactions / possible_interactions if possible_interactions > 0 else 0
        sparsity = 100 * (1 - density)
        print(f"\n[Matrix Stats]")
        print(f"  Total Interactions       : {num_interactions:,}")
        print(f"  Matrix Density           : {density:.6f}")
        print(f"  Sparsity                 : {sparsity:.4f}%")

    else:
        print("WARNING: Could not find User-Community edge type (active/active_in).")

    # --- 4. Text-User Connectivity ---
    print("\n=== Text <-> User Connectivity ===")
    # Usually ('text', 'post_by', 'user')
    text_user_edge = ('text', 'post_by', 'user')
    if text_user_edge in data.edge_types:
        edge_index = data[text_user_edge].edge_index
        num_users = data['user'].num_nodes
        
        # User In-Degree (How many texts posted by user?)
        # For ('text', 'post_by', 'user'), User is target (index 1)
        stats, _ = get_degree_stats(edge_index, num_users, direction='target')
        
        print(f"  Avg posts per user       : {stats['mean']:.4f}")
        print(f"  Users with 0 posts       : {stats['zero_degree_count']:,} (Likely lurkers or from pre-training universe)")
        print(f"  Max posts by single user : {stats['max']:.0f}")
    else:
        print("  Edge ('text', 'post_by', 'user') not found.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_path', type=str, required=True, help="Path to the .pt file")
    parser.add_argument('--task_name', type=str, default="Graph", help="Name for plot titles")
    parser.add_argument('--plot', action='store_true', help="Generate histograms")
    args = parser.parse_args()
    
    main(args)