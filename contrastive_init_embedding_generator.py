import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import pyarrow.parquet as pq
import os
import numpy as np
import sys
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import wandb

# --- Configuration ---
PROJECT_NAME = "commuinity_contrastive_init_embedding_training" 
PARENT_DATA_PATH = "../GASTON/pretraining/graph_data/"

# --- File Paths ---
UNWEIGHTED_EDGES_PARQUET_PATH = os.path.join(PARENT_DATA_PATH, f"user_community_active_edges.parquet")
ALL_USERS_PARQUET_PATH = os.path.join(PARENT_DATA_PATH, f"unique_user_nodes.parquet")

print(f"Using unweighted edges file: {os.path.basename(UNWEIGHTED_EDGES_PARQUET_PATH)}")

# Output Paths for the learned embeddings
USER_EMBEDDINGS_OUTPUT_PATH = os.path.join(PARENT_DATA_PATH, f"user_node_embeddings_contrastive.pt")
COMMUNITY_EMBEDDINGS_OUTPUT_PATH = os.path.join(PARENT_DATA_PATH, f"community_node_embeddings_contrastive.pt")
CONTR_INIT_MASTER_USER_IDS_PATH = os.path.join(PARENT_DATA_PATH, f"contr_init_user_ids_master.parquet")
CONTR_INIT_MASTER_COMMUNITY_IDS_PATH = os.path.join(PARENT_DATA_PATH, f"contr_init_community_ids_master.parquet") 

# Hyperparameters
EMBEDDING_DIM = 768
LEARNING_RATE = 1e-3
REGULARIZATION_LAMBDA = 1e-4 
EPOCHS = 30
BATCH_SIZE = 256
LOG_FREQUENCY = 10

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- 1. Data Preparation and Dataset ---
class UnweightedContrInitDataset(Dataset):
    def __init__(self, edges_path, all_users_path):
        print("Loading all unique users to define the user universe...")
        try:
            all_users_df = pq.read_table(all_users_path).to_pandas()
            self.unique_users = all_users_df.iloc[:, 0].unique()
        except FileNotFoundError:
            raise FileNotFoundError(f"Master user file not found at {all_users_path}.")
            
        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(self.unique_users)}
        self.num_users = len(self.unique_users)
        self.final_user_ids = list(self.unique_users)
        
        del all_users_df
        print(f"User universe defined with {self.num_users:,} unique users. Memory released.")

        print("Loading user-community edges for training examples...")
        try:
            df = pq.read_table(edges_path, columns=['source_id', 'target_id']).to_pandas()
            df.drop_duplicates(inplace=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"Edges file not found at {edges_path}.")
        
        self.unique_communities = df['target_id'].unique()
        self.comm_to_idx = {comm_id: idx for idx, comm_id in enumerate(self.unique_communities)}
        
        df = df[df['source_id'].isin(self.user_to_idx)]

        self.pos_user_indices = torch.tensor([self.user_to_idx[uid] for uid in df['source_id']], dtype=torch.long)
        self.pos_comm_indices = torch.tensor([self.comm_to_idx[cid] for cid in df['target_id']], dtype=torch.long)

        self.user_pos_comms = {}
        for u_idx, c_idx in zip(self.pos_user_indices.tolist(), self.pos_comm_indices.tolist()):
            if u_idx not in self.user_pos_comms:
                self.user_pos_comms[u_idx] = set()
            self.user_pos_comms[u_idx].add(c_idx)

        num_users_with_communities = len(self.user_pos_comms)
        num_orphan_users = self.num_users - num_users_with_communities
        print(f"\n--- [DEBUG CHECK] ---")
        print(f"Total users in user universe: {self.num_users:,}")
        print(f"Users with at least one community: {num_users_with_communities:,}")
        if num_orphan_users > 0:
            print(f"WARNING: Found {num_orphan_users:,} users with NO community interactions.")
        else:
            print("SUCCESS: All users have at least one community interaction.")
        print(f"-----------------------\n")
        
        self.num_communities = len(self.unique_communities)
        self.final_community_ids = list(self.unique_communities)
        
        del df
        print(f"Dataset loaded: {len(self)} unique positive pairs. Memory released.")

    def __len__(self):
        return len(self.pos_user_indices)

    def __getitem__(self, idx):
        return (self.pos_user_indices[idx], self.pos_comm_indices[idx])
    
# --- 2. Contrastive Model ---
class ContrInitModel(nn.Module):
    def __init__(self, num_users, num_communities, emb_dim):
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, emb_dim, sparse=True)
        self.community_embeddings = nn.Embedding(num_communities, emb_dim)
        nn.init.uniform_(self.user_embeddings.weight, a=-0.01, b=0.01)
        nn.init.uniform_(self.community_embeddings.weight, a=-0.01, b=0.01)

    def forward(self, u_indices, pos_c_indices, neg_c_indices):
        u_emb = self.user_embeddings(u_indices.to('cpu')).to(u_indices.device)
        c_pos_emb = self.community_embeddings(pos_c_indices)
        c_neg_emb = self.community_embeddings(neg_c_indices)
        u_emb_normalized = F.normalize(u_emb, p=2, dim=1)
        c_pos_emb_normalized = F.normalize(c_pos_emb, p=2, dim=1)
        c_neg_emb_normalized = F.normalize(c_neg_emb, p=2, dim=1)
        pos_scores = torch.sum(u_emb_normalized * c_pos_emb_normalized, dim=1) 
        neg_scores = torch.sum(u_emb_normalized * c_neg_emb_normalized, dim=1) 
        return pos_scores, neg_scores, u_emb, c_pos_emb, c_neg_emb

    def save_user_embeddings(self, user_path):
        torch.save(self.user_embeddings.weight.data.cpu(), user_path)
        print(f"User embeddings saved to: {user_path}")
        
    def save_community_embeddings(self, community_path):
        torch.save(self.community_embeddings.weight.data.cpu(), community_path)
        print(f"Community embeddings saved to: {community_path}")
        
# --- 3. Training Loop ---
def train_contrastive_encoder(model, dataloader, dataset, optimizer_gpu, optimizer_cpu):
    model.train()
    total_loss, total_ranking_loss, total_reg_loss = 0, 0, 0
    num_communities = dataset.num_communities 

    for u_indices, pos_c_indices in tqdm(dataloader, desc="Training contr_init"): 
        u_indices, pos_c_indices = u_indices.to(device), pos_c_indices.to(device)
        
        neg_c_indices_list = []
        for i in range(u_indices.size(0)):
            u_idx = u_indices[i].item()
            while True:
                neg_c_candidate = torch.randint(0, num_communities, (1,)).item()
                if neg_c_candidate not in dataset.user_pos_comms.get(u_idx, set()):
                    neg_c_indices_list.append(neg_c_candidate)
                    break
        neg_c_indices = torch.tensor(neg_c_indices_list, dtype=torch.long, device=device)
        
        optimizer_gpu.zero_grad()
        optimizer_cpu.zero_grad()
        
        pos_scores, neg_scores, u_emb, c_pos_emb, c_neg_emb = model(u_indices, pos_c_indices, neg_c_indices)
        
        ranking_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        
        l2_reg = REGULARIZATION_LAMBDA * (torch.norm(u_emb)**2 + torch.norm(c_pos_emb)**2 + torch.norm(c_neg_emb)**2)
        loss = ranking_loss + l2_reg
        
        loss.backward()
        optimizer_gpu.step()
        optimizer_cpu.step()
        
        total_loss += loss.item()
        total_ranking_loss += ranking_loss.item()
        total_reg_loss += l2_reg.item()

    return total_loss / len(dataloader), total_ranking_loss / len(dataloader), total_reg_loss / len(dataloader)

# --- 4. Main Execution ---
if __name__ == '__main__':
    wandb.init(
        project=f"{PROJECT_NAME}",
        config={"embedding_dim": EMBEDDING_DIM, "learning_rate": LEARNING_RATE, "regularization_lambda": REGULARIZATION_LAMBDA,
                "epochs": EPOCHS, "batch_size": BATCH_SIZE, "negative_sampling": "True_Negative_Resampling (N=1)", "device": str(device)}
    )

    pretrain_dataset = UnweightedContrInitDataset(UNWEIGHTED_EDGES_PARQUET_PATH, ALL_USERS_PARQUET_PATH)

    user_id_df = pd.DataFrame({'id': pretrain_dataset.final_user_ids})
    user_id_df.to_parquet(CONTR_INIT_MASTER_USER_IDS_PATH, index=False)
    print(f"Master contr_init User IDs ({len(user_id_df):,}) saved to: {CONTR_INIT_MASTER_USER_IDS_PATH}")

    comm_id_df = pd.DataFrame({'id': pretrain_dataset.final_community_ids})
    comm_id_df.to_parquet(CONTR_INIT_MASTER_COMMUNITY_IDS_PATH, index=False)
    print(f"Master contr_init Community IDs ({len(comm_id_df):,}) saved to: {CONTR_INIT_MASTER_COMMUNITY_IDS_PATH}")
        
    pretrain_dataloader = DataLoader(
        pretrain_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=8, pin_memory=True
    )

    model = ContrInitModel(pretrain_dataset.num_users, pretrain_dataset.num_communities, EMBEDDING_DIM)
    model.to(device)
    model.user_embeddings.to('cpu')
    print("Model loaded to GPU, with user_embeddings explicitly on CPU.")
    
    optimizer_gpu = optim.Adam(model.community_embeddings.parameters(), lr=LEARNING_RATE)
    optimizer_cpu = optim.SparseAdam(model.user_embeddings.parameters(), lr=LEARNING_RATE)
    
    print("\nStarting Unweighted Contrastive Initialization Pre-training...")

    best_loss = float('inf')
    MODEL_CHECKPOINT_PATH = os.path.join(PARENT_DATA_PATH, f"contr_init_best_model_state.pt") 

    for epoch in range(1, EPOCHS + 1):
        epoch_loss, epoch_ranking_loss, epoch_reg_loss = train_contrastive_encoder(
            model, pretrain_dataloader, pretrain_dataset, optimizer_gpu, optimizer_cpu
        )
        
        wandb.log({"epoch": epoch, "Total_Loss": epoch_loss, "Ranking_Loss (Unweighted)": epoch_ranking_loss,
                   "Regularization_Loss (L2)": epoch_reg_loss})
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), MODEL_CHECKPOINT_PATH)
            
        if epoch % LOG_FREQUENCY == 0 or epoch == EPOCHS:
            print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {epoch_loss:.4f} | Ranking Loss: {epoch_ranking_loss:.4f} | Reg: {epoch_reg_loss:.6f}")

    print("\nContrastive Initialization Pre-training complete.")
    
    if os.path.exists(MODEL_CHECKPOINT_PATH):
         model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH))
         
    model.save_user_embeddings(USER_EMBEDDINGS_OUTPUT_PATH)
    model.save_community_embeddings(COMMUNITY_EMBEDDINGS_OUTPUT_PATH)
    
    wandb.finish()
    print("\nInitialization preparation finished.")