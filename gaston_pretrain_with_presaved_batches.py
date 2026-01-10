import torch
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear
from torch.utils.data import Dataset, DataLoader
import torch_geometric.transforms as T
from tqdm import tqdm
import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from datetime import datetime
from torch_scatter import scatter_mean

try:
    torch.multiprocessing.set_start_method('spawn', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')
except RuntimeError:
    pass

torch.set_float32_matmul_precision('high')

# --- Configuration ---
DATA_PATH = "../GASTON/pretraining/graph_data"
SAVE_DIR = os.path.join(DATA_PATH, "contr_init_pregenerated_batches_XLNet")
USER_COMMUNITY_AGG_INDEX_PATH = os.path.join(DATA_PATH, "user_community_agg_index_XLNet.pt")
PROCESSED_GRAPH_PATH = os.path.join(DATA_PATH, "processed/contrastive_init_heterogenous_graph_XLNet.pt")

RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
SAVE_MODEL_DIR = os.path.join(DATA_PATH, "model", RUN_TIMESTAMP)
SAVE_MODEL_PATH = os.path.join(SAVE_MODEL_DIR, "contrastive_init_pretrained_XLNet.pt")

# Hyperparameters
HIDDEN_CHANNELS = 768
NUM_HGT_LAYERS = 3
ATTENTION_HEADS = 8
LEARNING_RATE = 1e-4
EPOCHS = 6
BATCH_SIZE = 128
TEXT_MASKING_RATE = 0.50
LOSS_ALPHA = 0.5 
BETA = 1 - LOSS_ALPHA
GAMMA = 0.5 
CLIP_GRAD_NORM = 0.5
EDGE_MASKING_RATE = 0.5
NUM_WORKERS = 8
DROPOUT_RATE = 0.2

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Custom Dataset for Pre-generated Batches ---
class FileDataset(Dataset):
    def __init__(self, directory):
        self.file_list = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pt')]
        self.file_list.sort() 
        if not self.file_list:
            raise FileNotFoundError(f"No .pt files found in {directory}. Please run data preparation first.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        return torch.load(self.file_list[idx], weights_only=False)

# --- HGT Model Definition ---
class HGTModel(torch.nn.Module):
    def __init__(self, in_channels: dict, hidden_channels, num_hgt_layers, num_heads, metadata):
        super().__init__()
        
        self.lin_dict = torch.nn.ModuleDict()
        self.node_types = metadata[0]
        
        for node_type in self.node_types:
            in_dim = in_channels.get(node_type)
            if in_dim is None:
                 raise ValueError(f"Missing input dimension for node type: {node_type}")
            self.lin_dict[node_type] = Linear(in_dim, hidden_channels)
            
        self.convs = torch.nn.ModuleList()
        for _ in range(num_hgt_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata, num_heads)
            self.convs.append(conv)

        self.out_channels = hidden_channels
        self.dropout = torch.nn.Dropout(p=DROPOUT_RATE)

    def forward(self, x_dict, edge_index_dict):
        x_dict_processed = {}
        
        for node_type in self.node_types:
            x_raw = x_dict[node_type]
            x_dict_processed[node_type] = self.lin_dict[node_type](x_raw).relu() 

        for conv in self.convs:
            x_dict_processed = conv(x_dict_processed, edge_index_dict)
            for node_type in self.node_types:
                x_dict_processed[node_type] = self.dropout(x_dict_processed[node_type])
                
        return x_dict_processed

# --- PyTorch Lightning Module ---
class LitPreTrainingModel(pl.LightningModule):
    def __init__(self, in_channels: dict, metadata, text_features,
                 initial_comm_features, user_agg_map, 
                 hidden_channels, num_hgt_layers, attention_heads,
                 learning_rate, loss_alpha, beta, text_masking_rate,
                 edge_masking_rate, comm_emb_trainable: bool = True):
        super().__init__()
        self.save_hyperparameters(ignore=["metadata", "in_channels", "text_features", "initial_comm_features", "user_agg_map"]) 
        
        self.text_features = text_features
        self.metadata = metadata
        num_communities = initial_comm_features.size(0)
        embedding_dim = initial_comm_features.size(1)
        
        self.community_embedding = torch.nn.Embedding(num_communities, embedding_dim)
        with torch.no_grad():
            self.community_embedding.weight.data.copy_(initial_comm_features)

        # make the community embeddings trainable or not
        self.community_embedding.weight.requires_grad = self.hparams.comm_emb_trainable

        self.num_users = user_agg_map['user_indices'].max().item() + 1
        
        # --- TEMPORARY LINES FOR VALIDATION ---
        self.user_indices = user_agg_map['user_indices'].to('cpu')
        self.community_indices = user_agg_map['community_indices'].to('cpu')
        # --- END TEMPORARY LINES ---
        
        print("Pre-computing user-community sparse adjacency matrix (CSR)...")
        user_idx_cpu = user_agg_map['user_indices']
        comm_idx_cpu = user_agg_map['community_indices']
        values = torch.ones(user_idx_cpu.numel(), dtype=torch.float32)
        
        adj_coo = torch.sparse_coo_tensor(
            indices=torch.stack([user_idx_cpu, comm_idx_cpu]),
            values=values,
            size=(self.num_users, num_communities),
            dtype=torch.float32
        )
        
        adj_csr = adj_coo.to_sparse_csr()
        
        print("...Calculating user degrees from CSR structure...")
        crow_indices = adj_csr.crow_indices()
        user_degrees_1d = (crow_indices[1:] - crow_indices[:-1]).to(torch.float32)
        user_degrees = user_degrees_1d.clamp(min=1).unsqueeze(-1)
        
        # --- NEW LOGIC: Fork based on 'comm_emb_trainable' ---
        if self.hparams.comm_emb_trainable:
            print("...Strategy: Using fast scatter_mean (sorting indices for fast lookup)...")
            # Sort the user-comm map ONCE for fast O(log N) batch lookup
            # This is for the memory-efficient scatter_mean path
            perm = torch.argsort(self.user_indices)
            self.sorted_user_indices = self.user_indices[perm].to(device)
            self.sorted_comm_indices = self.community_indices[perm].to(device)
            print("...Sorting complete.")
        else:
            print("...Strategy: Pre-computing all user features (embeddings are frozen).")
            # Embeddings are frozen, so we can calculate all user features ONCE.
            # This is the fastest possible path.
            summed_user_features_ALL = torch.sparse.mm(adj_csr, initial_comm_features.float())
            user_features_ALL = summed_user_features_ALL / user_degrees
            
            # Store the final, pre-computed user features on the GPU
            self.precomputed_user_features = user_features_ALL.to(device)
            
            # We can delete the CSR matrix and degrees from memory
            # del self.user_indices, self.community_indices
            del adj_coo, adj_csr, user_degrees
            
        print("...Sparse matrix pre-computation complete.")

        in_channels_for_hgt = {
            'text': self.text_features.size(1),
            'user': embedding_dim,
            'community': embedding_dim
        }

        self.hgt = HGTModel(
            in_channels=in_channels_for_hgt,
            hidden_channels=self.hparams.hidden_channels,
            num_hgt_layers=self.hparams.num_hgt_layers,
            num_heads=self.hparams.attention_heads,
            metadata=metadata
        )
        
        text_feature_size = self.text_features.size(1)
        self.text_decoder = torch.nn.Linear(self.hparams.hidden_channels, text_feature_size)
        self.mask_embedding = torch.nn.Parameter(torch.randn(text_feature_size))

    def _re_attach_features_OLD(self, batch):
        """
        The ORIGINAL slow, CPU-based method for validation.
        """
        # 1. Attach static text features.
        batch['text'].x = self.text_features[batch['text'].n_id]
        
        # 2. Attach the current trainable community embeddings for the batch.
        X_community = self.community_embedding.weight
        batch['community'].x = X_community[batch['community'].n_id]
        
        # 3. Efficiently calculate user features ONLY for users in this batch.
        batch_user_n_ids = batch['user'].n_id 
        
        # This mask finds *which* of those 7.5M links belong to the users in *this* batch.
        mask = torch.isin(self.user_indices, batch_user_n_ids.to('cpu'))
        
        # Filter the global map to get only the relevant user-comm pairs for this batch.
        batch_agg_user_indices = self.user_indices[mask] 
        batch_agg_comm_indices = self.community_indices[mask]
        
        # Create a mapping from a user's global ID to their local batch index.
        global_to_local_user_map = {global_id.item(): local_id for local_id, global_id in enumerate(batch_user_n_ids)}
        
        # Convert the list of global user IDs (batch_agg_user_indices) into the
        # required 'index' tensor of local batch indices for scatter_mean.
        local_user_indices = torch.tensor([global_to_local_user_map[uid.item()] for uid in batch_agg_user_indices], dtype=torch.long, device=self.device)

        # Perform the scatter_mean operation.
        batch_user_features = scatter_mean(
            src=X_community[batch_agg_comm_indices.to(self.device)], # The embeddings to be averaged
            index=local_user_indices,  # How to group the embeddings
            dim=0,                     # Perform the mean across dimension 0
            dim_size=batch['user'].num_nodes # Set the output size
        )
        
        batch['user'].x = batch_user_features
        
        return batch

    def re_attach_features(self, batch):
        """
        Rehydrates a blueprint batch by attaching static text features and
        generating dynamic user and community features ONLY for the nodes in the batch.
        """
        # 1. Attach static text features.
        # 'text_features' is the master tensor (e.g., [1M, 768]) on the GPU.
        # 'batch['text'].n_id' are the global indices of text nodes in this batch.
        # This line efficiently grabs the features for just the texts in this batch.
        batch['text'].x = self.text_features[batch['text'].n_id]
        
        # 2. Attach the current trainable community embeddings for the batch.
        # 'self.community_embedding.weight' is the full, trainable embedding table (e.g., [6k, 768]).
        X_community = self.community_embedding.weight
        # 'batch['community'].n_id' are the global indices of communities in this batch.
        # This line grabs the current embeddings for just the communities in this batch.
        batch['community'].x = X_community[batch['community'].n_id]

        # 3. Efficiently calculate user features for this batch.
        batch_user_n_ids = batch['user'].n_id 

        if self.hparams.comm_emb_trainable:
            # --- STRATEGY: DYNAMIC (Trainable) - FAST SCATTER_MEAN ---
            # This is the memory-efficient path.
            
            # 1. Find the start (left) and end (right) boundaries for each user
            #    in our pre-sorted edge list. This is O(log N) and very fast.
            lower_bounds = torch.searchsorted(self.sorted_user_indices, batch_user_n_ids, side='left')
            upper_bounds = torch.searchsorted(self.sorted_user_indices, batch_user_n_ids, side='right')

            # 2. Build the 'src' and 'index' tensors for scatter_mean.
            #    We must loop over the batch users (e.g., ~10k), which is
            #    infinitely faster than looping over the edge list (20.6M).
            all_comm_indices = []
            local_user_indices = []
            
            for local_id, (start, end) in enumerate(zip(lower_bounds, upper_bounds)):
                num_comms = end - start
                if num_comms > 0:
                    # Get the global comm IDs for this user
                    comm_ids_for_user = self.sorted_comm_indices[start:end]
                    all_comm_indices.append(comm_ids_for_user)
                    
                    # Create the index tensor mapping to this local_id
                    local_user_indices.append(torch.full((num_comms,), local_id, device=self.device, dtype=torch.long))

            # 3. Check if any links were found at all
            if not local_user_indices:
                # No users in this batch have links, return zero-vectors
                batch['user'].x = torch.zeros(batch['user'].num_nodes, X_community.size(1), device=self.device, dtype=X_community.dtype)
                return batch

            # 4. Concatenate all indices
            final_comm_indices = torch.cat(all_comm_indices)
            final_local_indices = torch.cat(local_user_indices)
            
            # 5. Get the actual embeddings for all communities to be averaged
            src_embeddings = X_community[final_comm_indices]

            # 6. Compute the mean
            batch_user_features = scatter_mean(
                src=src_embeddings,
                index=final_local_indices,
                dim=0,
                dim_size=batch['user'].num_nodes
            )
            
        else:
            # --- STRATEGY: STATIC (Embeddings are frozen) ---
            # We simply look up the features we pre-computed in __init__.
            # This is the fastest possible method.
            batch_user_features = self.precomputed_user_features[batch_user_n_ids]

        # 5. Assign the computed features to the batch object.
        batch['user'].x = batch_user_features
    
        # # 3. Efficiently calculate user features ONLY for users in this batch.
        # # This is done by averaging the embeddings of all communities each user is connected to,
        # # based on the pre-computed 'user_agg_map'.

        # # Get the global IDs of all users present in this specific batch (e.g., [300, 100, 400])
        # batch_user_n_ids = batch['user'].n_id 
        
        # # 'self.user_indices' is the giant list of all user-comm links (e.g., [7.5M]).
        # # This 'mask' finds *which* of those 7.5M links belong to the users in *this* batch.
        # mask = torch.isin(self.user_indices, batch_user_n_ids.to('cpu'))
        
        # # Filter the global map to get only the relevant user-comm pairs for this batch.
        # # e.g., [100, 300, 300, 300, 400]
        # batch_agg_user_indices = self.user_indices[mask] 
        # # e.g., [  1,   1,   3,   4,   0]
        # batch_agg_comm_indices = self.community_indices[mask]
        
        # # Create a mapping from a user's global ID to their local batch index.
        # # This is crucial because scatter_mean needs local indices from 0 to batch_size-1.
        # # e.g., {300: 0, 100: 1, 400: 2}
        # global_to_local_user_map = {global_id.item(): local_id for local_id, global_id in enumerate(batch_user_n_ids)}
        
        # # Convert the list of global user IDs (batch_agg_user_indices) into the
        # # required 'index' tensor of local batch indices for scatter_mean.
        # # e.g., [  1,   0,   0,   0,   2]
        # local_user_indices = torch.tensor([global_to_local_user_map[uid.item()] for uid in batch_agg_user_indices], dtype=torch.long, device=self.device)

        # # We check if training has just started (global_step is low)
        # # if self.trainer is not None and self.trainer.global_step < 3:
        # #     print("\n" + "="*50)
        # #     print(f"--- DEBUGGING BATCH {self.trainer.global_step} (Epoch {self.current_epoch}) ---")
        # #     print(f"--- re_attach_features ---")
        # #     print(f"Num users in batch: {batch['user'].num_nodes}")
        # #     print(f"Num communities in batch: {batch['community'].num_nodes}")
            
        # #     # This counts how many communities each local user is connected to
        # #     if local_user_indices.numel() > 0:
        # #         unique_local_users, comm_counts_per_user = torch.unique(local_user_indices, return_counts=True)
        # #         print(f"Number of users being constructed: {unique_local_users.numel()}")
        # #         print(f"Community counts per user (first 20 users): {comm_counts_per_user[:20]}")
        # #         print(f"Avg communities per user in batch: {comm_counts_per_user.float().mean():.2f}")
                
        # #         # Check for users in the batch that are NOT in the agg_map
        # #         unmapped_users = batch['user'].num_nodes - unique_local_users.numel()
        # #         if unmapped_users > 0:
        # #             print(f"WARNING: {unmapped_users} users in the batch have NO entries in the agg_map and will get 0-vec features.")
        # #     else:
        # #         print("WARNING: No user-community links found in agg_map for ANY user in this batch.")
        # #     print("="*50 + "\n")

        # # Perform the scatter_mean operation.
        # # 'src' contains the community embeddings for every link (e.g., [emb_1, emb_1, emb_3, emb_4, emb_0]).
        # # 'index' tells scatter_mean how to group them (e.g., [  1,   0,   0,   0,   2]).
        # # 'dim_size' ensures the output tensor has the correct size (num_users_in_batch, embed_dim),
        # # even if some users have 0 connections (they get a 0-vector).
        # batch_user_features = scatter_mean(
        #     src=X_community[batch_agg_comm_indices.to(self.device)], # The embeddings to be averaged
        #     index=local_user_indices,  # How to group the embeddings
        #     dim=0,                     # Perform the mean across dimension 0
        #     dim_size=batch['user'].num_nodes # Set the output size
        # )
        
        # # The result is a [num_users_in_batch, embed_dim] tensor where each row
        # # 'i' is the average of all community embeddings for the user at local index 'i'.
        # # e.g., batch_user_features[0] = avg(emb_1, emb_3, emb_4)

        # # Finally, assign these newly computed features to the batch object.
        # batch['user'].x = batch_user_features
        
        return batch

    def forward(self, batch):
        return self.hgt(batch.x_dict, batch.edge_index_dict)
    
    def decode_text(self, text_embeddings):
        return self.text_decoder(text_embeddings)

    def text_reconstruction_loss(self, original_text_features, reconstructed_text_features):
        return F.mse_loss(reconstructed_text_features, original_text_features)
    
    def _prepare_text_reconstruction(self, batch):
        mask = torch.rand(batch['text'].num_nodes, device=self.device) < TEXT_MASKING_RATE
        if not mask.any():
            # Ensure at least one node is masked
            rand_index = torch.randint(0, mask.numel(), (1,), device=self.device)
            mask[rand_index] = True
        
        # Store original features for loss calculation
        original_text_features = batch['text'].x[mask].clone().detach()
        
        # Apply mask embedding to the batch
        batch['text'].x[mask] = self.mask_embedding.expand_as(batch['text'].x[mask])
        
        return original_text_features, mask

    def _calculate_text_reconstruction_loss(self, out_dict, original_text_features, mask, batch_idx):
        # Decode the embeddings of only the masked nodes
        reconstructed_text_features = self.decode_text(out_dict['text'][mask])
        
        # Calculate loss
        loss_reconstruction = self.text_reconstruction_loss(original_text_features, reconstructed_text_features)

        # Log number of masked texts
        num_nodes = out_dict['text'].size(0) 
        self.log('debug/num_masked_texts', mask.sum().float(), prog_bar=False, logger=True, batch_size=num_nodes)

        # Debug printing for early batches
        # if batch_idx < 3:
        #     print("\n" + "="*50)
        #     print(f"--- DEBUGGING BATCH {batch_idx} (Epoch {self.current_epoch}) ---")
        #     print("--- Text Reconstruction Tensors ---")
            
        #     print(f"\n1. Original Features (Ground Truth):")
        #     print(f"   Shape: {original_text_features.shape}")
        #     if original_text_features.numel() > 0:
        #         print(f"   Sample (first 5 values of 1st vector): {original_text_features[0, :5]}")

        #     gnn_output_for_masked = out_dict['text'][mask]
        #     print(f"\n2. GNN Output (Input to Decoder):")
        #     print(f"   Shape: {gnn_output_for_masked.shape}")
        #     if gnn_output_for_masked.numel() > 0:
        #         print(f"   Sample (first 5 values of 1st vector): {gnn_output_for_masked[0, :5]}")
                
        #     print(f"\n3. Reconstructed Features (Model's Guess):")
        #     print(f"   Shape: {reconstructed_text_features.shape}")
        #     if reconstructed_text_features.numel() > 0:
        #         print(f"   Sample (first 5 values of 1st vector): {reconstructed_text_features[0, :5]}")
        #     print("="*50 + "\n")
            
        return loss_reconstruction

    def _calculate_edge_generation_loss(self, batch, out_dict, step_prefix='train'):
        loss_edge_gen_total = torch.tensor(0.0, device=self.device)
        text_emb = out_dict['text'] # (T, 768)
        user_emb = out_dict['user'] # (U, 768)
        comm_emb = out_dict['community'] # (C, 768)
        have_loss_tc = False
        have_loss_tu = False
        have_loss_uc = False
        
        # Use num_nodes for consistent logging batch_size
        batch_size_for_logging = batch['text'].num_nodes
        
        edge_type_tc = ('text', 'post_in', 'community')
        edge_type_tu = ('text', 'post_by', 'user')
        edge_type_uc = ('user', 'active', 'community')

        # --- Text-to-Community (CrossEntropy) ---
        tc_edge_index = batch.edge_index_dict[edge_type_tc] # (2, E)
        unique_tc_edges = torch.unique(tc_edge_index, dim=1)
        text_comm_logits = torch.matmul(text_emb, comm_emb.t())
        logits_for_loss_tc = text_comm_logits[unique_tc_edges[0]]
        labels_for_loss_tc = unique_tc_edges[1]
        loss_tc = F.cross_entropy(logits_for_loss_tc, labels_for_loss_tc)
        loss_edge_gen_total += loss_tc
        have_loss_tc = True
        self.log(f'{step_prefix}_edge_loss_tc', loss_tc, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size_for_logging)

        # --- Text-to-User (CrossEntropy) ---
        tu_edge_index = batch.edge_index_dict[edge_type_tu]
        unique_tu_edges = torch.unique(tu_edge_index, dim=1)
        text_user_logits = torch.matmul(text_emb, user_emb.t())
        logits_for_loss_tu = text_user_logits[unique_tu_edges[0]]
        labels_for_loss_tu = unique_tu_edges[1]
        loss_tu = F.cross_entropy(logits_for_loss_tu, labels_for_loss_tu)
        loss_edge_gen_total += loss_tu
        have_loss_tu = True
        self.log(f'{step_prefix}_edge_loss_tu', loss_tu, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size_for_logging)

        # --- User-to-Community (BCE) ---
        uc_edge_index = batch.edge_index_dict[edge_type_uc]
        num_users = user_emb.size(0)
        num_comms = comm_emb.size(0)
        user_comm_logits = torch.matmul(user_emb, comm_emb.t())
        uc_labels = torch.zeros(num_users, num_comms, device=self.device)
        uc_labels[uc_edge_index[0], uc_edge_index[1]] = 1.0
        loss_uc = F.binary_cross_entropy_with_logits(user_comm_logits, uc_labels)
        loss_edge_gen_total += loss_uc
        have_loss_uc = True
        self.log(f'{step_prefix}_edge_loss_uc', loss_uc, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size_for_logging)

        # --- Accuracy Calculation ---
        with torch.no_grad():
            total_acc = 0.0
            num_acc_calcs = 0
            # Text-to-Community Accuracy
            if have_loss_tc:
                preds_tc = logits_for_loss_tc.argmax(dim=-1)
                acc_tc = (preds_tc == labels_for_loss_tc).float().mean()
                self.log(f'{step_prefix}_edge_acc_tc', acc_tc, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size_for_logging)
                total_acc += acc_tc
                num_acc_calcs += 1
            # Text-to-User Accuracy
            if have_loss_tu:
                preds_tu = logits_for_loss_tu.argmax(dim=-1)
                acc_tu = (preds_tu == labels_for_loss_tu).float().mean()
                self.log(f'{step_prefix}_edge_acc_tu', acc_tu, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size_for_logging)
                total_acc += acc_tu
                num_acc_calcs += 1
            # User-to-Community Accuracy
            if have_loss_uc:
                preds_uc_indices = user_comm_logits.argmax(dim=-1) 
                correct_predictions = uc_labels.gather(1, preds_uc_indices.unsqueeze(-1)).squeeze()
                acc_uc = correct_predictions.float().mean() 
                self.log(f'{step_prefix}_edge_acc_uc', acc_uc, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size_for_logging)
                total_acc += acc_uc
                num_acc_calcs += 1
            
            if num_acc_calcs > 0:
                avg_acc = total_acc / num_acc_calcs
                self.log(f'{step_prefix}_edge_acc', avg_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size_for_logging)

        # Log total edge loss
        self.log(f'{step_prefix}_edge_gen_loss', loss_edge_gen_total, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size_for_logging)
        
        return loss_edge_gen_total

    def training_step(self, batch, batch_idx):
        batch = self.re_attach_features(batch)
        has_text_nodes = 'text' in batch.x_dict and batch['text'].num_nodes > 0
        assert(has_text_nodes)

        edge_type_tc = ('text', 'post_in', 'community')
        edge_type_tu = ('text', 'post_by', 'user')
        edge_type_uc = ('user', 'active', 'community')

        assert edge_type_tc in batch.edge_index_dict and \
               batch.edge_index_dict[edge_type_tc].size(1) > 0, \
               f"Batch {batch_idx} is missing 'text_to_community' edges"

        assert edge_type_tu in batch.edge_index_dict and \
               batch.edge_index_dict[edge_type_tu].size(1) > 0, \
               f"Batch {batch_idx} is missing 'text_to_user' edges"

        assert edge_type_uc in batch.edge_index_dict and \
               batch.edge_index_dict[edge_type_uc].size(1) > 0, \
               f"Batch {batch_idx} is missing 'user_to_community' edges"

        # # --- DEBUGGING BLOCK ---
        # with torch.no_grad():
        #     uc_edge_type = ('user', 'active', 'community')

        #     if uc_edge_type in batch.edge_index_dict and batch.edge_index_dict[uc_edge_type].numel() > 0:
        #         user_input_features = batch['user'].x
        #         comm_input_features = batch['community'].x
        #         pos_edge_index = batch.edge_index_dict[uc_edge_type]
                
        #         if user_input_features.numel() > 0 and comm_input_features.numel() > 0 and \
        #            pos_edge_index[0].max() < user_input_features.size(0) and \
        #            pos_edge_index[1].max() < comm_input_features.size(0):
                   
        #             pos_user_feats = user_input_features[pos_edge_index[0]]
        #             pos_comm_feats = comm_input_features[pos_edge_index[1]]
        #             pos_sim = F.cosine_similarity(pos_user_feats, pos_comm_feats).mean()
        #             self.log('debug/pos_input_sim', pos_sim, on_step=True, on_epoch=False, prog_bar=False, batch_size=batch['text'].num_nodes)

        #             num_neg_samples = pos_edge_index.size(1)
        #             neg_user_idx = torch.randint(0, user_input_features.size(0), (num_neg_samples,), device=self.device)
        #             neg_comm_idx = torch.randint(0, comm_input_features.size(0), (num_neg_samples,), device=self.device)
        #             neg_user_feats = user_input_features[neg_user_idx]
        #             neg_comm_feats = comm_input_features[neg_comm_idx]
        #             neg_sim = F.cosine_similarity(neg_user_feats, neg_comm_feats).mean()
        #             self.log('debug/neg_input_sim', neg_sim, on_step=True, on_epoch=False, prog_bar=False, batch_size=batch['text'].num_nodes)
        # # --- END OF DEBUGGING BLOCK ---

        # Text Reconstruction Task
        original_text_features, mask = self._prepare_text_reconstruction(batch)
        out_dict = self(batch)

        loss_reconstruction = self._calculate_text_reconstruction_loss(
            out_dict, original_text_features, mask, batch_idx
        )

        self.log('train_text_recon_loss', loss_reconstruction, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch['text'].num_nodes)

        # Edge Generation Task
        loss_edge_gen_total = self._calculate_edge_generation_loss(
            batch, out_dict, step_prefix='train'
        )

        loss = self.hparams.loss_alpha * loss_reconstruction + self.hparams.beta * loss_edge_gen_total
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch['text'].num_nodes)
        
        return loss

    def validation_step(self, batch, batch_idx):
        batch = self.re_attach_features(batch)
        # Create a deep copy to avoid modifying the original validation data
        batch_copy = batch.clone()

        # Text Reconstruction Task
        original_text_features, mask = self._prepare_text_reconstruction(batch_copy)
        out_dict = self(batch_copy)

        loss_reconstruction = self._calculate_text_reconstruction_loss(
            out_dict, original_text_features, mask, batch_idx
        )

        self.log('val_text_recon_loss', loss_reconstruction, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_copy['text'].num_nodes)

        # Edge Generation Task
        loss_edge_gen_total = self._calculate_edge_generation_loss(
            batch_copy, out_dict, step_prefix='val'
        )

        loss = self.hparams.loss_alpha * loss_reconstruction + self.hparams.beta * loss_edge_gen_total
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_copy['text'].num_nodes)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        batch = self.re_attach_features(batch)
        # Create a deep copy to avoid modifying the original validation data
        batch_copy = batch.clone()

        # Text Reconstruction Task
        original_text_features, mask = self._prepare_text_reconstruction(batch_copy)
        out_dict = self(batch_copy)

        loss_reconstruction = self._calculate_text_reconstruction_loss(
            out_dict, original_text_features, mask, batch_idx
        )

        self.log('test_text_recon_loss', loss_reconstruction, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_copy['text'].num_nodes)

        # Edge Generation Task
        loss_edge_gen_total = self._calculate_edge_generation_loss(
            batch_copy, out_dict, step_prefix='test'
        )

        loss = self.hparams.loss_alpha * loss_reconstruction + self.hparams.beta * loss_edge_gen_total
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_copy['text'].num_nodes)
        
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

# --- Main Execution ---
if __name__ == '__main__':
    print("--- Initializing Model, DataLoaders, and Trainer ---")

    parser = argparse.ArgumentParser(description="GASTON Pre-training")
    parser.add_argument('--freeze-comm-emb', action='store_true',
                        help="If set, freezes the community embeddings (sets requires_grad=False).")
    args = parser.parse_args()
    
    is_comm_emb_trainable = not args.freeze_comm_emb

    print(f"is_comm_emb_trainable: {is_comm_emb_trainable}")
    
    os.makedirs(SAVE_MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.join(SAVE_MODEL_DIR, "checkpoints"), exist_ok=True)

    agg_index_map = torch.load(USER_COMMUNITY_AGG_INDEX_PATH, weights_only=False)
    user_indices = agg_index_map['user_indices']
    community_indices = agg_index_map['community_indices']
    
    print("Loading master feature tensors from the full graph...")
    full_hetero_graph = torch.load(PROCESSED_GRAPH_PATH, weights_only=False)
    text_features = full_hetero_graph['text'].x.to(device)
    initial_comm_features = full_hetero_graph['community'].x
    print("Master features loaded.")

    print(full_hetero_graph)

    # Load a blueprint batch to get metadata
    try:
        first_batch = torch.load(os.path.join(SAVE_DIR, 'train', os.listdir(os.path.join(SAVE_DIR, 'train'))[0]), weights_only=False)
    except FileNotFoundError as e:
        raise e
        
    metadata = first_batch.metadata()
    in_channels = {'text': text_features.size(1), 'user': HIDDEN_CHANNELS, 'community': HIDDEN_CHANNELS}
    print(f"Raw Input Channels: {in_channels}")
    
    model = LitPreTrainingModel(
            in_channels={}, # Derived inside the model
            metadata=metadata,
            text_features=text_features,
            initial_comm_features=initial_comm_features, 
            user_agg_map={'user_indices': user_indices, 'community_indices': community_indices},
            hidden_channels=HIDDEN_CHANNELS,
            num_hgt_layers=NUM_HGT_LAYERS,
            attention_heads=ATTENTION_HEADS,
            learning_rate=LEARNING_RATE,
            loss_alpha=LOSS_ALPHA, 
            beta=BETA,
            text_masking_rate=TEXT_MASKING_RATE,
            edge_masking_rate=EDGE_MASKING_RATE,
            comm_emb_trainable=is_comm_emb_trainable
        )
    
    train_dataset = FileDataset(os.path.join(SAVE_DIR, 'train'))
    val_dataset = FileDataset(os.path.join(SAVE_DIR, 'val'))
    test_dataset = FileDataset(os.path.join(SAVE_DIR, 'test'))
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=None, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True)

    val_loader = DataLoader(
        val_dataset, 
        batch_size=None, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True)
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=None, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    print("\n--- Starting Validation Check ---")
    print("Comparing output of old and new `re_attach_features` methods...")
    
    # Put model in eval mode (disables dropout, etc.) and move to GPU
    model.to(device)
    model.eval() 

    try:
        # Get one batch from the dataloader
        val_batch = next(iter(train_loader))
        
        # --- Run OLD Method ---
        # We must clone() the batch and move it to the device
        batch_old = val_batch.clone().to(device)
        with torch.no_grad():
            batch_old = model._re_attach_features_OLD(batch_old)
            x_user_old = batch_old['user'].x
            
        # --- Run NEW Method ---
        batch_new = val_batch.clone().to(device)
        with torch.no_grad():
            batch_new = model.re_attach_features(batch_new)
            x_user_new = batch_new['user'].x

        # --- Compare Results ---
        print(f"Old user features tensor shape: {x_user_old.shape}")
        print(f"New user features tensor shape: {x_user_new.shape}")
        
        are_close = torch.allclose(x_user_old, x_user_new, atol=1e-6)
        
        if are_close:
            print(f"✅ SUCCESS: Tensors are identical (within float tolerance).")
        else:
            print(f"❌ FAILURE: Tensors do not match.")
            diff = torch.abs(x_user_old - x_user_new).max()
            print(f"   Max absolute difference: {diff.item()}")

    except Exception as e:
        print(f"An error occurred during validation: {e}")
        
    print("--- Validation Check Complete ---\n")
    
    # Put model back in train mode before starting trainer.fit()
    model.train()
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(SAVE_MODEL_DIR, "checkpoints"),
        filename='model_XLNet-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        monitor='val_loss',
        mode='min',
    )

    wandb_logger = WandbLogger(project="GASTON_pretraining_XLNet", log_model=False, save_dir=SAVE_MODEL_DIR)

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=wandb_logger,
        log_every_n_steps=100,
        gradient_clip_val=CLIP_GRAD_NORM,
        callbacks=[checkpoint_callback],
    )

    print("\nStarting pre-training with PyTorch Lightning...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print("\n--- Starting Test Phase ---")
    trainer.test(model, dataloaders=test_loader, ckpt_path='best')
    
    print("\nPre-training complete.")
    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    print(f"Pre-trained model saved to {SAVE_MODEL_PATH}")