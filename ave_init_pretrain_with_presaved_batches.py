import torch
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
import torch_geometric.transforms as T
from tqdm import tqdm
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from datetime import datetime

try:
    torch.multiprocessing.set_start_method('spawn', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')
except RuntimeError:
    pass

torch.set_float32_matmul_precision('high')

# --- Configuration ---
DATA_PATH = "../GASTON/pretraining/graph_data/"
SAVE_DIR = os.path.join(DATA_PATH, "ave_init_pregenerated_batches")
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
SAVE_MODEL_DIR = os.path.join(DATA_PATH, "model", RUN_TIMESTAMP)
SAVE_MODEL_PATH = os.path.join(SAVE_MODEL_DIR, "ave_init_pretrained.pt")
PROCESSED_GRAPH_PATH = os.path.join(DATA_PATH, "processed/ave_init_pretrain_heterogeneous_graph.pt")

# Hyperparameters
HIDDEN_CHANNELS = 768
NUM_HGT_LAYERS = 3
ATTENTION_HEADS = 8
LEARNING_RATE = 1e-4
EPOCHS = 6
BATCH_SIZE = 128
TEXT_MASKING_RATE = 0.15
LOSS_ALPHA = 0.7 # Weighting factor for text reconstruction loss
BETA = 1-LOSS_ALPHA # Weighting factor for edge generation loss
GAMMA = 0.5 # For edge generation, a ranking loss
CLIP_GRAD_NORM = 0.25 
EDGE_MASKING_RATE = 0.50
NUM_WORKERS = 8
DROPOUT_RATE = 0.3

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
    def __init__(self, hidden_channels, out_channels, num_hgt_layers, num_heads, metadata):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_hgt_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata, num_heads)
            self.convs.append(conv)
        self.node_types = metadata[0]
        self.out_channels = hidden_channels
        self.dropout = torch.nn.Dropout(p=DROPOUT_RATE)

    def forward(self, x_dict, edge_index_dict):
        x_dict_processed = x_dict
        for conv in self.convs:
            x_dict_processed = conv(x_dict_processed, edge_index_dict)
            for node_type in self.node_types:
                x_dict_processed[node_type] = self.dropout(x_dict_processed[node_type])
        return x_dict_processed

# --- PyTorch Lightning Module ---
class LitPreTrainingModel(pl.LightningModule):
    def __init__(self, metadata, text_features, user_features, community_features, 
                 hidden_channels, num_hgt_layers, attention_heads, learning_rate, 
                 loss_alpha, beta, text_masking_rate, edge_masking_rate):
        super().__init__()
        self.save_hyperparameters(ignore=["metadata", "text_features", "user_features", "community_features"])
        
        self.text_features = text_features
        self.user_features = user_features
        self.community_features = community_features
        self.metadata = metadata
        
        self.hgt = HGTModel(self.hparams.hidden_channels, self.hparams.hidden_channels, self.hparams.num_hgt_layers, self.hparams.attention_heads, metadata)
        
        text_feature_size = self.text_features.size(1)
        self.text_decoder = torch.nn.Linear(self.hparams.hidden_channels, text_feature_size)
        self.mask_embedding = torch.nn.Parameter(torch.randn(text_feature_size))

    def re_attach_features(self, batch):
        """Helper function to add features back to a blueprint batch."""
        batch['text'].x = self.text_features[batch['text'].n_id]
        batch['user'].x = self.user_features[batch['user'].n_id]
        batch['community'].x = self.community_features[batch['community'].n_id]
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
        if batch_idx < 3:
            print("\n" + "="*50)
            print(f"--- DEBUGGING BATCH {batch_idx} (Epoch {self.current_epoch}) ---")
            print("--- Text Reconstruction Tensors ---")
            
            print(f"\n1. Original Features (Ground Truth):")
            print(f"   Shape: {original_text_features.shape}")
            if original_text_features.numel() > 0:
                print(f"   Sample (first 5 values of 1st vector): {original_text_features[0, :5]}")

            gnn_output_for_masked = out_dict['text'][mask]
            print(f"\n2. GNN Output (Input to Decoder):")
            print(f"   Shape: {gnn_output_for_masked.shape}")
            if gnn_output_for_masked.numel() > 0:
                print(f"   Sample (first 5 values of 1st vector): {gnn_output_for_masked[0, :5]}")
                
            print(f"\n3. Reconstructed Features (Model's Guess):")
            print(f"   Shape: {reconstructed_text_features.shape}")
            if reconstructed_text_features.numel() > 0:
                print(f"   Sample (first 5 values of 1st vector): {reconstructed_text_features[0, :5]}")
            print("="*50 + "\n")
            
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
        edge_type_uc = ('user', 'active_in', 'community')

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
        edge_type_uc = ('user', 'active_in', 'community')

        assert edge_type_tc in batch.edge_index_dict and \
               batch.edge_index_dict[edge_type_tc].size(1) > 0, \
               f"Batch {batch_idx} is missing 'text_to_community' edges"

        assert edge_type_tu in batch.edge_index_dict and \
               batch.edge_index_dict[edge_type_tu].size(1) > 0, \
               f"Batch {batch_idx} is missing 'text_to_user' edges"

        assert edge_type_uc in batch.edge_index_dict and \
               batch.edge_index_dict[edge_type_uc].size(1) > 0, \
               f"Batch {batch_idx} is missing 'user_to_community' edges"

        # --- DEBUGGING BLOCK ---
        with torch.no_grad():
            uc_edge_type = ('user', 'active_in', 'community')

            if uc_edge_type in batch.edge_index_dict and batch.edge_index_dict[uc_edge_type].numel() > 0:
                user_input_features = batch['user'].x
                comm_input_features = batch['community'].x
                pos_edge_index = batch.edge_index_dict[uc_edge_type]
                
                if user_input_features.numel() > 0 and comm_input_features.numel() > 0 and \
                   pos_edge_index[0].max() < user_input_features.size(0) and \
                   pos_edge_index[1].max() < comm_input_features.size(0):
                   
                    pos_user_feats = user_input_features[pos_edge_index[0]]
                    pos_comm_feats = comm_input_features[pos_edge_index[1]]
                    pos_sim = F.cosine_similarity(pos_user_feats, pos_comm_feats).mean()
                    self.log('debug/pos_input_sim', pos_sim, on_step=True, on_epoch=False, prog_bar=False, batch_size=batch['text'].num_nodes)

                    num_neg_samples = pos_edge_index.size(1)
                    neg_user_idx = torch.randint(0, user_input_features.size(0), (num_neg_samples,), device=self.device)
                    neg_comm_idx = torch.randint(0, comm_input_features.size(0), (num_neg_samples,), device=self.device)
                    neg_user_feats = user_input_features[neg_user_idx]
                    neg_comm_feats = comm_input_features[neg_comm_idx]
                    neg_sim = F.cosine_similarity(neg_user_feats, neg_comm_feats).mean()
                    self.log('debug/neg_input_sim', neg_sim, on_step=True, on_epoch=False, prog_bar=False, batch_size=batch['text'].num_nodes)
        # --- END OF DEBUGGING BLOCK ---

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
    
    os.makedirs(SAVE_MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.join(SAVE_MODEL_DIR, "checkpoints"), exist_ok=True)

    print("Loading master feature tensors from the full graph...")
    full_graph = torch.load(PROCESSED_GRAPH_PATH, weights_only=False)
    text_features = full_graph['text'].x.to(device)
    user_features = full_graph['user'].x.to(device)
    community_features = full_graph['community'].x.to(device)
    print("Master features loaded to GPU.")

    print(full_graph)
    
    # Load a single blueprint batch to get the graph metadata
    try:
        first_batch_path = os.path.join(SAVE_DIR, 'train', os.listdir(os.path.join(SAVE_DIR, 'train'))[0])
        first_batch = torch.load(first_batch_path, weights_only=False)
    except FileNotFoundError as e:
        print("Error: Could not find the first training batch. Please run data preparation first.")
        raise e
        
    metadata = first_batch.metadata()
    
    model = LitPreTrainingModel(
        metadata=metadata,
        text_features=text_features,
        user_features=user_features,
        community_features=community_features,
        hidden_channels=HIDDEN_CHANNELS,
        num_hgt_layers=NUM_HGT_LAYERS,
        attention_heads=ATTENTION_HEADS,
        learning_rate=LEARNING_RATE,
        loss_alpha=LOSS_ALPHA, 
        beta=BETA, 
        text_masking_rate=TEXT_MASKING_RATE,
        edge_masking_rate=EDGE_MASKING_RATE
    )
    
    # 3. Now initialize the datasets and dataloaders
    train_dataset = FileDataset(os.path.join(SAVE_DIR, 'train'))
    val_dataset = FileDataset(os.path.join(SAVE_DIR, 'val'))
    test_dataset = FileDataset(os.path.join(SAVE_DIR, 'test'))
    
    # Use standard DataLoaders with the FileDatasets
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(SAVE_MODEL_DIR, "checkpoints"),
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        monitor='val_loss',
        mode='min',
    )

    wandb_logger = WandbLogger(
        project="nov_ave_init_GASTON_pretraining",
        log_model=False,
        save_dir=SAVE_MODEL_DIR,
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        logger=wandb_logger,
        log_every_n_steps=100,
        enable_progress_bar=True,
        enable_checkpointing=True,
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