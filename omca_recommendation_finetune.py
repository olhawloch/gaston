import torch
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader as NodeLoader
import torch_geometric.transforms as T
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import os
import argparse
from datetime import datetime
import gc
from tqdm import tqdm
import random
import numpy as np

LEARNING_RATE = 1e-5
EPOCHS = 30
BATCH_SIZE = 512
HIDDEN_CHANNELS = 768
NUM_HGT_LAYERS = 3
ATTENTION_HEADS = 8
NUM_WORKERS = 8
DROPOUT_RATE = 0.2

RANDOM_SEED = 42

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- HGT Model Definition ---
class HGTModel(torch.nn.Module):
    def __init__(self, hidden_channels, num_hgt_layers, num_heads, metadata):
        super().__init__()
        
        self.node_types = metadata[0]
            
        self.convs = torch.nn.ModuleList()
        for _ in range(num_hgt_layers):
            # The HGTConv layer will operate directly on the input features
            conv = HGTConv(hidden_channels, hidden_channels, metadata, num_heads)
            self.convs.append(conv)

        self.out_channels = hidden_channels
        self.dropout = torch.nn.Dropout(p=DROPOUT_RATE)

    def forward(self, x_dict, edge_index_dict):
        x_dict_processed = x_dict 

        # Message Passing
        for conv in self.convs:
            x_dict_processed = conv(x_dict_processed, edge_index_dict)
            for node_type in self.node_types:
                x_dict_processed[node_type] = self.dropout(x_dict_processed[node_type])
                
        return x_dict_processed

# --- PyTorch Lightning Module for Fine-Tuning ---
class LitFinetuningModel(pl.LightningModule):
    def __init__(self, hidden_channels, num_hgt_layers, num_heads, metadata, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        
        self.hgt = HGTModel(
            hidden_channels=hidden_channels,
            num_hgt_layers=num_hgt_layers,
            num_heads=num_heads,
            metadata=metadata
        )

        self.user_proj = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )
        
        self.comm_proj = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )

        self.fwd_edge_type = ('user', 'active_in', 'community')

        metrics = torchmetrics.MetricCollection([
            torchmetrics.Accuracy(task="binary"),
            torchmetrics.AUROC(task="binary")
        ])
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')
    
    def load_pretrained_hgt_weights(self, path):
        print(f"Loading pre-trained HGT model from {path}")
        state_dict = torch.load(path, map_location='cpu')

        # Extract only the weights belonging to the HGT model
        hgt_state_dict = {k.replace('hgt.', ''): v for k, v in state_dict.items() if k.startswith('hgt.')}

        if not hgt_state_dict:
            print("WARNING: No weights with 'hgt.' prefix found in checkpoint. Model is NOT loading pretrained weights.")
            return

        load_result = self.hgt.load_state_dict(hgt_state_dict, strict=False)
        print(f"  Pre-trained HGT weights loaded successfully ({len(hgt_state_dict)} tensors). Result: {load_result}")

    def forward(self, x_dict, edge_index_dict):
        return self.hgt(x_dict, edge_index_dict)

    def _shared_step(self, batch, step_prefix):
        supervision_store = batch[self.fwd_edge_type]
        edge_label_index = supervision_store.edge_label_index
        edge_label = supervision_store.edge_label

        out_dict = self(batch.x_dict, batch.edge_index_dict)
        
        raw_user_emb = out_dict['user'][edge_label_index[0]]
        raw_comm_emb = out_dict['community'][edge_label_index[1]]
        
        user_emb = self.user_proj(raw_user_emb)
        comm_emb = self.comm_proj(raw_comm_emb)

        logits = (user_emb * comm_emb).sum(dim=-1)
        
        loss = F.binary_cross_entropy_with_logits(logits, edge_label.float())
        probs = torch.sigmoid(logits)

        metrics = getattr(self, f"{step_prefix}_metrics")
        metrics.update(probs, edge_label.int())
        self.log(f'{step_prefix}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=edge_label.size(0))

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, 'test')

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute(), on_step=False, on_epoch=True)
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), on_step=False, on_epoch=True)
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), on_step=False, on_epoch=True)
        self.test_metrics.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

# --- Main Execution ---
if __name__ == '__main__':
    pl.seed_everything(42)
    
    parser = argparse.ArgumentParser(description="Fine-tune HGT for Recommendation")
    parser.add_argument('--full_graph_path', type=str, required=True, help="Path to the .pt file")
    parser.add_argument('--pretrained_model_path', type=str, required=True, help="Path to the .pt file")
    parser.add_argument('--save_dir', type=str, required=True, help="Base directory to save models and results")
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    print(f"--- Random seed set to: {RANDOM_SEED} ---")

    RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    SAVE_MODEL_DIR = os.path.join(args.save_dir, f"recommendation_finetune_{RUN_TIMESTAMP}")
    os.makedirs(SAVE_MODEL_DIR, exist_ok=True)
    
    OFFLINE_EMBEDDING_PATH = os.path.join(SAVE_MODEL_DIR, "final_embeddings.pt")
    TRAIN_DATA_PATH = os.path.join(SAVE_MODEL_DIR, "train_data_for_eval.pt")
    TEST_DATA_PATH = os.path.join(SAVE_MODEL_DIR, "test_data_for_eval.pt")
    
    print(f"Run output directory: {SAVE_MODEL_DIR}")

    print(f"Loading graph from: {args.full_graph_path}")
    if not os.path.exists(args.full_graph_path):
        raise FileNotFoundError(f"Graph file not found at: {args.full_graph_path}")
    full_graph = torch.load(args.full_graph_path, weights_only=False)
    print("Graph loaded successfully.")

    print("Making graph undirected...")
    transform_undirected = T.ToUndirected(merge=False)
    full_graph = transform_undirected(full_graph)
    print(full_graph)
    
    fwd_edge_type = ('user', 'active_in', 'community')
    rev_edge_type = ('community', 'rev_active_in', 'user')

    print("Splitting links...")
    transform_split = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=False, 
        add_negative_train_samples=False, # We will sample negatives on-the-fly
        neg_sampling_ratio=0.0,
        edge_types=fwd_edge_type,
        rev_edge_types=rev_edge_type
    )
    train_data, val_data, test_data = transform_split(full_graph)

    # print("Patching train_data for message passing...")
    
    # store_fwd = train_data[fwd_edge_type]
    # store_fwd.edge_index = store_fwd.edge_label_index
    
    # store_rev = train_data[rev_edge_type]
    # store_rev.edge_index = store_rev.edge_label_index
    
    # print("Train data patched.")
    print("\n--- Final Train Data (for loaders) ---")
    print(train_data)

    print("Setting up DataLoaders...")
    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[10, 5], # Sample 10 neighbors at hop 1, 5 at hop 2
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        edge_label_index=(fwd_edge_type, train_data[fwd_edge_type].edge_label_index),
        edge_label=train_data[fwd_edge_type].edge_label,
        neg_sampling_ratio=1.0 
    )

    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=[10, 5],
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        edge_label_index=(fwd_edge_type, val_data[fwd_edge_type].edge_label_index),
        edge_label=val_data[fwd_edge_type].edge_label,
        neg_sampling_ratio=1.0
    )

    test_loader = LinkNeighborLoader(
        data=test_data,
        num_neighbors=[10, 5],
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        edge_label_index=(fwd_edge_type, test_data[fwd_edge_type].edge_label_index),
        edge_label=test_data[fwd_edge_type].edge_label,
        neg_sampling_ratio=1.0
    )

    print("Initializing model...")
    # in_channels = {
    #     'text': full_graph['text'].x.size(1),
    #     'user': full_graph['user'].x.size(1),
    #     'community': full_graph['community'].x.size(1)
    # }

    model = LitFinetuningModel(
        hidden_channels=HIDDEN_CHANNELS,
        num_hgt_layers=NUM_HGT_LAYERS,
        num_heads=ATTENTION_HEADS,
        metadata=full_graph.metadata(),
        learning_rate=LEARNING_RATE
    )

    model.load_pretrained_hgt_weights(args.pretrained_model_path)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(SAVE_MODEL_DIR, "checkpoints"),
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        monitor='val_loss',
        mode='min',
    )

    wandb_logger = WandbLogger(
        project="recommendation_finetune_ave_init", 
        log_model=False, 
        save_dir=SAVE_MODEL_DIR
    )

    early_stop_callback = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=5,
    verbose=True
)

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=wandb_logger,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback, early_stop_callback],
        limit_train_batches=2000,
        limit_val_batches=300
    )

    print("\nStarting fine-tuning...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print("Cleaning up training resources...")
    del train_loader
    del val_loader
    gc.collect()
    torch.cuda.empty_cache()

    test_loader = LinkNeighborLoader(
        data=test_data,
        num_neighbors=[10, 5],
        batch_size=BATCH_SIZE,
        num_workers=7,
        persistent_workers=False,
        edge_label_index=(fwd_edge_type, test_data[fwd_edge_type].edge_label_index),
        edge_label=test_data[fwd_edge_type].edge_label,
        neg_sampling_ratio=1.0
    )

    print("\nRunning sampled test set...")
    with torch.inference_mode():
        trainer.test(model, dataloaders=test_loader, ckpt_path='best')
    
    del test_loader
    gc.collect()

    print("\nStarting offline full-graph evaluation on CPU...")
    
    best_model_path = checkpoint_callback.best_model_path
    if not best_model_path:
        print("No best model found, using last trained model.")
    else:
        print(f"Loading best model from: {best_model_path}")
        model = LitFinetuningModel.load_from_checkpoint(best_model_path)

    model.to('cpu')
    model.eval()
    
    # --- ROBUST BATCHED INFERENCE ---
    print("Generating final embeddings using Batched NeighborLoader...")
    
    final_embeddings = {}
    
    for target_type in ['user', 'community']:
        print(f"  Processing {target_type} nodes...")
        
        # num_neighbors=[-1] gets ALL neighbors (needed for accurate full-graph evaluation)
        node_loader = NodeLoader(
            train_data, 
            num_neighbors=[-1], 
            batch_size=1024, 
            input_nodes=target_type,
            shuffle=False,
            num_workers=0 # Use main process to save RAM overhead
        )
        
        type_embeddings = []
        
        with torch.no_grad():
            for batch in tqdm(node_loader, desc=f"    Generating {target_type}"):
                out_dict = model(batch.x_dict, batch.edge_index_dict)
                
                batch_size = batch[target_type].batch_size
                raw_emb = out_dict[target_type][:batch_size]
                
                if target_type == 'user':
                    proj_emb = model.user_proj(raw_emb)
                elif target_type == 'community':
                    proj_emb = model.comm_proj(raw_emb)
                else:
                    proj_emb = raw_emb

                type_embeddings.append(proj_emb.detach().cpu())
        
        # Concatenate all batches to get the full embedding matrix
        final_embeddings[target_type] = torch.cat(type_embeddings, dim=0)
        print(f"  Finished {target_type}. Shape: {final_embeddings[target_type].shape}")

    print(f"Saving final embeddings to: {OFFLINE_EMBEDDING_PATH}")
    torch.save(final_embeddings, OFFLINE_EMBEDDING_PATH)
    
    print(f"Saving ground truth train data to: {TRAIN_DATA_PATH}")
    torch.save(train_data.to('cpu'), TRAIN_DATA_PATH)
    
    print(f"Saving ground truth test data to: {TEST_DATA_PATH}")
    torch.save(test_data.to('cpu'), TEST_DATA_PATH)
    
    print("\nOffline evaluation data saved.")
    print("Fine-tuning complete.")