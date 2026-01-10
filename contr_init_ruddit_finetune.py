import torch
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, EarlyStopping
from torch_geometric.loader import NeighborLoader
import torchmetrics
from sklearn.model_selection import train_test_split as sk_train_test_split
import os
from datetime import datetime
import numpy as np
import torch_geometric.transforms as T
from collections import Counter
import argparse
import random

try:
    torch.multiprocessing.set_start_method('spawn', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')
except RuntimeError:
    pass

# --- Task Configuration ---
WANDB_PROJECT = "GASTON_contr_init_ruddit_finetuning"

# Split Configuration
VAL_SPLIT_RATIO = 0.15 # Use 15% of labeled train data for validation
RANDOM_SEED = 42

# Label Configuration
UNLABELED_SENTINEL = -999.0
TARGET_NODE_TYPE = 'text'

# Hyperparameters
HIDDEN_CHANNELS = 768
HEAD_HIDDEN_CHANNELS = 256 
NUM_HGT_LAYERS = 3
ATTENTION_HEADS = 8
HEAD_LEARNING_RATE = 3e-4
BASE_LEARNING_RATE = 3e-5 
EPOCHS = 50
BATCH_SIZE = 128
CLIP_GRAD_NORM = 0.25 
DROPOUT_RATE = 0.15
NUM_WORKERS = 8
WEIGHT_DECAY = 1e-4

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

# --- Utility Function: Train/Validation Masking (with stratification fix) ---
def create_train_val_masks(data_object, target_node_type, val_ratio, seed, sentinel):
    num_nodes = data_object[target_node_type].x.size(0)
    
    labels = data_object[target_node_type].y
    labeled_indices = torch.where(labels != sentinel)[0]
    labeled_labels = labels[labeled_indices].cpu().numpy()

    if labeled_indices.numel() == 0:
        print("Warning: No labeled nodes found for creating masks.")
        return data_object
    
    try:
        # Try to stratify (for classification)
        train_idx, val_idx = sk_train_test_split(
            labeled_indices.cpu().numpy(),
            test_size=val_ratio,
            stratify=labeled_labels,
            random_state=seed
        )
        print("--- Creating stratified train/val split. ---")
    except ValueError:
        # Fallback to random split (for regression)
        print("--- Creating random (non-stratified) train/val split (likely regression task). ---")
        train_idx, val_idx = sk_train_test_split(
            labeled_indices.cpu().numpy(),
            test_size=val_ratio,
            random_state=seed
        )

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True

    data_object[target_node_type].train_mask = train_mask
    data_object[target_node_type].val_mask = val_mask
    
    print(f"  Total Labeled Nodes: {labeled_indices.numel()}")
    print(f"  Train Labeled Nodes: {train_mask.sum().item()}")
    print(f"  Validation Labeled Nodes: {val_mask.sum().item()}")
    
    return data_object

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
                if node_type in x_dict_processed:
                    x_dict_processed[node_type] = self.dropout(x_dict_processed[node_type])

        return x_dict_processed

# --- Classification Head ---
class NodeClassificationHead(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.lin1(x).relu()
        x = self.dropout(x)
        # Squeeze the final output for regression
        return self.lin2(x).squeeze(1)

# --- Lightning Module (Adapted for Regression) ---
class LitFineTuningModel(pl.LightningModule):
    def __init__(self, in_channels, metadata, learning_rate, pretrained_model_path=None):
        super().__init__()
        self.save_hyperparameters(ignore=["metadata", "in_channels"]) # Ignore in_channels
        self.metadata = metadata
        self.hgt = HGTModel(
            in_channels=in_channels,
            hidden_channels=HIDDEN_CHANNELS, 
            num_hgt_layers=NUM_HGT_LAYERS, 
            num_heads=ATTENTION_HEADS, 
            metadata=metadata
        )

        if pretrained_model_path:
            self.load_pretrained_hgt_weights(pretrained_model_path)
            print("INFO: HGT layers are NOT frozen. Full model will fine-tune.")
        else:
            print("WARNING: No pre-trained model path provided.")

        self.classification_head = NodeClassificationHead(
            HIDDEN_CHANNELS, 
            HEAD_HIDDEN_CHANNELS,
            1, # Output dim is 1 for regression
            DROPOUT_RATE
        )
        self.target_node_type = TARGET_NODE_TYPE

        metrics = torchmetrics.MetricCollection({
            'MAE': torchmetrics.MeanAbsoluteError(),
            'RMSE': torchmetrics.MeanSquaredError(squared=False), # False = RMSE
            'r': torchmetrics.PearsonCorrCoef(),
            'r2': torchmetrics.R2Score(),
        })
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

        self.loss_fn = torch.nn.MSELoss()
        print(f"INFO: Using MSELoss for regression task.")

    def load_pretrained_hgt_weights(self, path):
        print(f"Loading pre-trained HGT model from {path}")
        state_dict = torch.load(path, map_location='cpu')
        hgt_state_dict = {k.replace('hgt.', ''): v for k, v in state_dict.items() if k.startswith('hgt.')}
        load_result = self.hgt.load_state_dict(hgt_state_dict, strict=False)
        print(f"Pre-trained HGT weights loaded successfully ({len(hgt_state_dict)} tensors). Result: {load_result}")

    def forward(self, x_dict, edge_index_dict):
        return self.hgt(x_dict, edge_index_dict)

    def _common_step(self, batch, phase):
        # 1. Get all node embeddings from HGT
        node_embeddings_dict = self.forward(batch.x_dict, batch.edge_index_dict)

        # 2. Check if target node type ('text') is present
        if self.target_node_type not in node_embeddings_dict:
            print(f"Warning: Target node type '{self.target_node_type}' not found...")
            self.log(f'{phase}_loss', 0.0, ...)
            return None

        batch_size = batch[self.target_node_type].batch_size
        if batch_size == 0:
            print(f"Warning: Batch size is 0...")
            self.log(f'{phase}_loss', 0.0, ...)
            return None

        # 3. Get seed text embeddings and labels
        seed_text_embeddings = node_embeddings_dict[self.target_node_type][:batch_size]
        seed_node_labels = batch[self.target_node_type].y[:batch_size] # Labels are float

        # 4. Pass ONLY text embeddings to the classification head
        logits = self.classification_head(seed_text_embeddings)

        # 5. Filter unlabeled, calculate loss and metrics (remains the same)
        labeled_mask = seed_node_labels != UNLABELED_SENTINEL
        if labeled_mask.sum() == 0:
            return None

        logits = logits[labeled_mask]
        seed_node_labels = seed_node_labels[labeled_mask]

        loss = self.loss_fn(logits, seed_node_labels.float()) # Loss remains MSE

        metrics = getattr(self, f"{phase}_metrics")
        if logits.numel() > 0:
            try:
                metrics.update(logits, seed_node_labels.float()) # Update regression metrics
            except Exception as e:
                print(f"Error updating regression metrics in {phase}: {e}")
                pass

        self.log(f'{phase}_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute(), on_step=False, on_epoch=True)
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), on_step=False, on_epoch=True)
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), on_step=False, on_epoch=True)
        self.test_metrics.reset()

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        self._common_step(batch, 'test')

    def on_fit_start(self):
        if self.trainer.train_dataloader is not None and self.trainer.max_epochs > 0:
            try:
                batches_per_epoch = len(self.trainer.train_dataloader)
                t_max = self.trainer.max_epochs * batches_per_epoch
                print(f"INFO: Estimated total training steps for scheduler: {t_max}")
                if self.lr_schedulers():
                    scheduler = self.lr_schedulers()[0]
                    scheduler.T_max = max(1, t_max)
                    print(f"INFO: Successfully set scheduler T_max to {scheduler.T_max} steps.")
                else:
                    print("INFO: No LR scheduler found.")
            except Exception as e:
                 print(f"WARNING: Could not determine T_max for scheduler. Error: {e}")

    def configure_optimizers(self):
        optimizer_grouped_parameters = [
            {
                "params": self.classification_head.parameters(),
                "lr": self.hparams.learning_rate # This will be HEAD_LEARNING_RATE
            },
            {
                "params": self.hgt.parameters(),
                "lr": BASE_LEARNING_RATE # The new, lower LR for the HGT body
            },
        ]

        optimizer = torch.optim.Adam(
            optimizer_grouped_parameters,
            weight_decay=WEIGHT_DECAY
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=1, # Placeholder, will be set in on_fit_start
            eta_min=BASE_LEARNING_RATE # Use base LR as the minimum
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }

# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune HGT for Ruddit regression (Contr_Init).") # <-- Desc Updated

    parser.add_argument("--train_graph_path", type=str, required=True,
                        help="Path to the processed train graph .pt file.")
    parser.add_argument("--test_graph_path", type=str, required=True,
                        help="Path to the processed test graph .pt file.")
    parser.add_argument("--pretrained_model_path", type=str, required=True,
                        help="Path to the CONTR_INIT pretrained model .pt file.")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Base directory to save model checkpoints and logs.")

    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    print(f"--- Random seed set to: {RANDOM_SEED} ---")

    print("--- Initializing Ruddit Regression Fine-Tuning Process (Contr_Init) ---") # <-- Title Updated

    if not os.path.exists(args.pretrained_model_path):
        raise FileNotFoundError(f"Pre-trained model not found at {args.pretrained_model_path}. Please provide a valid path.")

    try:
        train_data = torch.load(args.train_graph_path, weights_only=False).to('cpu')
        test_data = torch.load(args.test_graph_path, weights_only=False).to('cpu')
        print("Training and testing graphs loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: Graph data not found. Missing file: {e.filename}")
        exit()

    print(f"\n--- Creating Explicit Train/Validation Split ({100 * (1 - VAL_SPLIT_RATIO):.0f}/{100 * VAL_SPLIT_RATIO:.0f}) ---")
    # --- Make sure sentinel is float ---
    train_data = create_train_val_masks(train_data, TARGET_NODE_TYPE, VAL_SPLIT_RATIO, RANDOM_SEED, float(UNLABELED_SENTINEL))

    print("Adding reverse edges...")
    train_data = T.ToUndirected(merge=True)(train_data)
    test_data = T.ToUndirected(merge=True)(test_data)
    print("Graphs are now fully undirected.")

    train_data = train_data.to(device)
    test_data = test_data.to(device)

    print("Building in_channels dictionary...")
    in_channels = {
        'text': train_data['text'].x.size(1),
        'user': train_data['user'].x.size(1),
        'community': train_data['community'].x.size(1)
    }
    print(f"  in_channels: {in_channels}")

    model = LitFineTuningModel(
        in_channels=in_channels,
        metadata=train_data.metadata(),
        learning_rate=HEAD_LEARNING_RATE,
        pretrained_model_path=args.pretrained_model_path
    ).to(device)

    num_neighbors_dict = {
        ('text', 'post_in', 'community'): [25, 15, 10],
        ('community', 'rev_post_in', 'text'): [25, 15, 10],
        ('text', 'post_by', 'user'): [25, 15, 10],
        ('user', 'rev_post_by', 'text'): [25, 15, 10],
        ('user', 'active', 'community'): [25, 15, 10],
        ('community', 'rev_active', 'user'): [25, 15, 10]
    }

    def get_filtered_neighbors(data):
        return {k: v for k, v in num_neighbors_dict.items() if k in data.edge_types}

    train_loader = NeighborLoader(
        train_data,
        num_neighbors=get_filtered_neighbors(train_data),
        input_nodes=(TARGET_NODE_TYPE, train_data[TARGET_NODE_TYPE].train_mask),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        persistent_workers=True if NUM_WORKERS > 0 else False,
    )

    val_loader = NeighborLoader(
        train_data,
        num_neighbors=get_filtered_neighbors(train_data),
        input_nodes=(TARGET_NODE_TYPE, train_data[TARGET_NODE_TYPE].val_mask),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=True if NUM_WORKERS > 0 else False,
    )

    test_loader = NeighborLoader(
        test_data,
        num_neighbors=get_filtered_neighbors(test_data),
        # --- Use float sentinel for test loader input filter ---
        input_nodes=(TARGET_NODE_TYPE, test_data[TARGET_NODE_TYPE].y != float(UNLABELED_SENTINEL)),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=True if NUM_WORKERS > 0 else False,
    )

    RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    SAVE_MODEL_DIR = os.path.join(args.save_dir, f"contr_init_{RUN_TIMESTAMP}")
    os.makedirs(os.path.join(SAVE_MODEL_DIR, "checkpoints"), exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(SAVE_MODEL_DIR, "checkpoints"),
        filename='model-{epoch:02d}-{val_MeanAbsoluteError:.3f}', # Use metric name
        save_top_k=1,
        monitor='val_MAE', # Use metric name
        mode='min', # Minimize error
    )

    wandb_logger = WandbLogger(
        project=WANDB_PROJECT,
        log_model=False,
        save_dir=SAVE_MODEL_DIR,
    )

    early_stop_callback = EarlyStopping(
        monitor='val_MAE', # Monitor the validation MAE
        patience=10, # Stop if no improvement for 10 epochs
        mode='min',  # We want to minimize MAE
        verbose=True
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        logger=wandb_logger,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback, early_stop_callback],
        gradient_clip_val=CLIP_GRAD_NORM,
    )

    print("\nStarting fine-tuning with PyTorch Lightning (Contr_Init)...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print("\nFine-tuning complete. Evaluating best model on the test set...")
    trainer.test(model, dataloaders=test_loader, ckpt_path='best')

    print("--- Test evaluation complete. ---")