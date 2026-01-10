import pandas as pd
import pyarrow.parquet as pq
import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from torch.utils.data import TensorDataset, DataLoader, Subset
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split as sk_train_test_split
from typing import Tuple

# --- Configuration ---
INPUT_EMBEDDING_FILE = "text_node_embeddings_bert.parquet"

# Matches HGT Script
VAL_SPLIT_RATIO = 0.15
RANDOM_SEED = 42
POS_WEIGHT_VALUE = 0.2911  # From HGT Script

BATCH_SIZE = 128           # Matches HGT
LEARNING_RATE = 1e-4       
EPOCHS = 30                # Matches HGT
HIDDEN_DIM = 256
WEIGHT_DECAY = 1e-4        # Matches HGT
NUM_WORKERS = 8

# --- PyTorch Lightning Model ---
class LitBaselineClassifier(pl.LightningModule):
    def __init__(self, input_dim: int, hidden_dim: int, learning_rate: float, weight_decay: float, pos_weight: float):
        super().__init__()
        self.save_hyperparameters()
        
        # Architecture: Simple MLP Head
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4), # Matches generic dropout often used in baselines
            nn.Linear(hidden_dim, 1) 
        )

        # Weighted BCE Loss
        pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        print(f"INFO: Using weighted BCE loss with pos_weight={pos_weight:.4f}")

        # Metrics (Binary)
        metrics = torchmetrics.MetricCollection([
            torchmetrics.Accuracy(task="binary"),
            torchmetrics.F1Score(task="binary"),
            torchmetrics.Precision(task="binary"),
            torchmetrics.AUROC(task="binary")
        ])
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def forward(self, x):
        return self.head(x).squeeze(1)

    def _common_step(self, batch, step_name):
        features, labels = batch
        logits = self(features)
        loss = self.loss_fn(logits, labels.float())
        
        probs = torch.sigmoid(logits)
        metrics_to_update = getattr(self, f"{step_name}_metrics")
        metrics_to_update.update(probs, labels)
        
        self.log(f'{step_name}_loss', loss, prog_bar=(step_name=='val'), on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        self._common_step(batch, 'test')

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute(), on_step=False, on_epoch=True)
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), on_step=False, on_epoch=True)
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), on_step=False, on_epoch=True)
        self.test_metrics.reset()

    def on_fit_start(self):
        if self.trainer.train_dataloader is not None and self.trainer.max_epochs > 0:
            loader = self.trainer.train_dataloader
            if not hasattr(loader, '__len__'):
                 num_batches = len(list(loader)) 
            else:
                 num_batches = len(loader)
            t_max = self.trainer.max_epochs * num_batches
            if self.lr_schedulers():
                scheduler = self.lr_schedulers()
                if isinstance(scheduler, list): scheduler = scheduler[0]
                scheduler.T_max = t_max
                print(f"INFO: Successfully set scheduler T_max to {t_max} steps.")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=1, 
            eta_min=1e-7 # Matches HGT script
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }

def load_and_prep_data(data_path: str) -> Tuple[TensorDataset, int]:
    input_path = os.path.join(data_path, INPUT_EMBEDDING_FILE)
    print(f"Loading data from {input_path}...")
    
    try:
        df = pq.read_table(input_path).to_pandas()
    except Exception as e:
        print(f"ERROR: Could not load file {input_path}. Details: {e}")
        return None, 0

    original_count = len(df)
    df = df.dropna(subset=['label'])
    print(f"Found {len(df)} labeled records (dropped {original_count - len(df)} unlabeled).")

    if len(df) == 0:
        return None, 0

    features = torch.tensor(np.vstack(df['embedding'].tolist()), dtype=torch.float32)
    labels = torch.tensor(df['label'].values, dtype=torch.long)
    
    input_dim = features.shape[1]
    
    dataset = TensorDataset(features, labels)
    return dataset, input_dim

# --- Main execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a baseline text classifier from BERT embeddings for Dreaddit.")
    parser.add_argument("--train_data_path", required=True, help="Path to the training data directory.")
    parser.add_argument("--test_data_path", required=True, help="Path to the testing data directory.")
    parser.add_argument("--task_name", required=True, help="Task name for logging.")
    parser.add_argument("--save_dir", required=True, help="Directory to save model checkpoints and logs.")
    args = parser.parse_args()

    # Ensure save directory exists
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"--- Output directory set to: {args.save_dir} ---")

    # --- 1. Load Train Data ---
    full_train_dataset, input_dim = load_and_prep_data(args.train_data_path)
    if full_train_dataset is None:
        exit(1)
        
    print(f"\nTraining data loaded: {len(full_train_dataset)} samples.")
    print(f"Input dimension: {input_dim}")

    # --- 2. Stratified Split ---
    # Note: HGT uses a user-centric split. Here we approximate with Stratified Split
    # to ensure label distribution consistency.
    train_indices = np.arange(len(full_train_dataset))
    train_labels_all = full_train_dataset.tensors[1].numpy()
    
    try:
        print("--- Creating stratified train/val split ---")
        train_idx, val_idx = sk_train_test_split(
            train_indices,
            test_size=VAL_SPLIT_RATIO,
            stratify=train_labels_all,
            random_state=RANDOM_SEED
        )
    except ValueError as e:
        print(f"Stratification failed ({e}). Falling back to random split.")
        train_idx, val_idx = sk_train_test_split(
            train_indices,
            test_size=VAL_SPLIT_RATIO,
            random_state=RANDOM_SEED
        )
        
    train_dataset = Subset(full_train_dataset, train_idx)
    val_dataset = Subset(full_train_dataset, val_idx)
    print(f"Splits created: {len(train_dataset)} train, {len(val_dataset)} val")

    # --- 3. Load Test Data ---
    test_dataset, _ = load_and_prep_data(args.test_data_path)
    if test_dataset is None:
        exit(1)

    # --- 4. Create DataLoaders ---
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # --- 5. Initialize Model ---
    model = LitBaselineClassifier(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        pos_weight=POS_WEIGHT_VALUE # Hardcoded from config
    )

    # --- 6. Initialize Trainer ---
    wandb_logger = WandbLogger(
        project="GASTON_ave_init_dreaddit_binary_finetuning", # Matches HGT Project
        name=f"BASELINE_BERT_{args.task_name}",
        save_dir=args.save_dir
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.save_dir, "checkpoints"),
        save_top_k=1,
        monitor='val_BinaryAUROC', 
        mode='max',
        filename='baseline-{epoch:02d}-{val_BinaryAUROC:.3f}'
    )

    early_stop_callback = EarlyStopping(
        monitor='val_BinaryAUROC',
        mode='max',
        patience=5,
        verbose=True
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator='gpu',
        devices=1,
        logger=wandb_logger,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback, early_stop_callback]
    )

    print(f"--- Starting BERT baseline training for {args.task_name} ---")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    print("--- Training complete. Running test set... ---")
    trainer.test(model, dataloaders=test_loader, ckpt_path='best')
    print("--- Baseline training finished. ---")