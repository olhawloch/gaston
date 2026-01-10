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

VAL_SPLIT_RATIO = 0.15
RANDOM_SEED = 42

# Hyperparameters (Matched to HGT Ruddit Script)
BATCH_SIZE = 128
LEARNING_RATE = 1e-4 # Standard for MLP heads
EPOCHS = 50
HIDDEN_DIM = 256
WEIGHT_DECAY = 5e-4  # Matches HGT
DROPOUT_RATE = 0.4   # Matches HGT
NUM_WORKERS = 8

# --- PyTorch Lightning Model (The Regressor) ---
class LitBaselineRegressor(pl.LightningModule):
    def __init__(self, input_dim: int, hidden_dim: int, 
                 learning_rate: float, weight_decay: float, dropout_rate: float):
        super().__init__()
        self.save_hyperparameters()
        
        # Head outputs 1 value for regression
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1) 
        )

        # Use MSELoss for regression
        self.loss_fn = nn.MSELoss()
        print("INFO: Using MSELoss for regression task.")

        # Metrics (Matches HGT)
        metrics = torchmetrics.MetricCollection({
            'MAE': torchmetrics.MeanAbsoluteError(),
            'RMSE': torchmetrics.MeanSquaredError(squared=False), # False = RMSE
            'r': torchmetrics.PearsonCorrCoef(),
            'r2': torchmetrics.R2Score(),
        })
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def forward(self, x):
        return self.head(x).squeeze(1) 

    def _common_step(self, batch, step_name):
        features, labels = batch
        preds = self(features)
        
        # Calculate loss on float labels
        loss = self.loss_fn(preds, labels.float())
        
        # Get metrics
        metrics_to_update = getattr(self, f"{step_name}_metrics")
        metrics_to_update.update(preds, labels.float())
        
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
        # Matches HGT Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=1, 
            eta_min=1e-6
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
    # Labels are floats for regression
    labels = torch.tensor(df['label'].values, dtype=torch.float32)
    
    input_dim = features.shape[1]
    dataset = TensorDataset(features, labels)
    return dataset, input_dim

# --- Main execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a baseline text regressor from BERT embeddings for Ruddit.")
    parser.add_argument("--train_data_path", required=True, help="Path to the training data directory.")
    parser.add_argument("--test_data_path", required=True, help="Path to the testing data directory.")
    parser.add_argument("--task_name", required=True, help="Task name for logging.")
    parser.add_argument("--save_dir", required=True, help="Directory to save model checkpoints and logs.")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"--- Output directory set to: {args.save_dir} ---")

    # --- 1. Load Train Data ---
    full_train_dataset, input_dim = load_and_prep_data(args.train_data_path)
    if full_train_dataset is None:
        exit(1)
        
    print(f"\nTraining data loaded: {len(full_train_dataset)} samples.")
    print(f"Input dimension: {input_dim}")

    # --- 2. Split Data (Matches HGT: Random Split for Regression) ---
    # We use sklearn split to be consistent with HGT logic
    train_indices_all = np.arange(len(full_train_dataset))
    
    print("--- Creating random train/val split (Matching HGT Regression logic) ---")
    train_idx, val_idx = sk_train_test_split(
        train_indices_all,
        test_size=VAL_SPLIT_RATIO,
        random_state=RANDOM_SEED
    )
    
    train_dataset = Subset(full_train_dataset, train_idx)
    val_dataset = Subset(full_train_dataset, val_idx)
    print(f"Splits created: {len(train_dataset)} train, {len(val_dataset)} val")

    # --- 3. Load Test Data ---
    test_dataset, test_input_dim = load_and_prep_data(args.test_data_path)
    if test_dataset is None:
        exit(1)
        
    if input_dim != test_input_dim:
        print("WARNING: Mismatch in data properties between train and test!")

    # --- 4. Create DataLoaders ---
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # --- 5. Initialize Model ---
    model = LitBaselineRegressor(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        dropout_rate=DROPOUT_RATE
    )

    # --- 6. Initialize Trainer ---
    wandb_logger = WandbLogger(
        project="GASTON_ruddit_regression_finetuning", # Matches HGT Project
        name=f"BASELINE_BERT_{args.task_name}",
        save_dir=args.save_dir
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.save_dir, "checkpoints"),
        save_top_k=1,
        monitor='val_MAE',
        mode='min',
        filename='baseline-{epoch:02d}-{val_MAE:.3f}'
    )

    early_stop_callback = EarlyStopping(
        monitor='val_MAE',
        patience=10, 
        mode='min', 
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