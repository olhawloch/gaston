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
from gensim.models import KeyedVectors
from typing import Tuple, Optional

# --- Hateful Task Configuration Defaults ---
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 30
DEFAULT_LR = 1e-4
DEFAULT_DROPOUT = 0.4 # Matches Hateful HGT dropout
DEFAULT_WEIGHT_DECAY = 1e-4 # Matches Hateful HGT weight decay
DEFAULT_HIDDEN_DIM = 256
DEFAULT_SEED = 42
DEFAULT_PATIENCE = 5 # Matches Hateful HGT patience

# --- PyTorch Lightning Model ---
class LitContextualBaseline(pl.LightningModule):
    def __init__(self, 
                 text_dim: int, 
                 context_dim: int, 
                 hidden_dim: int, 
                 task_type: str, # 'binary' or 'regression'
                 learning_rate: float, 
                 weight_decay: float, 
                 dropout_rate: float,
                 pos_weight: float = None):
        super().__init__()
        self.save_hyperparameters()
        self.task_type = task_type
        
        # We project the combined dimensions down to the hidden dim
        self.fusion_layer = nn.Linear(text_dim + context_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        # Final prediction head
        output_dim = 1
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # --- Loss & Metrics Setup ---
        if task_type == 'regression':
            self.loss_fn = nn.MSELoss()
            metrics = torchmetrics.MetricCollection([
                torchmetrics.MeanAbsoluteError(),
                torchmetrics.MeanSquaredError(squared=False), # RMSE
                torchmetrics.R2Score()
            ])
        else: # Binary Classification
            if pos_weight and pos_weight != 1.0:
                self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
                print(f"INFO: Using Weighted BCE Loss (pos_weight={pos_weight:.4f})")
            else:
                self.loss_fn = nn.BCEWithLogitsLoss()
                print("INFO: Using Standard BCE Loss")
                
            metrics = torchmetrics.MetricCollection([
                torchmetrics.Accuracy(task="binary"),
                torchmetrics.F1Score(task="binary"),
                torchmetrics.Precision(task="binary"),
                torchmetrics.AUROC(task="binary")
            ])

        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def forward(self, text_emb, context_emb):
        # 1. Concatenate Features
        combined = torch.cat([text_emb, context_emb], dim=1)
        
        # 2. Project/Fuse
        x = self.fusion_layer(combined)
        x = self.activation(x)
        x = self.dropout(x)
        
        # 3. Predict
        return self.output_layer(x).squeeze(1)

    def _common_step(self, batch, step_name):
        text_emb, context_emb, labels = batch
        preds = self(text_emb, context_emb)
        loss = self.loss_fn(preds, labels.float())
        
        metrics = getattr(self, f"{step_name}_metrics")
        if self.task_type == 'binary':
            metrics.update(torch.sigmoid(preds), labels)
        else:
            metrics.update(preds, labels.float())
            
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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1, eta_min=1e-7)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
    
    def on_fit_start(self):
        if self.trainer.train_dataloader and self.trainer.max_epochs > 0:
            loader = self.trainer.train_dataloader
            num_batches = len(list(loader)) if not hasattr(loader, '__len__') else len(loader)
            t_max = self.trainer.max_epochs * num_batches
            if self.lr_schedulers():
                self.lr_schedulers().T_max = t_max

def load_and_prep_contextual_data(
    bert_path: str, 
    edge_file_path: str, 
    vectors_path: str
) -> Tuple[TensorDataset, int, int]:
    """
    Merges BERT embeddings with Subreddit2Vec embeddings via the community edge file.
    Performs normalization (lowercasing) on community names to ensure matches.
    """
    # 1. Load BERT Embeddings
    print(f"Loading BERT embeddings from: {bert_path}")
    df_bert = pq.read_table(bert_path).to_pandas()
    df_bert = df_bert.dropna(subset=['label'])
    # Ensure ID is string to avoid int/str mismatch during merge
    df_bert['id'] = df_bert['id'].astype(str)
    
    # 2. Load Edge Data (Text ID -> Community ID)
    print(f"Loading community edge data from: {edge_file_path}")
    df_edges = pq.read_table(
        edge_file_path,
        columns=['source_id', 'target_id']
    ).to_pandas()
    
    # Ensure IDs are strings and Normalize Community Name (target_id) to lowercase
    df_edges['source_id'] = df_edges['source_id'].astype(str)
    df_edges['target_id'] = df_edges['target_id'].astype(str).str.lower()
    
    # Rename for merging consistency
    df_edges.rename(columns={'source_id': 'id', 'target_id': 'community_name'}, inplace=True)
    
    # Ensure no duplicates (1 post maps to 1 community)
    df_edges.drop_duplicates(subset=['id'], inplace=True)
    
    # 3. Merge Dataframes
    print("Merging BERT data with Community Edges...")
    df_merged = pd.merge(df_bert, df_edges, on='id', how='inner')
    print(f"Merged Data Size: {len(df_merged)} (Dropped {len(df_bert) - len(df_merged)} rows due to missing community mapping)")

    # 4. Load Subreddit2Vec Model
    print(f"Loading Subreddit2Vec vectors from: {vectors_path}")
    wv = KeyedVectors.load(vectors_path)
    vector_size = wv.vector_size
    
    # 5. Map Subreddits to Vectors
    print("Mapping subreddits to vectors...")
    def get_comm_vector(subreddit):
        # Lookup should also be robust, though we already lowercased the column
        sub_lower = str(subreddit).lower()
        if sub_lower in wv:
            return wv[sub_lower]
        else:
            # OOV Handling: Return zero vector (neutral context)
            return np.zeros(vector_size, dtype=np.float32)

    context_vectors = np.vstack(df_merged['community_name'].apply(get_comm_vector).tolist())
    text_vectors = np.vstack(df_merged['embedding'].tolist())
    
    # 6. Create Tensors
    tensor_text = torch.tensor(text_vectors, dtype=torch.float32)
    tensor_context = torch.tensor(context_vectors, dtype=torch.float32)
    
    # If using binary, we need long labels for stratification. If regression, float.
    labels_dtype = torch.float32 if df_merged['label'].dtype == float else torch.long
    tensor_labels = torch.tensor(df_merged['label'].values, dtype=labels_dtype)
    
    print(f"Final Shapes :: Text: {tensor_text.shape}, Context: {tensor_context.shape}, Labels: {tensor_labels.shape}")
    
    dataset = TensorDataset(tensor_text, tensor_context, tensor_labels)
    return dataset, tensor_text.shape[1], tensor_context.shape[1]

# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Contextual (BERT+Community) Baseline.")
    
    # Paths
    parser.add_argument("--train_path", required=True, help="Path to train folder containing text_node_embeddings_bert.parquet")
    parser.add_argument("--test_path", required=True, help="Path to test folder containing text_node_embeddings_bert.parquet")
    parser.add_argument("--edge_file_name", default="text_community_edges.parquet", help="Name of the file containing text-to-community edges inside train/test folders.")
    parser.add_argument("--vectors_path", required=True, help="Path to the .kv Subreddit2Vec vectors file.")
    parser.add_argument("--save_dir", required=True, help="Where to save results.")
    
    # Task Config
    parser.add_argument("--task_name", required=True, help="Name of the task (normvio, hateful, ruddit).")
    parser.add_argument("--task_type", required=True, choices=['binary', 'regression'], help="Type of task.")
    
    # Hyperparams (Using Hateful Defaults)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--hidden_dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # --- Load Data ---
    train_bert_file = os.path.join(args.train_path, "text_node_embeddings_bert.parquet")
    train_edge_file = os.path.join(args.train_path, args.edge_file_name)
    
    test_bert_file = os.path.join(args.test_path, "text_node_embeddings_bert.parquet")
    test_edge_file = os.path.join(args.test_path, args.edge_file_name)

    train_dataset, text_dim, context_dim = load_and_prep_contextual_data(train_bert_file, train_edge_file, args.vectors_path)
    test_dataset, _, _ = load_and_prep_contextual_data(test_bert_file, test_edge_file, args.vectors_path)

    # --- Splits ---
    train_indices = np.arange(len(train_dataset))
    train_labels = train_dataset.tensors[2].numpy()
    
    if args.task_type == 'binary':
        # Stratified Split (Required for Hateful)
        print("--- Creating Stratified Validation Split ---")
        train_idx, val_idx = sk_train_test_split(
            train_indices, test_size=0.15, stratify=train_labels, random_state=args.seed
        )
        
        train_subset_labels = train_dataset.tensors[2][train_idx]
        num_neg = (train_subset_labels == 0).sum().item()
        num_pos = (train_subset_labels == 1).sum().item()
        pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
        print(f"Binary Class Imbalance :: Neg: {num_neg}, Pos: {num_pos}, PosWeight: {pos_weight:.4f}")
        
        monitor_metric = "val_BinaryAUROC"
        monitor_mode = "max"
        pos_weight = None # Ensure unweighted loss for Hateful
    
    else:
        # Regression (Random Split)
        print("--- Creating Random Validation Split (Regression) ---")
        train_idx, val_idx = sk_train_test_split(
            train_indices, test_size=0.15, random_state=args.seed
        )
        pos_weight = None
        monitor_metric = "val_MeanAbsoluteError"
        monitor_mode = "min"

    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)

    # --- Loaders ---
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # --- Model ---
    model = LitContextualBaseline(
        text_dim=text_dim,
        context_dim=context_dim,
        hidden_dim=args.hidden_dim,
        task_type=args.task_type,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        dropout_rate=args.dropout,
        pos_weight=pos_weight 
    )

    # --- Trainer ---
    wandb_logger = WandbLogger(
        project=f"GASTON_hateful_binary_contextual_finetuning",
        name=f"CTX_BASELINE_{args.task_name}",
        save_dir=args.save_dir
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.save_dir, "checkpoints"),
        save_top_k=1,
        monitor=monitor_metric,
        mode=monitor_mode,
        filename='ctx_model-{epoch:02d}-{' + monitor_metric + ':.3f}'
    )

    early_stop_callback = EarlyStopping(
        monitor=monitor_metric,
        mode=monitor_mode,
        patience=args.patience,
        verbose=True
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu',
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback]
    )

    print(f"--- Starting Contextual Baseline Training for {args.task_name} ---")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    print("--- Evaluation on Test Set ---")
    trainer.test(model, dataloaders=test_loader, ckpt_path='best')