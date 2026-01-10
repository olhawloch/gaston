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

# --- Dreaddit Task Configuration (Hardcoded) ---
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.1        
WEIGHT_DECAY = 1e-4    
HIDDEN_DIM = 256
RANDOM_SEED = 42
PATIENCE = 5
POS_WEIGHT = 0.2911      

# --- PyTorch Lightning Model ---
class LitContextualBaseline(pl.LightningModule):
    def __init__(self, 
                 text_dim: int, 
                 context_dim: int, 
                 hidden_dim: int, 
                 task_type: str, 
                 learning_rate: float, 
                 weight_decay: float, 
                 dropout_rate: float,
                 pos_weight: float = None):
        super().__init__()
        self.save_hyperparameters()
        self.task_type = task_type
        
        # Project concatenated features (Text + Context) -> Hidden
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
                torchmetrics.MeanSquaredError(squared=False),
                torchmetrics.R2Score()
            ])
        else: # Binary Classification
            if pos_weight is not None:
                weight_tensor = torch.tensor(pos_weight, dtype=torch.float32)
                self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
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
        # 1. Concatenate
        combined = torch.cat([text_emb, context_emb], dim=1)
        
        # 2. Project
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
    Merges BERT embeddings with Subreddit2Vec embeddings.
    """
    # 1. Load BERT Embeddings
    print(f"Loading BERT embeddings from: {bert_path}")
    df_bert = pq.read_table(bert_path).to_pandas()
    df_bert = df_bert.dropna(subset=['label'])
    df_bert['id'] = df_bert['id'].astype(str)
    
    # 2. Load Edge Data
    print(f"Loading community edge data from: {edge_file_path}")
    df_edges = pq.read_table(
        edge_file_path,
        columns=['source_id', 'target_id']
    ).to_pandas()
    
    # Normalize IDs
    df_edges['source_id'] = df_edges['source_id'].astype(str)
    df_edges['target_id'] = df_edges['target_id'].astype(str).str.lower()
    df_edges.rename(columns={'source_id': 'id', 'target_id': 'community_name'}, inplace=True)
    df_edges.drop_duplicates(subset=['id'], inplace=True)
    
    # 3. Merge
    print("Merging BERT data with Community Edges...")
    df_merged = pd.merge(df_bert, df_edges, on='id', how='inner')
    print(f"Merged Data Size: {len(df_merged)} (Dropped {len(df_bert) - len(df_merged)} rows due to missing community mapping)")

    # 4. Load Subreddit2Vec
    print(f"Loading Subreddit2Vec vectors from: {vectors_path}")
    try:
        wv = KeyedVectors.load_word2vec_format(vectors_path, binary=True)
    except:
        try:
            loaded_obj = KeyedVectors.load(vectors_path)
            if hasattr(loaded_obj, 'wv'):
                wv = loaded_obj.wv
            else:
                wv = loaded_obj
        except Exception as e:
            raise ValueError(f"Could not load vectors file: {e}")
        
    vector_size = wv.vector_size
    
    # 5. Map
    print("Mapping subreddits to vectors...")
    def get_comm_vector(subreddit):
        sub_lower = str(subreddit).lower().strip()
        if sub_lower in wv:
            return wv[sub_lower]
        else:
            return np.zeros(vector_size, dtype=np.float32)

    context_vectors = np.vstack(df_merged['community_name'].apply(get_comm_vector).tolist())
    text_vectors = np.vstack(df_merged['embedding'].tolist())
    
    # 6. Tensors
    tensor_text = torch.tensor(text_vectors, dtype=torch.float32)
    tensor_context = torch.tensor(context_vectors, dtype=torch.float32)
    labels_dtype = torch.float32 if df_merged['label'].dtype == float else torch.long
    tensor_labels = torch.tensor(df_merged['label'].values, dtype=labels_dtype)
    
    print(f"Final Shapes :: Text: {tensor_text.shape}, Context: {tensor_context.shape}, Labels: {tensor_labels.shape}")
    dataset = TensorDataset(tensor_text, tensor_context, tensor_labels)
    return dataset, tensor_text.shape[1], tensor_context.shape[1]

# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Contextual Baseline for Dreaddit.")
    
    # Paths (Still need these via CLI)
    parser.add_argument("--train_path", required=True, help="Path to train folder.")
    parser.add_argument("--test_path", required=True, help="Path to test folder.")
    parser.add_argument("--edge_file_name", default="text_community_edges.parquet", help="Name of the edge file.")
    parser.add_argument("--vectors_path", required=True, help="Path to the .kv vectors file.")
    parser.add_argument("--save_dir", required=True, help="Where to save results.")
    
    # Task Config
    parser.add_argument("--task_name", required=True, help="Name of the task.")
    parser.add_argument("--task_type", required=True, choices=['binary', 'regression'], help="Type of task.")
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # --- Load Data ---
    train_bert = os.path.join(args.train_path, "text_node_embeddings_bert.parquet")
    train_edge = os.path.join(args.train_path, args.edge_file_name)
    test_bert = os.path.join(args.test_path, "text_node_embeddings_bert.parquet")
    test_edge = os.path.join(args.test_path, args.edge_file_name)

    train_dataset, text_dim, context_dim = load_and_prep_contextual_data(train_bert, train_edge, args.vectors_path)
    test_dataset, _, _ = load_and_prep_contextual_data(test_bert, test_edge, args.vectors_path)

    # --- Splits ---
    train_indices = np.arange(len(train_dataset))
    train_labels = train_dataset.tensors[2].numpy()
    
    # Stratified Split for Dreaddit
    print("--- Creating Stratified Validation Split ---")
    train_idx, val_idx = sk_train_test_split(
        train_indices, test_size=0.15, stratify=train_labels, random_state=RANDOM_SEED
    )
    
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)

    # --- Loaders ---
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    # --- Model ---
    model = LitContextualBaseline(
        text_dim=text_dim,
        context_dim=context_dim,
        hidden_dim=HIDDEN_DIM,
        task_type=args.task_type,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        dropout_rate=DROPOUT_RATE,
        pos_weight=POS_WEIGHT 
    )

    # --- Trainer ---
    wandb_logger = WandbLogger(
        project=f"GASTON_dreaddit_binary_contextual_finetuning",
        name=f"CTX_BASELINE_{args.task_name}",
        save_dir=args.save_dir
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.save_dir, "checkpoints"),
        save_top_k=1,
        monitor="val_BinaryAUROC",
        mode="max",
        filename='ctx_model-{epoch:02d}-{val_BinaryAUROC:.3f}'
    )

    early_stop_callback = EarlyStopping(
        monitor="val_BinaryAUROC",
        mode="max",
        patience=PATIENCE,
        verbose=True
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator='gpu',
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback]
    )

    print(f"--- Starting Contextual Baseline Training for {args.task_name} ---")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    print("--- Evaluation on Test Set ---")
    trainer.test(model, dataloaders=test_loader, ckpt_path='best')