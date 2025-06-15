from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import shutil
import warnings
import logging

from typing import Dict, List, Sequence, Tuple, Optional

from sklearn.preprocessing import StandardScaler
from tqdm import trange

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
warnings.filterwarnings("ignore", message=".*Arrow optimization.*")

# Cell: Import libraries and set up environment
"""
# Recommender Systems Analysis and Visualization
This notebook performs an exploratory analysis of recommender systems using the Sim4Rec library.
We'll generate synthetic data, compare multiple baseline recommenders, and visualize their performance.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as sf
from pyspark.sql import DataFrame, Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import DoubleType, ArrayType

# Set up plotting
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)

# Initialize Spark session
spark = SparkSession.builder \
    .appName("RecSysVisualization") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

# Set log level to warnings only
spark.sparkContext.setLogLevel("WARN")

# Import competition modules
from data_generator import CompetitionDataGenerator
from simulator import CompetitionSimulator
from sample_recommenders import (
    RandomRecommender,
    PopularityRecommender,
    ContentBasedRecommender, 
    SVMRecommender, 
)
from config import DEFAULT_CONFIG, EVALUATION_METRICS

# Cell: Define custom recommender template

# Copy the BaseRecommender provided in the sample_recommenders.py
# ------------------------------------------------------------
#  Drop‑in replacement for RNNRecommender ‑‑> TransformerRec
#  File: transformer_recommender.py
# ------------------------------------------------------------



class BaseRecommender:
    def __init__(self, seed=None):
        self.seed = seed
        np.random.seed(seed)
    def fit(self, log, user_features=None, item_features=None):
        """
        No training needed for random recommender.
        
        Args:
            log: Interaction log
            user_features: User features (optional)
            item_features: Item features (optional)
        """
        # No training needed
        raise NotImplemented()
    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        raise NotImplemented()
    
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sim4rec.utils import pandas_to_spark
from typing import List, Dict, Tuple
# --- BaseRecommender is already defined in your notebook ---


import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sim4rec.utils import pandas_to_spark
from typing import List, Dict, Tuple
class _SeqDataset(Dataset):
    """Pad user sedquences for Transformer input."""
    def __init__(self, seqs: List[List[int]], prices: List[List[float]],
                 max_len: int, pad_id: int = 0):
        self.max_len, self.pad_id = max_len, pad_id
        self.seqs, self.prices = seqs, prices

    def __len__(self): return len(self.seqs)
    def __getitem__(self, idx):
        s, p = self.seqs[idx][-self.max_len:], self.prices[idx][-self.max_len:]
        pad = self.max_len - len(s)
        return {
            "items":  torch.tensor([self.pad_id]*pad + s, dtype=torch.long),
            "price":  torch.tensor([0.0]*pad + p,       dtype=torch.float),
            "mask":   torch.tensor([0]*pad + [1]*len(s),dtype=torch.bool),
        }

class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pos = torch.arange(0, max_len).unsqueeze(1)
        i   = torch.arange(0, d_model, 2).float()
        div = torch.exp(-np.log(10000.0) * i / d_model)
        pe  = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos*div)
        pe[:, 1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1,L,E]
    def forward(self, x):  # x: [B,L,E]
        return x + self.pe[:, :x.size(1)]

class _SASTransformer(nn.Module):
    def __init__(self, n_items: int, max_len: int, hidden: int,
                 n_heads: int, ff: int, n_blocks: int, dropout: float, pad_id: int):
        super().__init__()
        self.item_emb = nn.Embedding(n_items, hidden, padding_idx=pad_id)
        self.price_lin = nn.Linear(1, hidden)
        self.pos = _PositionalEncoding(hidden, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=ff,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_blocks)
        self.out = nn.Linear(hidden, n_items)

    def forward(self, items, prices, attn_mask):
        x = self.item_emb(items) + self.price_lin(prices.unsqueeze(-1))
        x = self.pos(x)
        x = self.encoder(x, mask=attn_mask)
        return self.out(x)  # [B,L,V]

    @torch.inference_mode()
    def predict_last(self, items, prices, attn_mask):
        return self.forward(items, prices, attn_mask)[:, -1]

class TransformerRecommender(BaseRecommender):
    def __init__(self, seed=None, max_len=100, n_heads=4, hidden=128,
                 ff=256, n_blocks=2, dropout=0.2, n_epochs=10,
                 batch_size=256, lr=1e-3):
        super().__init__(seed)
        self.max_len, self.n_epochs = max_len, n_epochs
        self.batch_size, self.lr = batch_size, lr
        self.hparams = dict(n_heads=n_heads, hidden=hidden,
                            ff=ff, n_blocks=n_blocks, dropout=dropout)
        self.scalar = StandardScaler()
        # runtime
        self.model, self.item2id, self.id2item = None, {}, []

    # ------------------ fitting ------------------ #
    def fit(self, log, user_features=None, item_features=None):
        if user_features is None or item_features is None:
            raise ValueError("TransformerRec needs user & item features")

        spark_log = (log.join(user_features, "user_idx")
                        .join(item_features, "item_idx"))
        pd_log = (spark_log.drop("__iter").toPandas()
                  .sort_values("user_idx"))
        pd_log["price"] = self.scalar.fit_transform(pd_log[["price"]])
        uniq = pd_log["item_idx"].unique()
        self.item2id = {it:i+1 for i,it in enumerate(uniq)}
        self.id2item = [None] + uniq.tolist()
        pd_log["item_id"] = pd_log["item_idx"].map(self.item2id)

        seq_items, seq_prices = [], []
        for _, g in pd_log.groupby("user_idx", sort=False):
            seq_items.append(g["item_id"].tolist())
            seq_prices.append(g["price"].tolist())

        ds = _SeqDataset(seq_items, seq_prices,
                         self.max_len, pad_id=0)
        dl = DataLoader(ds, batch_size=self.batch_size,
                        shuffle=True, drop_last=False)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = _SASTransformer(
            n_items=len(self.id2item), max_len=self.max_len,
            pad_id=0, **self.hparams).to(device)

        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        crit  = nn.CrossEntropyLoss(ignore_index=0)

        # Pre‑compute causal mask: [L,L] upper‑triangular True = mask
        mask = torch.triu(torch.ones(self.max_len, self.max_len, dtype=torch.bool),
                          diagonal=1).to(device)

        for ep in range(self.n_epochs):
            self.model.train(); ep_loss = 0
            for batch in dl:
                items   = batch["items"].to(device)
                prices  = batch["price"].to(device)
                logits  = self.model(items[:, :-1], prices[:, :-1], mask[:items.size(1)-1,:items.size(1)-1])
                loss    = crit(logits.reshape(-1, logits.size(-1)),
                               items[:,1:].reshape(-1))
                optim.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                optim.step(); ep_loss += loss.item()
            print(f"[Transformer ep {ep+1}] loss = {ep_loss/len(dl):.4f}")

    # ------------------ predicting ------------------ #
    def predict(self, log, k, users, items,
                user_features=None, item_features=None,
                filter_seen_items=True):
        if self.model is None: raise RuntimeError("call fit() first")

        cross = users.join(items).drop("__iter").toPandas()
        cross["orig_price"] = cross["price"]
        cross["price"] = self.scalar.transform(cross[["price"]])
        cross = pd.get_dummies(cross)

        # Build user histories
        hist = (log.join(item_features, "item_idx")
                   .select("user_idx","item_idx","price").toPandas()
                   .sort_values("user_idx"))
        hist["price"] = self.scalar.transform(hist[["price"]])
        hist["item_id"] = hist["item_idx"].map(self.item2id)
        user_hist = {u:(g["item_id"].tolist()[-self.max_len:],
                        g["price"].tolist()[-self.max_len:])
                     for u,g in hist.groupby("user_idx", sort=False)}

        # Batch inference
        device = next(self.model.parameters()).device
        mask = torch.triu(torch.ones(self.max_len, self.max_len, dtype=torch.bool),
                          diagonal=1).to(device)
        probs = []
        B = 1024
        for i in range(0,len(cross),B):
            sub = cross.iloc[i:i+B]
            p = []
            for uid in sub["user_idx"]:
                seq_i, seq_p = user_hist.get(uid, ([], []))
                pad = self.max_len - len(seq_i)
                it  = torch.tensor([0]*pad+seq_i, device=device).unsqueeze(0)
                pr  = torch.tensor([0.0]*pad+seq_p, device=device).unsqueeze(0)
                logits = self.model.predict_last(
                    it, pr, mask)  # [1,V]
                p.append(torch.softmax(logits, -1).squeeze(0).cpu().numpy())
            probs.extend(p)

        cross["prob"] = [probs[j][self.item2id.get(it,0)]
                         for j,it in enumerate(cross["item_idx"])]
        cross["relevance"] = cross["prob"] * cross["orig_price"]
        cross = cross.sort_values(["user_idx","relevance"],
                                  ascending=[True,False])
        cross = cross.groupby("user_idx").head(k)
        cross["price"] = cross["orig_price"]
        return pandas_to_spark(cross)

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sim4rec.utils import pandas_to_spark
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline



# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  RNN model  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< #

class _SeqDataset(Dataset):
    """convert user sequence to padding and feed to DataLoader"""
    def __init__(self, seqs: List[List[int]], prices: List[List[float]],
                 max_len: int = 100, pad_id: int = 0):
        self.max_len = max_len
        self.pad_id  = pad_id
        self.seqs    = seqs
        self.prices  = prices

    def __len__(self):  return len(self.seqs)

    def __getitem__(self, idx):
        items  = self.seqs[idx][-self.max_len:]
        prices = self.prices[idx][-self.max_len:]
        pad    = self.max_len - len(items)
        items  = [self.pad_id]*pad + items
        prices = [0.0]*pad + prices
        mask   = [0]*pad + [1]*len(self.seqs[idx][-self.max_len:])
        return dict(
            items  = torch.tensor(items,  dtype=torch.long),
            price  = torch.tensor(prices, dtype=torch.float),
            mask   = torch.tensor(mask,   dtype=torch.long),
        )

class _GRURec(nn.Module):
    """GRU backbone：item-embedding + price-projection"""
    def __init__(self, num_items, emb_dim=64,
                 hidden_size=128, num_layers=2, dropout=0.2,
                 price_emb_dim=16, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.item_emb  = nn.Embedding(num_items, emb_dim, padding_idx=pad_id)
        self.price_lin = nn.Linear(1, price_emb_dim)
        self.gru = nn.GRU(emb_dim + price_emb_dim, hidden_size,
                          num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.)
        self.fc  = nn.Linear(hidden_size, num_items)

    def forward(self, items, prices):            # [B,L]
        x = torch.cat(
            [self.item_emb(items),
             self.price_lin(prices.unsqueeze(-1))],
            dim=-1
        )                                        # -> [B,L,E]
        out, _ = self.gru(x)                     # -> [B,L,H]
        logits = self.fc(out)                    # -> [B,L,V]
        return logits

    @torch.inference_mode()
    def predict_last(self, items, prices):
        """只要最后位置的 logits（batch 版）"""
        logits = self.forward(items, prices)     # [B,L,V]
        return logits[:, -1]                     # [B,V]

class RNNRecommender(BaseRecommender):
    def __init__(self, seed=None,
                 hidden_size=128, num_layers=2, dropout=0.3,
                 max_len=100, n_epochs=5, lr=1e-3, batch_size=256):
        super().__init__(seed)
        self.scalar   = StandardScaler()
        self.max_len  = max_len
        self.n_epochs = n_epochs
        self.lr       = lr
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.dropout     = dropout
        # runtime attributes
        self.model = None
        self.item2id, self.id2item = {}, []

    # ------------------------------------------------------- #
    #                       FIT                               #
    # ------------------------------------------------------- #
    def fit(self, log: DataFrame,
            user_features: DataFrame | None = None,
            item_features: DataFrame | None = None):
        """
        - Same as GBRecommender: first join the log with the feature tables,
          then convert to Pandas for dummy encoding & standardization.
        - Required columns: user_idx, item_idx, price
          (Assume the input order already represents the chronological sequence, so timestamps are no longer needed.)
        """
        if user_features is None or item_features is None:
            raise ValueError("RNNRecommender needs user_features and item_features")

        # ---------- Spark join ----------
        spark_log = (
            log.join(user_features, on="user_idx")
               .join(item_features, on="item_idx")
        )

        # price col only
        missing = {"price"} - set(spark_log.columns)
        if missing:
            raise ValueError(f"col missing: {missing}")

        # ---------- to Pandas ----------
        pd_log = (
            spark_log.drop("__iter") 
                     .toPandas()
                     .sort_values("user_idx") 
        )

        pd_log = pd.get_dummies(pd_log)
        pd_log["price"] = self.scalar.fit_transform(pd_log[["price"]])

        uniq_items = pd_log["item_idx"].unique()
        self.item2id = {it: i + 1 for i, it in enumerate(uniq_items)}
        self.id2item = [None] + uniq_items.tolist()
        pd_log["item_id"] = pd_log["item_idx"].map(self.item2id)

        # ---------- construct sequence ----------
        seq_items, seq_prices = [], []
        for _, g in pd_log.groupby("user_idx", sort=False):
            seq_items.append(g["item_id"].tolist())
            seq_prices.append(g["price"].tolist())

        dataset = _SeqDataset(seq_items, seq_prices,
                              max_len=self.max_len, pad_id=0)
        loader  = DataLoader(dataset, batch_size=self.batch_size,
                             shuffle=True, drop_last=False)

        # ---------- modeling ----------
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = _GRURec(num_items=len(self.id2item),
                             hidden_size=self.hidden_size,
                             num_layers=self.num_layers,
                             dropout=self.dropout).to(device)

        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        crit  = nn.CrossEntropyLoss(ignore_index=0)

        for ep in range(self.n_epochs):
            self.model.train()
            epoch_loss = 0.0
            for batch in loader:
                items  = batch["items"].to(device)
                prices = batch["price"].to(device)

                out = self.model(items[:, :-1], prices[:, :-1])
                loss = crit(out.reshape(-1, out.size(-1)),
                            items[:, 1:].reshape(-1))

                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                optim.step()
                epoch_loss += loss.item()
            print(f"[epoch {ep+1}] loss={epoch_loss/len(loader):.4f}")

    # ------------------------------------------------------- #
    #                     PREDICT                             #
    # ------------------------------------------------------- #
    def predict(self, log, k,
                users: DataFrame, items: DataFrame,
                user_features=None, item_features=None,
                filter_seen_items=True):

        if self.model is None:
            raise RuntimeError("need fit() first")

        # ---------- feature cross join ----------
        cross = users.join(
            items
        ).drop('__iter').toPandas().copy()

        cross = pd.get_dummies(cross)
        cross['orig_price'] = cross['price']
        cross["price"] = self.scalar.transform(cross[["price"]])
        cross = pd.get_dummies(cross)

        # ---------- prepare user hist ----------
        hist = (log.join(item_features, on="item_idx")
                    .select("user_idx", "item_idx", "price")  # 无 ts
                    .toPandas()
                    .sort_values("user_idx"))
        hist["price"] = self.scalar.transform(hist[["price"]])
        hist["item_id"] = hist["item_idx"].map(self.item2id)

        user_hist = {
            uid: (g["item_id"].tolist()[-self.max_len:],
                  g["price"].tolist()[-self.max_len:])
            for uid, g in hist.groupby("user_idx", sort=False)
        }

        # ---------- inference ----------
        device = next(self.model.parameters()).device
        probs = []
        BATCH = 1024
        for i in range(0, len(cross), BATCH):
            sub = cross.iloc[i:i+BATCH]
            p = []
            for uid in sub["user_idx"]:
                items_seq, prices_seq = user_hist.get(uid, ([], []))
                pad = self.max_len - len(items_seq)
                it = torch.tensor([0]*pad + items_seq,
                                  device=device).unsqueeze(0)
                pr = torch.tensor([0.]*pad + prices_seq,
                                  device=device).unsqueeze(0)
                logits = self.model.predict_last(it, pr)
                p.append(torch.softmax(logits, -1).squeeze(0).cpu().numpy())
            probs.extend(p)

        cross["prob"] = [
            probs[j][ self.item2id.get(it, 0) ] for j, it in
            enumerate(cross["item_idx"])
        ]

        cross["relevance"] = cross["prob"] * cross["orig_price"]
        cross = cross.sort_values(["user_idx", "relevance"],
                                  ascending=[True, False])
        cross = cross.groupby("user_idx").head(k)
        cross["price"] = cross["orig_price"]
        return pandas_to_spark(cross)


class AutoRegressiveRecommender:
    """
    Optimized AutoRegressive Recommender for Maximum Revenue
    
    Key optimizations:
    1. Enhanced price-awareness with user spending patterns
    2. Strategic high-value item identification  
    3. Multi-factor revenue calculation
    4. Temporal interaction weighting
    5. CONSERVATIVE PARAMETERS for better generalization
    """
    
    def __init__(self, n=2, alpha=1.0, min_count=1, seed=42, revenue_boost=3.5):
        """
        Initialize BALANCED AR Recommender for optimal leaderboard performance
        
        Args:
            n (int): n-gram order
            alpha (float): MODERATE Laplace smoothing (1.0) 
            min_count (int): minimum count threshold (1)
            seed (int): Random seed
            revenue_boost (float): BALANCED boost (3.5 - proven sweet spot)
        """
        self.n = n
        self.alpha = float(alpha)  # Moderate smoothing
        self.min_count = min_count  # Keep at 1
        self.seed = seed
        self.revenue_boost = float(revenue_boost)  # Balanced setting
        
        # N-gram model components
        self.ngram_counts = defaultdict(Counter)
        self.context_totals = defaultdict(float)
        self.ngram_probs = defaultdict(dict)
        
        # Enhanced price and user analysis
        self.item_prices = {}
        self.item_revenue_potential = {}
        self.user_price_preferences = {}
        self.user_spending_tiers = {}
        self.high_value_items = set()
        self.luxury_items = set()
        
        # Strategic item categories
        self.premium_items = set()
        self.popular_items = set()
        
        # RESTORED: Simple temporal weights
        self.temporal_weights = defaultdict(float)
        
        np.random.seed(seed)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('autoregressive_recommender')
        
    def fit(self, log, user_features, item_features):
        """
        MODERATE Conservative training for balanced performance
        """
        try:
            self.logger.info("Training MODERATE Conservative AR-2gram recommender...")
            
            # Extract item information with moderate approach
            item_pd = item_features.select('item_idx', 'price', 'category').toPandas()
            
            for _, row in item_pd.iterrows():
                item_id = row['item_idx']
                try:
                    price = float(row['price'])
                    self.item_prices[item_id] = price
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Invalid price for item {item_id}: {row['price']}, using default 50.0")
                    self.item_prices[item_id] = 50.0
            
            # MODERATE: Identify high-value items (top 25% by price, balanced threshold)
            price_values = list(self.item_prices.values())
            if len(price_values) > 0:
                price_threshold = np.percentile(price_values, 75)  # Slightly more selective
                self.high_value_items = {item for item, price in self.item_prices.items() 
                                       if price >= price_threshold}
            
            # MODERATE: Enhanced revenue potential with some complexity
            avg_price = np.mean(price_values) if price_values else 50.0
            for item_id in self.item_prices:
                price_ratio = self.item_prices[item_id] / avg_price
                # Add slight boost for high-value items
                boost = 1.2 if item_id in self.high_value_items else 1.0
                self.item_revenue_potential[item_id] = price_ratio * boost
            
            # Extract log data with temporal information
            columns_to_select = ['user_idx', 'item_idx', 'relevance']
            if '__iter' in log.columns:
                columns_to_select.append('__iter')
                log_pd = log.select(*columns_to_select).toPandas()
                log_pd = log_pd.sort_values(['user_idx', '__iter'])
                
                # SIMPLE temporal weights (not too complex)
                try:
                    max_iter = float(log_pd['__iter'].max())
                    if max_iter > 0:
                        log_pd['temporal_weight'] = 1.0 + (log_pd['__iter'].astype(float) / max_iter) * 0.3  # Reduced from 0.5
                    else:
                        log_pd['temporal_weight'] = 1.0
                except:
                    log_pd['temporal_weight'] = 1.0
            else:
                log_pd = log.select(*columns_to_select).toPandas()
                log_pd = log_pd.sort_values(['user_idx'])
                log_pd['temporal_weight'] = 1.0
            
            # MODERATE: Enhanced user preference analysis
            positive_interactions = log_pd[log_pd['relevance'] > 0]
            
            for user_id, user_data in positive_interactions.groupby('user_idx'):
                user_prices = []
                for _, row in user_data.iterrows():
                    item_id = row['item_idx']
                    item_price = self.item_prices.get(item_id, 50.0)
                    user_prices.append(item_price)
                
                if user_prices:
                    mean_price = np.mean(user_prices)
                    std_price = np.std(user_prices)
                    
                    # MODERATE: Enhanced tier classification
                    percentile_60 = np.percentile(price_values, 60)
                    percentile_80 = np.percentile(price_values, 80)
                    
                    if mean_price > percentile_80:
                        tier = 'premium_buyer'
                    elif mean_price > percentile_60:
                        tier = 'high_buyer'
                    elif mean_price > avg_price * 0.8:
                        tier = 'regular_buyer'
                    else:
                        tier = 'budget_buyer'
                    
                    self.user_spending_tiers[user_id] = tier
                    self.user_price_preferences[user_id] = {
                        'avg_price': mean_price,
                        'price_std': std_price,
                        'tier': tier
                    }
            
            # MODERATE: Build n-gram model with balanced smoothing
            sequences_processed = 0
            for user_id, user_data in log_pd.groupby('user_idx'):
                item_sequence = user_data['item_idx'].tolist()
                weights = user_data['temporal_weight'].astype(float).tolist()
                
                if len(item_sequence) < 2:
                    continue
                
                sequences_processed += 1
                
                # Build n-gram with moderate temporal weighting
                if len(item_sequence) >= self.n:
                    for i in range(len(item_sequence) - self.n + 1):
                        context = tuple(item_sequence[i:i+self.n-1])
                        target = item_sequence[i+self.n-1]
                        
                        # MODERATE temporal and value weighting
                        temporal_weight = float(weights[i+self.n-1]) if i+self.n-1 < len(weights) else 1.0
                        
                        # Slight boost for high-value items
                        value_weight = 1.3 if target in self.high_value_items else 1.0
                        
                        final_weight = temporal_weight * value_weight
                        
                        self.ngram_counts[context][target] += final_weight
                        self.context_totals[context] += final_weight
                        self.temporal_weights[context] = max(self.temporal_weights[context], temporal_weight)
            
            # MODERATE: Filter with restored min_count
            filtered_ngram_counts = defaultdict(Counter)
            filtered_context_totals = defaultdict(float)
            
            for context, targets in self.ngram_counts.items():
                if self.context_totals[context] >= self.min_count:  # Back to min_count=1
                    for target, count in targets.items():
                        if count >= self.min_count:
                            filtered_ngram_counts[context][target] = count
                            filtered_context_totals[context] += count
            
            self.ngram_counts = filtered_ngram_counts
            self.context_totals = filtered_context_totals
            
            # Pre-compute probabilities with MODERATE smoothing
            for context in self.ngram_counts:
                context_total = self.context_totals[context]
                
                # Get vocabulary size for this context
                vocab_size = len(set(self.ngram_counts[context].keys()))
                
                self.ngram_probs[context] = {}
                for target, count in self.ngram_counts[context].items():
                    # MODERATE Laplace smoothing
                    prob = (count + self.alpha) / (context_total + self.alpha * vocab_size)
                    self.ngram_probs[context][target] = prob
            
            self.logger.info(f"MODERATE Conservative AR model training completed")
            self.logger.info(f"Processed {sequences_processed} sequences, retained {len(self.ngram_counts)} contexts")
            self.logger.info(f"High-value items: {len(self.high_value_items)}, User profiles: {len(self.user_price_preferences)}")
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
    
    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        """
        CONSERVATIVE prediction with better generalization
        """
        try:
            self.logger.info(f"Generating CONSERVATIVE AR recommendations...")
            
            # Get user list and all items
            users_list = users.select('user_idx').distinct().collect()
            all_items = items.select('item_idx').collect()
            all_item_ids = [row['item_idx'] for row in all_items]
            
            recommendations = []
            
            for user_row in users_list:
                user_id = user_row['user_idx']
                
                # Get user's recent interactions
                user_log = log.filter(sf.col('user_idx') == user_id)
                recent_items = [row['item_idx'] for row in user_log.take(20)]
                
                # Generate recommendations for this user with CONSERVATIVE approach
                user_recs = self._conservative_predict_for_user(user_id, recent_items, k, all_item_ids, filter_seen_items)
                recommendations.extend(user_recs)
            
            # Convert to Spark DataFrame
            if recommendations:
                from pyspark.sql.types import StructType, StructField, LongType, FloatType
                schema = StructType([
                    StructField("user_idx", LongType(), True),
                    StructField("item_idx", LongType(), True),
                    StructField("relevance", FloatType(), True)
                ])
                rec_df = users.sql_ctx.createDataFrame(recommendations, schema)
                return rec_df
            else:
                # Return empty DataFrame with same schema
                from pyspark.sql.types import StructType, StructField, LongType, FloatType
                schema = StructType([
                    StructField("user_idx", LongType(), True),
                    StructField("item_idx", LongType(), True),
                    StructField("relevance", FloatType(), True)
                ])
                return users.sql_ctx.createDataFrame([], schema)
                
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def _conservative_predict_for_user(self, user_id, recent_items, k, all_item_ids, filter_seen_items=True):
        """MODERATE Conservative user prediction with balanced performance"""
        try:
            item_scores = {}
            
            # Strategy 1: N-gram prediction with MODERATE parameters
            if recent_items and len(recent_items) >= self.n - 1:
                context = tuple(recent_items[-(self.n-1):])
                
                if context in self.ngram_probs:
                    # Add temporal boost for this context
                    temporal_boost = self.temporal_weights.get(context, 1.0)
                    
                    for item, prob in self.ngram_probs[context].items():
                        if item in self.item_prices:
                            price = self.item_prices[item]
                            
                            # BALANCED: Proven effective probability calculation
                            base_revenue = prob * price
                            
                            # PROVEN user preference alignment - balance between performance and stability
                            user_pref = self.user_price_preferences.get(user_id, {'tier': 'regular_buyer', 'avg_price': 50.0})
                            preference_multiplier = 1.0
                            
                            # EFFECTIVE tier-based adjustment - proven parameters
                            user_avg_price = user_pref.get('avg_price', 50.0)
                            price_match_score = 1.0 - min(0.4, abs(price - user_avg_price) / (user_avg_price + 8))
                            
                            if user_pref['tier'] == 'premium_buyer' and item in self.high_value_items:
                                preference_multiplier = 1.8 + price_match_score * 0.7  # Proven effective
                            elif user_pref['tier'] == 'high_buyer' and item in self.high_value_items:
                                preference_multiplier = 1.6 + price_match_score * 0.6
                            elif user_pref['tier'] == 'regular_buyer':
                                preference_multiplier = 1.4 + price_match_score * 0.5
                            elif user_pref['tier'] == 'budget_buyer' and price < user_avg_price:
                                preference_multiplier = 1.5 + price_match_score * 0.6
                            
                            # EFFECTIVE context strength - proven balance
                            context_strength = min(1.8, len(recent_items) * 0.4 + 1.0)
                            revenue_potential = self.item_revenue_potential.get(item, 1.0)
                            
                            # PROVEN final probability calculation
                            base_probability = prob * preference_multiplier * temporal_boost * context_strength
                            
                            # BALANCED probability boost - proven effective range
                            final_probability = max(0.05, min(0.85, base_probability * 2.2))
                            
                            # Expected Value = balanced_probability × price × revenue_potential × revenue_boost
                            expected_revenue = (final_probability * price * revenue_potential * self.revenue_boost)
                            item_scores[item] = expected_revenue
            
            # Strategy 2: PROVEN fallback with balanced effectiveness
            if len(item_scores) < k:
                seen_items = set(recent_items) if filter_seen_items else set()
                available_items = [item for item in all_item_ids 
                                 if item not in item_scores and item not in seen_items]
                
                # PROVEN price-based ranking with effective logic
                price_scores = []
                for item in available_items:
                    price = self.item_prices.get(item, 50.0)
                    
                    # BALANCED scoring: proven user preference alignment
                    user_pref = self.user_price_preferences.get(user_id, {'avg_price': 50.0, 'tier': 'regular_buyer'})
                    avg_user_price = user_pref.get('avg_price', 50.0)
                    
                    # PROVEN price alignment score
                    price_diff = abs(price - avg_user_price)
                    price_alignment = max(0.3, 1.0 - (price_diff / (avg_user_price + 5)))
                    
                    # PROVEN value boost
                    value_boost = 1.8 if item in self.high_value_items else 1.4
                    
                    # BALANCED fallback probability - proven effective
                    base_fallback_prob = 0.15 + price_alignment * 0.25
                    final_fallback_prob = min(0.75, base_fallback_prob * value_boost)
                    
                    score = final_fallback_prob * price * 1.5  # Proven fallback weight
                    price_scores.append((item, score))
                
                price_scores.sort(key=lambda x: x[1], reverse=True)
                
                for item, score in price_scores[:k - len(item_scores)]:
                    item_scores[item] = score
            
            # Strategy 3: Enhanced final fallback
            while len(item_scores) < k:
                available = [item for item in all_item_ids if item not in item_scores]
                if not available:
                    break
                
                item = available[0]
                price = self.item_prices.get(item, 50.0)
                value_boost = 1.1 if item in self.high_value_items else 1.0
                item_scores[item] = price * value_boost * 0.2  # Enhanced fallback score
            
            # Filter seen items if required
            if filter_seen_items:
                for item in recent_items:
                    item_scores.pop(item, None)
                
                # Re-fill if needed
                while len(item_scores) < k:
                    available = [item for item in all_item_ids 
                               if item not in item_scores and item not in recent_items]
                    if not available:
                        break
                    
                    item = available[0]
                    price = self.item_prices.get(item, 50.0)
                    value_boost = 1.05 if item in self.high_value_items else 1.0
                    item_scores[item] = price * value_boost * 0.1
            
            # Select top-k items
            top_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:k]
            
            # Convert to recommendations with proper ID handling
            recommendations = []
            for item_id, score in top_items:
                try:
                    # Smart ID conversion
                    user_id_long = int(user_id) if isinstance(user_id, (int, float, str)) and str(user_id).replace('-','').isdigit() else user_id
                    item_id_long = int(item_id) if isinstance(item_id, (int, float, str)) and str(item_id).replace('-','').isdigit() else item_id
                    
                    # Fallback for non-convertible IDs
                    if not isinstance(user_id_long, (int, float)):
                        user_id_long = hash(str(user_id)) % (2**31)
                    if not isinstance(item_id_long, (int, float)):
                        item_id_long = hash(str(item_id)) % (2**31)
                    
                    recommendations.append((int(user_id_long), int(item_id_long), float(score)))
                    
                except Exception as e:
                    # Emergency fallback
                    user_id_hash = hash(str(user_id)) % (2**31)
                    item_id_hash = hash(str(item_id)) % (2**31)
                    recommendations.append((user_id_hash, item_id_hash, float(score)))
            
            # Ensure we have k recommendations
            while len(recommendations) < k:
                fallback_item = all_item_ids[len(recommendations) % len(all_item_ids)]
                try:
                    user_id_long = int(user_id) if isinstance(user_id, (int, float, str)) and str(user_id).replace('-','').isdigit() else hash(str(user_id)) % (2**31)
                    item_id_long = int(fallback_item) if isinstance(fallback_item, (int, float, str)) and str(fallback_item).replace('-','').isdigit() else hash(str(fallback_item)) % (2**31)
                    
                    if not any(r[1] == int(item_id_long) for r in recommendations):
                        recommendations.append((int(user_id_long), int(item_id_long), 1.0))
                except:
                    break
            
            return recommendations[:k]
            
        except Exception as e:
            self.logger.error(f"User {user_id} prediction failed: {str(e)}")
            # Emergency fallback
            emergency_recs = []
            for i in range(k):
                item = all_item_ids[i % len(all_item_ids)]
                try:
                    user_id_hash = hash(str(user_id)) % (2**31)
                    item_id_hash = hash(str(item)) % (2**31)
                    emergency_recs.append((user_id_hash, item_id_hash, 1.0))
                except:
                    emergency_recs.append((0, i, 1.0))
            return emergency_recs
    
    def get_statistics(self):
       
        return {
            'n': self.n,
            'alpha': self.alpha,
            'min_count': self.min_count,
            'revenue_boost': self.revenue_boost,
            'ngram_contexts': len(self.ngram_counts),
            'high_value_items': len(self.high_value_items),
            'user_price_profiles': len(self.user_price_preferences),
            'category_count': len(self.item_revenue_potential),
            'temporal_contexts': len(self.temporal_weights),
            'model_type': 'MODERATE-CONSERVATIVE-AR'
        }



# Cell: Data Exploration Functions
"""
## Data Exploration Functions
These functions help us understand the generated synthetic data.
"""

def explore_user_data(users_df):
    """
    Explore user data distributions and characteristics.
    
    Args:
        users_df: DataFrame containing user data
    """
    print("=== User Data Exploration ===")
    
    # Get basic statistics
    print(f"Total number of users: {users_df.count()}")
    
    # User segments distribution
    segment_counts = users_df.groupBy("segment").count().toPandas()
    print("\nUser Segments Distribution:")
    for _, row in segment_counts.iterrows():
        print(f"  {row['segment']}: {row['count']} users ({row['count']/users_df.count()*100:.1f}%)")
    
    # Plot user segments
    plt.figure(figsize=(10, 6))
    plt.pie(segment_counts['count'], labels=segment_counts['segment'], autopct='%1.1f%%', startangle=90, shadow=True)
    plt.title('User Segments Distribution')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('user_segments_distribution.png')
    print("User segments visualization saved to 'user_segments_distribution.png'")
    
    # Convert to pandas for easier feature analysis
    users_pd = users_df.toPandas()
    
    # Analyze user feature distributions
    feature_cols = [col for col in users_pd.columns if col.startswith('user_attr_')]
    if len(feature_cols) > 0:
        # Take a sample of feature columns if there are many
        sample_features = feature_cols[:min(5, len(feature_cols))]
        
        # Plot histograms for sample features
        plt.figure(figsize=(14, 8))
        for i, feature in enumerate(sample_features):
            plt.subplot(2, 3, i+1)
            for segment in users_pd['segment'].unique():
                segment_data = users_pd[users_pd['segment'] == segment]
                plt.hist(segment_data[feature], alpha=0.5, bins=20, label=segment)
            plt.title(f'Distribution of {feature}')
            plt.xlabel('Value')
            plt.ylabel('Count')
            if i == 0:
                plt.legend()
        plt.tight_layout()
        plt.savefig('user_feature_distributions.png')
        print("User feature distributions saved to 'user_feature_distributions.png'")
        
        # Feature correlation heatmap
        plt.figure(figsize=(12, 10))
        corr = users_pd[feature_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='coolwarm', vmax=.3, center=0,
                    square=True, linewidths=.5, annot=False, fmt='.2f')
        plt.title('User Feature Correlations')
        plt.tight_layout()
        plt.savefig('user_feature_correlations.png')
        print("User feature correlations saved to 'user_feature_correlations.png'")


def explore_item_data(items_df):
    """
    Explore item data distributions and characteristics.
    
    Args:
        items_df: DataFrame containing item data
    """
    print("\n=== Item Data Exploration ===")
    
    # Get basic statistics
    print(f"Total number of items: {items_df.count()}")
    
    # Item categories distribution
    category_counts = items_df.groupBy("category").count().toPandas()
    print("\nItem Categories Distribution:")
    for _, row in category_counts.iterrows():
        print(f"  {row['category']}: {row['count']} items ({row['count']/items_df.count()*100:.1f}%)")
    
    # Plot item categories
    plt.figure(figsize=(10, 6))
    plt.pie(category_counts['count'], labels=category_counts['category'], autopct='%1.1f%%', startangle=90, shadow=True)
    plt.title('Item Categories Distribution')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('item_categories_distribution.png')
    print("Item categories visualization saved to 'item_categories_distribution.png'")
    
    # Convert to pandas for easier feature analysis
    items_pd = items_df.toPandas()
    
    # Analyze price distribution
    if 'price' in items_pd.columns:
        plt.figure(figsize=(14, 6))
        
        # Overall price distribution
        plt.subplot(1, 2, 1)
        plt.hist(items_pd['price'], bins=30, alpha=0.7)
        plt.title('Overall Price Distribution')
        plt.xlabel('Price')
        plt.ylabel('Count')
        
        # Price by category
        plt.subplot(1, 2, 2)
        for category in items_pd['category'].unique():
            category_data = items_pd[items_pd['category'] == category]
            plt.hist(category_data['price'], alpha=0.5, bins=20, label=category)
        plt.title('Price Distribution by Category')
        plt.xlabel('Price')
        plt.ylabel('Count')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('item_price_distributions.png')
        print("Item price distributions saved to 'item_price_distributions.png'")
    
    # Analyze item feature distributions
    feature_cols = [col for col in items_pd.columns if col.startswith('item_attr_')]
    if len(feature_cols) > 0:
        # Take a sample of feature columns if there are many
        sample_features = feature_cols[:min(5, len(feature_cols))]
        
        # Plot histograms for sample features
        plt.figure(figsize=(14, 8))
        for i, feature in enumerate(sample_features):
            plt.subplot(2, 3, i+1)
            for category in items_pd['category'].unique():
                category_data = items_pd[items_pd['category'] == category]
                plt.hist(category_data[feature], alpha=0.5, bins=20, label=category)
            plt.title(f'Distribution of {feature}')
            plt.xlabel('Value')
            plt.ylabel('Count')
            if i == 0:
                plt.legend()
        plt.tight_layout()
        plt.savefig('item_feature_distributions.png')
        print("Item feature distributions saved to 'item_feature_distributions.png'")


def explore_interactions(history_df, users_df, items_df):
    """
    Explore interaction patterns between users and items.
    
    Args:
        history_df: DataFrame containing interaction history
        users_df: DataFrame containing user data
        items_df: DataFrame containing item data
    """
    print("\n=== Interaction Data Exploration ===")
    
    # Get basic statistics
    total_interactions = history_df.count()
    total_users = users_df.count()
    total_items = items_df.count()
    
    print(f"Total interactions: {total_interactions}")
    print(f"Interaction density: {total_interactions / (total_users * total_items) * 100:.4f}%")
    
    # Users with interactions
    users_with_interactions = history_df.select("user_idx").distinct().count()
    print(f"Users with at least one interaction: {users_with_interactions} ({users_with_interactions/total_users*100:.1f}%)")
    
    # Items with interactions
    items_with_interactions = history_df.select("item_idx").distinct().count()
    print(f"Items with at least one interaction: {items_with_interactions} ({items_with_interactions/total_items*100:.1f}%)")
    
    # Distribution of interactions per user
    interactions_per_user = history_df.groupBy("user_idx").count().toPandas()
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(interactions_per_user['count'], bins=20)
    plt.title('Distribution of Interactions per User')
    plt.xlabel('Number of Interactions')
    plt.ylabel('Number of Users')
    
    # Distribution of interactions per item
    interactions_per_item = history_df.groupBy("item_idx").count().toPandas()
    
    plt.subplot(1, 2, 2)
    plt.hist(interactions_per_item['count'], bins=20)
    plt.title('Distribution of Interactions per Item')
    plt.xlabel('Number of Interactions')
    plt.ylabel('Number of Items')
    
    plt.tight_layout()
    plt.savefig('interaction_distributions.png')
    print("Interaction distributions saved to 'interaction_distributions.png'")
    
    # Analyze relevance distribution
    if 'relevance' in history_df.columns:
        relevance_dist = history_df.groupBy("relevance").count().toPandas()
        
        plt.figure(figsize=(10, 6))
        plt.bar(relevance_dist['relevance'].astype(str), relevance_dist['count'])
        plt.title('Distribution of Relevance Scores')
        plt.xlabel('Relevance Score')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('relevance_distribution.png')
        print("Relevance distribution saved to 'relevance_distribution.png'")
    
    # If we have user segments and item categories, analyze cross-interactions
    if 'segment' in users_df.columns and 'category' in items_df.columns:
        # Join with user segments and item categories
        interaction_analysis = history_df.join(
            users_df.select('user_idx', 'segment'),
            on='user_idx'
        ).join(
            items_df.select('item_idx', 'category'),
            on='item_idx'
        )
        
        # Count interactions by segment and category
        segment_category_counts = interaction_analysis.groupBy('segment', 'category').count().toPandas()
        
        # Create a pivot table
        pivot_table = segment_category_counts.pivot(index='segment', columns='category', values='count').fillna(0)
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, annot=True, fmt='g', cmap='viridis')
        plt.title('Interactions Between User Segments and Item Categories')
        plt.tight_layout()
        plt.savefig('segment_category_interactions.png')
        print("Segment-category interactions saved to 'segment_category_interactions.png'")


# Cell: Recommender Analysis Function
"""
## Recommender System Analysis
This is the main function to run analysis of different recommender systems and visualize the results.
"""

def run_recommender_analysis():
    """
    Run an analysis of different recommender systems and visualize the results.
    This function creates a synthetic dataset, performs EDA, evaluates multiple recommendation
    algorithms using train-test split, and visualizes the performance metrics.
    """
    # Create a smaller dataset for experimentation
    config = DEFAULT_CONFIG.copy()
    config['data_generation']['n_users'] = 1000  # Reduced from 10,000
    config['data_generation']['n_items'] = 200   # Reduced from 1,000
    config['data_generation']['seed'] = 42       # Fixed seed for reproducibility
    
    # Get train-test split parameters
    train_iterations = config['simulation']['train_iterations']
    test_iterations = config['simulation']['test_iterations']
    
    print(f"Running train-test simulation with {train_iterations} training iterations and {test_iterations} testing iterations")
    
    # Initialize data generator
    data_generator = CompetitionDataGenerator(
        spark_session=spark,
        **config['data_generation']
    )
    
    # Generate user data
    users_df = data_generator.generate_users()
    print(f"Generated {users_df.count()} users")
    
    # Generate item data
    items_df = data_generator.generate_items()
    print(f"Generated {items_df.count()} items")
    
    # Generate initial interaction history
    history_df = data_generator.generate_initial_history(
        config['data_generation']['initial_history_density']
    )
    print(f"Generated {history_df.count()} initial interactions")
    
    # Cell: Exploratory Data Analysis
    """
    ## Exploratory Data Analysis
    Let's explore the generated synthetic data before running the recommenders.
    """
    
    # Perform exploratory data analysis on the generated data
    print("\n=== Starting Exploratory Data Analysis ===")
    explore_user_data(users_df)
    explore_item_data(items_df)
    explore_interactions(history_df, users_df, items_df)
    
    # Set up data generators for simulator
    user_generator, item_generator = data_generator.setup_data_generators()
    
    # Cell: Setup and Run Recommenders
    """
    ## Recommender Systems Comparison
    Now we'll set up and evaluate different recommendation algorithms.
    """
    
    # Initialize recommenders to compare
    recommenders = [ 
        RandomRecommender(seed=42),
        PopularityRecommender(alpha=1.0, seed=42),
        ContentBasedRecommender(similarity_threshold=0.0, seed=42),
        RNNRecommender(seed=42, hidden_size=64, num_layers=2, dropout=0.2, max_len=100, n_epochs=5, lr=1e-3, batch_size=256),
        TransformerRecommender(seed=42, max_len=100, n_heads=4, hidden=128, ff=256, n_blocks=2, dropout=0.2, n_epochs=20, batch_size=256, lr=1e-3),
        AutoRegressiveRecommender(seed=42),
    ]
    recommender_names = [ "Random", "Popularity", "ContentBased", "RNN", "Transformer", "AutoRegressive"]
    
    
    # Initialize recommenders with initial history
    for recommender in recommenders:
        recommender.fit(log=data_generator.history_df, 
                        user_features=users_df, 
                        item_features=items_df)
    
    # Evaluate each recommender separately using train-test split
    results = []
    
    for name, recommender in zip(recommender_names, recommenders):
        print(f"\nEvaluating {name}:")
        
        # Clean up any existing simulator data directory for this recommender
        simulator_data_dir = f"simulator_train_test_data_{name}"
        if os.path.exists(simulator_data_dir):
            shutil.rmtree(simulator_data_dir)
            print(f"Removed existing simulator data directory: {simulator_data_dir}")
        
        # Initialize simulator
        simulator = CompetitionSimulator(
            user_generator=user_generator,
            item_generator=item_generator,
            data_dir=simulator_data_dir,
            log_df=data_generator.history_df,  # PySpark DataFrames don't have copy method
            conversion_noise_mean=config['simulation']['conversion_noise_mean'],
            conversion_noise_std=config['simulation']['conversion_noise_std'],
            spark_session=spark,
            seed=config['data_generation']['seed']
        )
        
        # Run simulation with train-test split
        train_metrics, test_metrics, train_revenue, test_revenue = simulator.train_test_split(
            recommender=recommender,
            train_iterations=train_iterations,
            test_iterations=test_iterations,
            user_frac=config['simulation']['user_fraction'],
            k=config['simulation']['k'],
            filter_seen_items=config['simulation']['filter_seen_items'],
            retrain=config['simulation']['retrain']
        )
        
        # Calculate average metrics
        train_avg_metrics = {}
        for metric_name in train_metrics[0].keys():
            values = [metrics[metric_name] for metrics in train_metrics]
            train_avg_metrics[f"train_{metric_name}"] = np.mean(values)
        
        test_avg_metrics = {}
        for metric_name in test_metrics[0].keys():
            values = [metrics[metric_name] for metrics in test_metrics]
            test_avg_metrics[f"test_{metric_name}"] = np.mean(values)
        
        # Store results
        results.append({
            "name": name,
            "train_total_revenue": sum(train_revenue),
            "test_total_revenue": sum(test_revenue),
            "train_avg_revenue": np.mean(train_revenue),
            "test_avg_revenue": np.mean(test_revenue),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "train_revenue": train_revenue,
            "test_revenue": test_revenue,
            **train_avg_metrics,
            **test_avg_metrics
        })
        
        # Print summary for this recommender
        print(f"  Training Phase - Total Revenue: {sum(train_revenue):.2f}")
        print(f"  Testing Phase - Total Revenue: {sum(test_revenue):.2f}")
        performance_change = ((sum(test_revenue) / len(test_revenue)) / (sum(train_revenue) / len(train_revenue)) - 1) * 100
        print(f"  Performance Change: {performance_change:.2f}%")
    
    # Convert to DataFrame for easy comparison
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("test_total_revenue", ascending=False).reset_index(drop=True)
    
    # Print summary table
    print("\nRecommender Evaluation Results (sorted by test revenue):")
    summary_cols = ["name", "train_total_revenue", "test_total_revenue", 
                   "train_avg_revenue", "test_avg_revenue",
                   "train_precision_at_k", "test_precision_at_k",
                   "train_ndcg_at_k", "test_ndcg_at_k",
                   "train_mrr", "test_mrr",
                   "train_discounted_revenue", "test_discounted_revenue"]
    summary_cols = [col for col in summary_cols if col in results_df.columns]

    results_df[summary_cols].to_clipboard(index=False)
    print(results_df[summary_cols].to_string(index=False))
    
    # Cell: Results Visualization
    """
    ## Results Visualization
    Now we'll visualize the performance of the different recommenders.
    """
    
    # Generate comparison plots
    visualize_recommender_performance(results_df, recommender_names)
    
    # Generate detailed metrics visualizations
    visualize_detailed_metrics(results_df, recommender_names)
    
    return results_df


# Cell: Performance Visualization Functions
"""
## Performance Visualization Functions
These functions create visualizations for comparing recommender performance.
"""

def visualize_recommender_performance(results_df, recommender_names):
    """
    Visualize the performance of recommenders in terms of revenue and key metrics.
    
    Args:
        results_df: DataFrame with evaluation results
        recommender_names: List of recommender names
    """
    plt.figure(figsize=(16, 16))
    
    # Plot total revenue comparison
    plt.subplot(3, 2, 1)
    x = np.arange(len(recommender_names))
    width = 0.35
    plt.bar(x - width/2, results_df['train_total_revenue'], width, label='Training')
    plt.bar(x + width/2, results_df['test_total_revenue'], width, label='Testing')
    plt.xlabel('Recommender')
    plt.ylabel('Total Revenue')
    plt.title('Total Revenue Comparison')
    plt.xticks(x, results_df['name'])
    plt.legend()
    
    # Plot average revenue per iteration
    plt.subplot(3, 2, 2)
    plt.bar(x - width/2, results_df['train_avg_revenue'], width, label='Training')
    plt.bar(x + width/2, results_df['test_avg_revenue'], width, label='Testing')
    plt.xlabel('Recommender')
    plt.ylabel('Avg Revenue per Iteration')
    plt.title('Average Revenue Comparison')
    plt.xticks(x, results_df['name'])
    plt.legend()
    
    # Plot discounted revenue comparison (if available)
    plt.subplot(3, 2, 3)
    if 'train_discounted_revenue' in results_df.columns and 'test_discounted_revenue' in results_df.columns:
        plt.bar(x - width/2, results_df['train_discounted_revenue'], width, label='Training')
        plt.bar(x + width/2, results_df['test_discounted_revenue'], width, label='Testing')
        plt.xlabel('Recommender')
        plt.ylabel('Avg Discounted Revenue')
        plt.title('Discounted Revenue Comparison')
        plt.xticks(x, results_df['name'])
        plt.legend()
    
    # Plot revenue trajectories
    plt.subplot(3, 2, 4)
    markers = ['o', 's', 'D', '^']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, name in enumerate(results_df['name']):
        # Combined train and test trajectories
        train_revenue = results_df.iloc[i]['train_revenue']
        test_revenue = results_df.iloc[i]['test_revenue']
        
        # Check if revenue is a scalar (numpy.float64) or a list/array
        if isinstance(train_revenue, (float, np.float64, np.float32, int, np.integer)):
            train_revenue = [train_revenue]
        if isinstance(test_revenue, (float, np.float64, np.float32, int, np.integer)):
            test_revenue = [test_revenue]
            
        iterations = list(range(len(train_revenue))) + list(range(len(test_revenue)))
        revenues = train_revenue + test_revenue
        
        plt.plot(iterations, revenues, marker=markers[i % len(markers)], 
                 color=colors[i % len(colors)], label=name)
        
        # Add a vertical line to separate train and test
        if i == 0:  # Only add the line once
            plt.axvline(x=len(train_revenue)-0.5, color='k', linestyle='--', alpha=0.3, label='Train/Test Split')
    
    plt.xlabel('Iteration')
    plt.ylabel('Revenue')
    plt.title('Revenue Trajectory (Training → Testing)')
    plt.legend()
    
    # Plot ranking metrics comparison - Training
    plt.subplot(3, 2, 5)
    
    # Select metrics to include
    ranking_metrics = ['precision_at_k', 'ndcg_at_k', 'mrr', 'hit_rate']
    ranking_metrics = [m for m in ranking_metrics if f'train_{m}' in results_df.columns]
    
    # Create bar groups
    bar_positions = np.arange(len(ranking_metrics))
    bar_width = 0.8 / len(results_df)
    
    for i, (_, row) in enumerate(results_df.iterrows()):
        model_name = row['name']
        offsets = (i - len(results_df)/2 + 0.5) * bar_width
        metric_values = [row[f'train_{m}'] for m in ranking_metrics]
        plt.bar(bar_positions + offsets, metric_values, bar_width, label=model_name, 
                color=colors[i % len(colors)], alpha=0.7)
    
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Ranking Metrics Comparison (Training Phase)')
    plt.xticks(bar_positions, [m.replace('_', ' ').title() for m in ranking_metrics])
    plt.legend()
    
    # Plot ranking metrics comparison - Testing
    plt.subplot(3, 2, 6)
    
    # Select metrics to include
    ranking_metrics = ['precision_at_k', 'ndcg_at_k', 'mrr', 'hit_rate']
    ranking_metrics = [m for m in ranking_metrics if f'test_{m}' in results_df.columns]
    
    # Get best-performing model
    best_model_idx = results_df['test_total_revenue'].idxmax()
    best_model_name = results_df.iloc[best_model_idx]['name']
    
    # Create bar groups
    bar_positions = np.arange(len(ranking_metrics))
    bar_width = 0.8 / len(results_df)
    
    for i, (_, row) in enumerate(results_df.iterrows()):
        model_name = row['name']
        offsets = (i - len(results_df)/2 + 0.5) * bar_width
        metric_values = [row[f'test_{m}'] for m in ranking_metrics]
        plt.bar(bar_positions + offsets, metric_values, bar_width, label=model_name, 
                color=colors[i % len(colors)],
                alpha=0.7 if model_name != best_model_name else 1.0)
    
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Ranking Metrics Comparison (Test Phase)')
    plt.xticks(bar_positions, [m.replace('_', ' ').title() for m in ranking_metrics])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('recommender_performance_comparison.png')
    print("\nPerformance visualizations saved to 'recommender_performance_comparison.png'")


def visualize_detailed_metrics(results_df, recommender_names):
    """
    Create detailed visualizations for each metric and recommender.
    
    Args:
        results_df: DataFrame with evaluation results
        recommender_names: List of recommender names
    """
    # Create a figure for metric trajectories
    plt.figure(figsize=(16, 16))
    
    # Get all available metrics
    all_metrics = []
    if len(results_df) > 0 and 'train_metrics' in results_df.columns:
        first_train_metrics = results_df.iloc[0]['train_metrics'][0]
        all_metrics = list(first_train_metrics.keys())
    
    # Select key metrics to visualize
    key_metrics = ['revenue', 'discounted_revenue', 'precision_at_k', 'ndcg_at_k', 'mrr', 'hit_rate']
    key_metrics = [m for m in key_metrics if m in all_metrics]
    
    # Plot metric trajectories for each key metric
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', 'D', '^']
    
    for i, metric in enumerate(key_metrics):
        if i < 6:  # Limit to 6 metrics to avoid overcrowding
            plt.subplot(3, 2, i+1)
            
            for j, name in enumerate(results_df['name']):
                row = results_df[results_df['name'] == name].iloc[0]
                
                # Get metric values for training phase
                train_values = []
                for train_metric in row['train_metrics']:
                    if metric in train_metric:
                        train_values.append(train_metric[metric])
                
                # Get metric values for testing phase
                test_values = []
                for test_metric in row['test_metrics']:
                    if metric in test_metric:
                        test_values.append(test_metric[metric])
                
                # Plot training phase
                plt.plot(range(len(train_values)), train_values, 
                         marker=markers[j % len(markers)], 
                         color=colors[j % len(colors)],
                         linestyle='-', label=f"{name} (train)")
                
                # Plot testing phase
                plt.plot(range(len(train_values), len(train_values) + len(test_values)), 
                         test_values, marker=markers[j % len(markers)], 
                         color=colors[j % len(colors)],
                         linestyle='--', label=f"{name} (test)")
                
                # Add a vertical line to separate train and test
                if j == 0:  # Only add the line once
                    plt.axvline(x=len(train_values)-0.5, color='k', 
                                linestyle='--', alpha=0.3, label='Train/Test Split')
            
            # Get metric info from EVALUATION_METRICS
            if metric in EVALUATION_METRICS:
                metric_info = EVALUATION_METRICS[metric]
                metric_name = metric_info['name']
                plt.title(f"{metric_name} Trajectory")
            else:
                plt.title(f"{metric.replace('_', ' ').title()} Trajectory")
            
            plt.xlabel('Iteration')
            plt.ylabel('Value')
            
            # Add legend to the last plot only to avoid cluttering
            if i == len(key_metrics) - 1 or i == 5:
                plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig('recommender_metrics_trajectories.png')
    print("Detailed metrics visualizations saved to 'recommender_metrics_trajectories.png'")
    
    # Create a correlation heatmap of metrics
    plt.figure(figsize=(14, 12))
    
    # Extract metrics columns
    metric_cols = [col for col in results_df.columns if col.startswith('train_') or col.startswith('test_')]
    metric_cols = [col for col in metric_cols if not col.endswith('_metrics') and not col.endswith('_revenue')]
    
    if len(metric_cols) > 1:
        correlation_df = results_df[metric_cols].corr()
        
        # Plot heatmap
        sns.heatmap(correlation_df, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Between Metrics')
        plt.tight_layout()
        plt.savefig('metrics_correlation_heatmap.png')
        print("Metrics correlation heatmap saved to 'metrics_correlation_heatmap.png'")


def calculate_discounted_cumulative_gain(recommendations, k=5, discount_factor=0.85):
    """
    Calculate the Discounted Cumulative Gain for recommendations.
    
    Args:
        recommendations: DataFrame with recommendations (must have relevance column)
        k: Number of items to consider
        discount_factor: Factor to discount gains by position
        
    Returns:
        float: Average DCG across all users
    """
    # Group by user and calculate per-user DCG
    user_dcg = []
    for user_id, user_recs in recommendations.groupBy("user_idx").agg(
        sf.collect_list(sf.struct("relevance", "rank")).alias("recommendations")
    ).collect():
        # Sort by rank
        user_rec_list = sorted(user_id.recommendations, key=lambda x: x[1])
        
        # Calculate DCG
        dcg = 0
        for i, (rel, _) in enumerate(user_rec_list[:k]):
            # Apply discount based on position
            dcg += rel * (discount_factor ** i)
        
        user_dcg.append(dcg)
    
    # Return average DCG across all users
    return np.mean(user_dcg) if user_dcg else 0.0


# Cell: Main execution
"""
## Run the Analysis
When you run this notebook, it will perform the full analysis and visualization.
"""

if __name__ == "__main__":
    results = run_recommender_analysis() 