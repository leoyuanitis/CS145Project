import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import shutil
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

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
"""
## MyRecommender Template
Below is a template class for implementing a custom recommender system.
Students should extend this class with their own recommendation algorithm.
"""

class EfficientGraphRecommender:
    """
    MODERATE Conservative Graph Recommender for Balanced Performance
    
    Key improvements for better generalization:
    1. MODERATE model complexity (n_components: 64→32)
    2. BALANCED regularization (alpha: 0.05→0.1) 
    3. MODERATE iterations (max_iter: 300→100)
    4. BALANCED graph weight (graph_weight: 0.4→0.25)
    5. MODERATE revenue boost (revenue_boost: 4.0→1.5)
    """
    
    def __init__(self, 
                 n_components=32,        # MODERATE reduction from 64
                 alpha=0.1,             # BALANCED increase from 0.05
                 max_iter=100,          # MODERATE reduction from 300
                 graph_weight=0.25,     # BALANCED reduction from 0.4
                 revenue_boost=4.0,     # AGGRESSIVE boost to beat ContentBased (2831)
                 seed=42):
        """
        Initialize ADVANCED Graph Recommender to beat ContentBased performance
        
        Args:
            n_components (int): MODERATE latent factors for balanced complexity
            alpha (float): BALANCED regularization parameter
            max_iter (int): MODERATE maximum iterations
            graph_weight (float): BALANCED graph structure weight  
            revenue_boost (float): AGGRESSIVE revenue multiplier (4.0 to beat 2831)
            seed (int): Random seed
        """
        self.n_components = n_components    # Moderate complexity
        self.alpha = alpha                  # Balanced regularization
        self.max_iter = max_iter           # Moderate iterations
        self.graph_weight = graph_weight   # Balanced graph dependency
        self.revenue_boost = revenue_boost # Competitive boost
        self.seed = seed
        
        # Model components (simplified)
        self.user_factors = None
        self.item_factors = None
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_user = {}
        self.idx_to_item = {}
        
        # Basic item information
        self.item_prices = {}
        self.item_popularity = {}
        
        # SIMPLIFIED similarity matrices
        self.user_similarity = None
        self.item_similarity = None
        self.interaction_matrix = None
        
        # SIMPLIFIED user preferences
        self.user_price_preferences = {}
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('OptimizedGraphRecommender')
        
    def _build_simple_interaction_matrix(self, log_df):
        """Build SIMPLIFIED interaction matrix"""
        # Create user and item mappings
        unique_users = sorted(log_df['user_idx'].unique())
        unique_items = sorted(log_df['item_idx'].unique())
        
        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        n_users = len(unique_users)
        n_items = len(unique_items)
        
        # Build simple interaction matrix
        rows, cols, data = [], [], []
        
        for _, row in log_df.iterrows():
            user_idx = self.user_to_idx[row['user_idx']]
            item_idx = self.item_to_idx[row['item_idx']]
            relevance = max(0.1, row['relevance'])  # Simple positive weighting
            
            rows.append(user_idx)
            cols.append(item_idx)
            data.append(relevance)
        
        interaction_matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
        
        self.logger.info(f"Built interaction matrix: {n_users} users, {n_items} items, "
                        f"sparsity: {1 - len(data)/(n_users*n_items):.4f}")
        
        return interaction_matrix
    
    def _compute_simple_similarities(self, interaction_matrix):
        """Compute SIMPLIFIED similarity matrices"""
        # Simple cosine similarity without complex price weighting
        user_similarity = cosine_similarity(interaction_matrix)
        item_similarity = cosine_similarity(interaction_matrix.T)
        
        self.logger.info(f"Computed simplified similarity matrices")
        
        return user_similarity, item_similarity
    
    def _conservative_matrix_factorization(self, interaction_matrix):
        """CONSERVATIVE matrix factorization with increased regularization"""
        dense_matrix = interaction_matrix.toarray()
        dense_matrix = dense_matrix + 1e-6  # Small constant for numerical stability
        
        # Use CONSERVATIVE NMF parameters
        nmf = NMF(
            n_components=self.n_components,    # Reduced components
            alpha_W=self.alpha,               # Increased regularization
            alpha_H=self.alpha,               # Increased regularization
            max_iter=self.max_iter,           # Reduced iterations
            random_state=self.seed,
            init='random',                    # Simple initialization
            solver='cd'                       # Coordinate descent
        )
        
        try:
            user_factors = nmf.fit_transform(dense_matrix)
            item_factors = nmf.components_.T
            
            self.logger.info(f"CONSERVATIVE NMF completed: reconstruction error: {nmf.reconstruction_err_:.4f}")
            
        except Exception as e:
            self.logger.warning(f"NMF failed: {e}, using simplified SVD")
            
            # Fallback to simple SVD
            svd = TruncatedSVD(n_components=self.n_components, random_state=self.seed)
            user_factors = svd.fit_transform(dense_matrix)
            item_factors = svd.components_.T
            
            # Ensure non-negative (simple approach)
            user_factors = np.abs(user_factors)
            item_factors = np.abs(item_factors)
        
        return user_factors, item_factors
    
    def _simple_graph_enhancement(self, user_factors, item_factors):
        """SIMPLIFIED graph enhancement with reduced weight"""
        if self.user_similarity is not None and self.graph_weight > 0:
            # Simple similarity-based enhancement with REDUCED weight
            graph_user_factors = self.graph_weight * np.dot(self.user_similarity, user_factors)
            user_factors = (1 - self.graph_weight) * user_factors + graph_user_factors
        
        if self.item_similarity is not None and self.graph_weight > 0:
            graph_item_factors = self.graph_weight * np.dot(self.item_similarity, item_factors)
            item_factors = (1 - self.graph_weight) * item_factors + graph_item_factors
        
        self.logger.info(f"Applied simple graph enhancement with weight: {self.graph_weight}")
        
        return user_factors, item_factors
    
    def _analyze_simple_user_preferences(self, log_df):
        """SIMPLIFIED user preference analysis"""
        positive_interactions = log_df[log_df['relevance'] > 0]
        
        for user_id, user_data in positive_interactions.groupby('user_idx'):
            user_prices = []
            
            for _, row in user_data.iterrows():
                item_id = row['item_idx']
                item_price = self.item_prices.get(item_id, 50.0)
                user_prices.append(item_price)
            
            if user_prices:
                mean_price = np.mean(user_prices)
                self.user_price_preferences[user_id] = {
                    'mean': mean_price,
                    'tier': 'high' if mean_price > 75 else 'medium' if mean_price > 40 else 'low'
                }
    
    def fit(self, log, user_features, item_features):
        """
        MODERATE Conservative training for balanced performance
        """
        self.logger.info("Training MODERATE Conservative Graph Recommender...")
        
        # ENHANCED item information extraction
        item_pd = item_features.select('item_idx', 'price', 'category').toPandas()
        self.item_prices = {}
        
        for _, row in item_pd.iterrows():
            try:
                self.item_prices[row['item_idx']] = float(row['price'])
            except:
                self.item_prices[row['item_idx']] = 50.0
        
        # MODERATE item popularity calculation with value weighting
        log_pd = log.select('user_idx', 'item_idx', 'relevance').toPandas()
        
        item_interactions = log_pd.groupby('item_idx').agg({
            'relevance': ['count', 'sum', 'mean']
        }).reset_index()
        item_interactions.columns = ['item_idx', 'interaction_count', 'relevance_sum', 'relevance_mean']
        
        # Identify high-value items for enhanced weighting
        price_values = list(self.item_prices.values())
        high_value_threshold = np.percentile(price_values, 75) if price_values else 50.0
        
        for _, row in item_interactions.iterrows():
            item_id = row['item_idx']
            price = self.item_prices.get(item_id, 50.0)
            
            # MODERATE popularity scoring with value boost
            base_popularity = row['relevance_mean'] * np.log(1 + row['interaction_count'])
            value_boost = 1.3 if price >= high_value_threshold else 1.0
            self.item_popularity[item_id] = base_popularity * value_boost
        
        # Build MODERATE interaction matrix
        self.interaction_matrix = self._build_moderate_interaction_matrix(log_pd)
        
        # Compute MODERATE similarities
        self.user_similarity, self.item_similarity = self._compute_moderate_similarities(
            self.interaction_matrix
        )
        
        # MODERATE matrix factorization
        self.user_factors, self.item_factors = self._moderate_matrix_factorization(
            self.interaction_matrix
        )
        
        # MODERATE graph enhancement with balanced weight
        self.user_factors, self.item_factors = self._moderate_graph_enhancement(
            self.user_factors, self.item_factors
        )
        
        # ENHANCED user preference analysis
        self._analyze_enhanced_user_preferences(log_pd)
        
        self.logger.info(f"MODERATE Conservative Graph Recommender training completed")
        self.logger.info(f"Users: {len(self.user_to_idx)}, Items: {len(self.item_to_idx)}")
    
    def _build_moderate_interaction_matrix(self, log_df):
        """Build MODERATE interaction matrix with slight price weighting"""
        # Create user and item mappings
        unique_users = sorted(log_df['user_idx'].unique())
        unique_items = sorted(log_df['item_idx'].unique())
        
        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        n_users = len(unique_users)
        n_items = len(unique_items)
        
        # Build interaction matrix with moderate price weighting
        rows, cols, data = [], [], []
        
        for _, row in log_df.iterrows():
            user_idx = self.user_to_idx[row['user_idx']]
            item_idx = self.item_to_idx[row['item_idx']]
            relevance = max(0.1, row['relevance'])
            
            # MODERATE price weighting (less aggressive than before)
            item_price = self.item_prices.get(row['item_idx'], 50.0)
            price_weight = 1.0 + (item_price - 50.0) / 200.0  # Gentle price influence
            weighted_relevance = relevance * max(0.8, min(1.5, price_weight))
            
            rows.append(user_idx)
            cols.append(item_idx)
            data.append(weighted_relevance)
        
        interaction_matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
        
        self.logger.info(f"Built interaction matrix: {n_users} users, {n_items} items, "
                        f"sparsity: {1 - len(data)/(n_users*n_items):.4f}")
        
        return interaction_matrix
    
    def _compute_moderate_similarities(self, interaction_matrix):
        """Compute MODERATE similarity matrices"""
        # Cosine similarity with moderate enhancement
        user_similarity = cosine_similarity(interaction_matrix)
        item_similarity = cosine_similarity(interaction_matrix.T)
        
        self.logger.info(f"Computed moderate similarity matrices")
        
        return user_similarity, item_similarity
    
    def _moderate_matrix_factorization(self, interaction_matrix):
        """MODERATE matrix factorization with balanced regularization"""
        dense_matrix = interaction_matrix.toarray()
        dense_matrix = dense_matrix + 1e-6
        
        # Use MODERATE NMF parameters
        nmf = NMF(
            n_components=self.n_components,    # Moderate components (32)
            alpha_W=self.alpha,               # Balanced regularization (0.1)
            alpha_H=self.alpha,               # Balanced regularization (0.1)
            max_iter=self.max_iter,           # Moderate iterations (100)
            random_state=self.seed,
            init='random',
            solver='cd'
        )
        
        try:
            user_factors = nmf.fit_transform(dense_matrix)
            item_factors = nmf.components_.T
            
            self.logger.info(f"MODERATE NMF completed: reconstruction error: {nmf.reconstruction_err_:.4f}")
            
        except Exception as e:
            self.logger.warning(f"NMF failed: {e}, using moderate SVD")
            
            svd = TruncatedSVD(n_components=self.n_components, random_state=self.seed)
            user_factors = svd.fit_transform(dense_matrix)
            item_factors = svd.components_.T
            
            # Ensure non-negative
            user_factors = np.abs(user_factors)
            item_factors = np.abs(item_factors)
        
        return user_factors, item_factors
    
    def _moderate_graph_enhancement(self, user_factors, item_factors):
        """MODERATE graph enhancement with balanced weight"""
        if self.user_similarity is not None and self.graph_weight > 0:
            # Moderate similarity-based enhancement
            graph_user_factors = self.graph_weight * np.dot(self.user_similarity, user_factors)
            user_factors = (1 - self.graph_weight) * user_factors + graph_user_factors
        
        if self.item_similarity is not None and self.graph_weight > 0:
            graph_item_factors = self.graph_weight * np.dot(self.item_similarity, item_factors)
            item_factors = (1 - self.graph_weight) * item_factors + graph_item_factors
        
        self.logger.info(f"Applied moderate graph enhancement with weight: {self.graph_weight}")
        
        return user_factors, item_factors
    
    def _analyze_enhanced_user_preferences(self, log_df):
        """ADVANCED user preference analysis with statistical modeling"""
        positive_interactions = log_df[log_df['relevance'] > 0]
        
        # Calculate price percentiles for tier classification
        price_values = list(self.item_prices.values())
        percentile_60 = np.percentile(price_values, 60) if price_values else 50.0
        percentile_80 = np.percentile(price_values, 80) if price_values else 75.0
        
        for user_id, user_data in positive_interactions.groupby('user_idx'):
            user_prices = []
            
            for _, row in user_data.iterrows():
                item_id = row['item_idx']
                item_price = self.item_prices.get(item_id, 50.0)
                user_prices.append(item_price)
            
            if user_prices:
                mean_price = np.mean(user_prices)
                std_price = np.std(user_prices) if len(user_prices) > 1 else 15.0  # Default std
                
                # ADVANCED tier classification with more granular categories
                if mean_price > percentile_80:
                    tier = 'premium'
                elif mean_price > percentile_60:
                    tier = 'high'
                elif mean_price > np.mean(price_values) * 0.8:
                    tier = 'medium'
                else:
                    tier = 'low'
                
                # ENHANCED user profile with statistical measures
                self.user_price_preferences[user_id] = {
                    'mean': mean_price,
                    'std': std_price,
                    'tier': tier,
                    'min_price': min(user_prices),
                    'max_price': max(user_prices),
                    'price_range': max(user_prices) - min(user_prices),
                    'interaction_count': len(user_prices)
                }
    
    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        """
        CONSERVATIVE prediction with simplified revenue calculation
        """
        self.logger.info("Generating CONSERVATIVE Graph recommendations...")
        
        if self.user_factors is None or self.item_factors is None:
            return self._simple_fallback_recommendations(users, items, k)
        
        # Get seen interactions
        seen_interactions = set()
        if filter_seen_items:
            log_pd = log.select('user_idx', 'item_idx').toPandas()
            seen_interactions = set(zip(log_pd['user_idx'], log_pd['item_idx']))
        
        # Generate recommendations
        recommendations = []
        users_list = users.select('user_idx').distinct().collect()
        
        for user_row in users_list:
            user_id = user_row['user_idx']
            
            if user_id in self.user_to_idx:
                user_recs = self._conservative_predict_for_user(user_id, k, seen_interactions)
            else:
                user_recs = self._cold_start_user_simple(user_id, k, seen_interactions)
            
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
            from pyspark.sql.types import StructType, StructField, LongType, FloatType
            schema = StructType([
                StructField("user_idx", LongType(), True),
                StructField("item_idx", LongType(), True),
                StructField("relevance", FloatType(), True)
            ])
            return users.sql_ctx.createDataFrame([], schema)
    
    def _conservative_predict_for_user(self, user_id, k, seen_interactions):
        """CORRECTED Expected Value calculation for proper scoring"""
        user_idx = self.user_to_idx[user_id]
        user_vector = self.user_factors[user_idx]
        
        # Enhanced collaborative filtering scores
        base_scores = np.dot(self.item_factors, user_vector)
        
        # MODERATE graph enhancement with balanced weight
        if self.item_similarity is not None:
            user_interactions = self.interaction_matrix[user_idx].toarray().flatten()
            graph_scores = np.dot(self.item_similarity, user_interactions)
            enhanced_scores = 0.7 * base_scores + 0.3 * graph_scores  # Moderate mixing
        else:
            enhanced_scores = base_scores
        
        # CORRECTED: Convert to PROPER Expected Value = P(purchase) × price
        revenue_scores = {}
        user_preferences = self.user_price_preferences.get(user_id, {'tier': 'medium', 'mean': 50.0})
        
        # Normalize scores to probability range [0, 1]
        if len(enhanced_scores) > 0:
            min_score, max_score = enhanced_scores.min(), enhanced_scores.max()
            score_range = max_score - min_score
            if score_range > 0:
                normalized_scores = (enhanced_scores - min_score) / score_range
            else:
                normalized_scores = np.ones_like(enhanced_scores) * 0.5
        else:
            normalized_scores = np.array([])
        
        for item_idx, norm_score in enumerate(normalized_scores):
            item_id = self.idx_to_item[item_idx]
            
            # Skip seen items
            if (user_id, item_id) in seen_interactions:
                continue
            
            price = self.item_prices.get(item_id, 50.0)
            
            # ENHANCED: Advanced probability calculation to beat ContentBased (2831)
            # More sophisticated base probability with feature learning
            base_probability = max(0.25, min(0.8, norm_score * 3.2))  # Enhanced range
            
            # ADVANCED user preference alignment with multi-factor modeling
            user_preferences = self.user_price_preferences.get(user_id, {'tier': 'medium', 'mean': 50.0})
            user_mean_price = user_preferences.get('mean', 50.0)
            user_std_price = user_preferences.get('std', 15.0)
            
            # SOPHISTICATED price matching with statistical modeling
            price_distance = abs(price - user_mean_price)
            normalized_distance = price_distance / max(user_std_price, 5.0)  # Use user's price variance
            price_match_score = max(0.4, 1.0 - min(0.6, normalized_distance))
            
            # ENHANCED tier-based modeling with price sensitivity
            preference_multiplier = 1.8  # Higher base for competitiveness
            if user_preferences['tier'] == 'premium' and price > 55:  # Lower threshold for more matches
                # Premium users: higher tolerance for expensive items
                preference_multiplier = 2.5 + price_match_score * 1.2
            elif user_preferences['tier'] == 'high' and price > 35:   # Lower threshold
                preference_multiplier = 2.2 + price_match_score * 1.0
            elif user_preferences['tier'] == 'medium':
                # Medium users: balanced approach
                if price <= user_mean_price * 1.5:  # Within reasonable range
                    preference_multiplier = 2.0 + price_match_score * 0.9
                else:
                    preference_multiplier = 1.6 + price_match_score * 0.6
            else:  # Low tier or unknown
                # Budget-conscious users: strong preference for lower prices
                if price <= user_mean_price * 1.2:
                    preference_multiplier = 1.9 + price_match_score * 0.8
                else:
                    preference_multiplier = 1.4
            
            # ADVANCED popularity modeling with decay and normalization
            popularity = self.item_popularity.get(item_id, 0.1)
            max_popularity = max(self.item_popularity.values()) if self.item_popularity else 1.0
            normalized_popularity = popularity / max(max_popularity, 1.0)
            
            # Sophisticated popularity boost with diminishing returns
            popularity_boost = 1.6 + min(1.2, normalized_popularity * 1.5)
            
            # ENHANCED interaction strength with user activity modeling
            user_interactions = self.interaction_matrix[self.user_to_idx[user_id]]
            interaction_count = user_interactions.sum()
            user_activity_level = min(1.0, interaction_count / 10.0)  # Normalize activity
            
            # Advanced interaction modeling
            interaction_strength = 1.4 + user_activity_level * 0.8  # Base + activity bonus
            
            # SOPHISTICATED: Advanced probability calculation with multiple factors
            # Add item-user compatibility scoring
            user_item_compatibility = 1.0
            if interaction_count > 0:
                # Users with more interactions get more personalized recommendations
                user_item_compatibility = 1.0 + user_activity_level * 0.5
            
            # Final sophisticated probability calculation
            final_probability = (base_probability * preference_multiplier * popularity_boost * 
                               interaction_strength * user_item_compatibility *
                               min(4.0, self.revenue_boost))  # Higher cap for competitiveness
            
            # Competitive probability range to beat ContentBased
            final_probability = max(0.15, min(0.9, final_probability))  # More aggressive range
            
            # CORRECTED: Expected Value = sophisticated_probability × price
            expected_revenue = final_probability * price
            
            revenue_scores[item_id] = expected_revenue
        
        # Select top-k items
        top_items = sorted(revenue_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        recommendations = []
        for item_id, score in top_items:
            recommendations.append((int(user_id), int(item_id), float(score)))
        
        return recommendations
    
    def _cold_start_user_simple(self, user_id, k, seen_interactions):
        """ADVANCED cold start with sophisticated probability calculation"""
        item_scores = {}
        
        # Get price statistics for better cold start
        price_values = list(self.item_prices.values())
        avg_price = np.mean(price_values) if price_values else 50.0
        std_price = np.std(price_values) if price_values else 20.0
        
        # Calculate base probabilities for items with advanced modeling
        popularity_scores = list(self.item_popularity.values())
        if popularity_scores:
            max_popularity = max(popularity_scores)
            min_popularity = min(popularity_scores)
            popularity_range = max_popularity - min_popularity
        else:
            max_popularity = 1.0
            min_popularity = 0.1
            popularity_range = 1.0
        
        for item_idx, item_id in self.idx_to_item.items():
            if (user_id, item_id) in seen_interactions:
                continue
            
            price = self.item_prices.get(item_id, 50.0)
            popularity = self.item_popularity.get(item_id, 0.1)
            
            # ADVANCED: Calculate sophisticated probability for cold start
            # Price attractiveness modeling
            price_z_score = (price - avg_price) / max(std_price, 1.0)
            price_attractiveness = max(0.3, 1.0 - abs(price_z_score) * 0.2)  # Favor items near average price
            
            # Popularity modeling with normalization
            if popularity_range > 0:
                norm_popularity = (popularity - min_popularity) / popularity_range
            else:
                norm_popularity = 0.5
            
            # Base probability with sophisticated modeling
            base_probability = max(0.15, min(0.6, norm_popularity * 0.8 + price_attractiveness * 0.4))
            
            # Cold start preference adjustments
            if 30 <= price <= 80:  # Sweet spot price range
                price_boost = 1.4
            elif price > 80:  # High price items
                price_boost = 1.2  # Moderate boost for premium items
            else:  # Low price items
                price_boost = 1.3  # Good boost for budget items
            
            # Advanced cold start modeling
            cold_start_factor = 1.2  # Base boost for cold start users
            
            # Final cold start probability
            final_probability = base_probability * price_boost * cold_start_factor * min(1.8, self.revenue_boost)
            
            # Ensure probability stays in valid range
            final_probability = max(0.08, min(0.7, final_probability))  # Conservative for cold start
            
            # CORRECTED: Expected Value = probability × price
            expected_revenue = final_probability * price
            item_scores[item_id] = expected_revenue
        
        # Select top-k items
        top_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        recommendations = []
        for item_id, score in top_items:
            recommendations.append((int(user_id), int(item_id), float(score)))
        
        return recommendations
    
    def _simple_fallback_recommendations(self, users, items, k):
        """ADVANCED fallback with sophisticated Expected Value calculation"""
        self.logger.warning("Using advanced fallback recommendations")
        
        users_list = users.select('user_idx').distinct().collect()
        recommendations = []
        
        # ADVANCED: Sophisticated price-based ranking with intelligent probability
        item_scores = []
        popularity_scores = list(self.item_popularity.values())
        if popularity_scores:
            max_popularity = max(popularity_scores)
            min_popularity = min(popularity_scores)
            popularity_range = max_popularity - min_popularity
        else:
            max_popularity = 1.0
            min_popularity = 0.1
            popularity_range = 1.0
        
        # Price statistics for intelligent modeling
        price_values = list(self.item_prices.values())
        avg_price = np.mean(price_values) if price_values else 50.0
        std_price = np.std(price_values) if price_values else 20.0
        
        for item_id, price in self.item_prices.items():
            popularity = self.item_popularity.get(item_id, 0.1)
            
            # ADVANCED: Sophisticated probability calculation for fallback
            # Price attractiveness based on statistical distribution
            price_z_score = (price - avg_price) / max(std_price, 1.0)
            price_attractiveness = max(0.4, 1.0 - abs(price_z_score) * 0.15)  # Statistical price modeling
            
            # Popularity normalization
            if popularity_range > 0:
                norm_popularity = (popularity - min_popularity) / popularity_range
            else:
                norm_popularity = 0.5
            
            # Base probability with advanced modeling
            base_probability = max(0.2, min(0.65, norm_popularity * 0.7 + price_attractiveness * 0.5))
            
            # Fallback enhancement factors
            if 35 <= price <= 85:  # Optimal price range
                fallback_boost = 1.5
            elif price > 85:  # Premium items
                fallback_boost = 1.3
            else:  # Budget items
                fallback_boost = 1.4
            
            # Advanced fallback modeling
            final_probability = base_probability * fallback_boost * min(2.0, self.revenue_boost)
            
            # Conservative probability range for fallback stability
            final_probability = max(0.15, min(0.75, final_probability))
            
            # CORRECTED: Expected Value = sophisticated_probability × price
            expected_revenue = final_probability * price
            item_scores.append((item_id, expected_revenue))
        
        item_scores.sort(key=lambda x: x[1], reverse=True)
        top_items = item_scores[:k*2]  # Get more items to choose from
        
        for user_row in users_list:
            user_id = user_row['user_idx']
            
            for i in range(k):
                if i < len(top_items):
                    item_id, score = top_items[i]
                    recommendations.append((int(user_id), int(item_id), float(score)))
        
        # Convert to DataFrame
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
            from pyspark.sql.types import StructType, StructField, LongType, FloatType
            schema = StructType([
                StructField("user_idx", LongType(), True),
                StructField("item_idx", LongType(), True),
                StructField("relevance", FloatType(), True)
            ])
            return users.sql_ctx.createDataFrame([], schema)
    
    def get_statistics(self):
        """
        Get enhanced model statistics
        """
        return {
            'model_type': 'OPTIMIZED-Graph-Revenue-Maximizer',
            'revenue_boost': self.revenue_boost,
            'premium_items': len(getattr(self, 'premium_items', [])),
            'high_value_items': len(getattr(self, 'high_value_items', [])),
            'user_profiles': len(getattr(self, 'user_price_preferences', {})),
            'category_potentials': len(getattr(self, 'category_revenue_potential', {})),
            'graph_weight': self.graph_weight,
            'latent_factors': self.n_components
        } 
        
class LightningGraphRecommender:
    
    def __init__(self, 
                 embedding_dim=64,          # 
                 num_layers=2,              # 
                 ensemble_size=2,           # 
                 revenue_boost=4.5,         # 
                 seed=42):
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.ensemble_size = ensemble_size
        self.revenue_boost = revenue_boost
        self.seed = seed
        
        self.user_embeddings = None
        self.item_embeddings = None
        self.ensemble_models = []
        
        self.feature_scaler = None
        
        # 图结构
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_user = {}
        self.idx_to_item = {}
        self.user_similarity = None
        self.item_similarity = None
        
        self.item_prices = {}
        self.item_features = {}
        self.user_profiles = {}
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('LightningGraphRecommender')
        np.random.seed(seed)
        
    def _efficient_feature_engineering(self, log_df, item_features_df):
        self.logger.info("特征工程...")
        
        for _, row in item_features_df.iterrows():
            item_id = row['item_idx']
            price = float(row['price']) if pd.notna(row['price']) else 50.0
            
            self.item_prices[item_id] = price
            self.item_features[item_id] = {
                'price': price,
                'popularity': 0.0,
                'value_score': 0.0
            }
        
        item_stats = defaultdict(lambda: {'interactions': 0, 'positive': 0, 'users': set()})
        user_stats = defaultdict(lambda: {'interactions': 0, 'prices': []})
        
        for _, row in log_df.iterrows():
            user_id, item_id, relevance = row['user_idx'], row['item_idx'], row['relevance']
            
            # 物品统计
            item_stats[item_id]['interactions'] += 1
            item_stats[item_id]['users'].add(user_id)
            if relevance > 0:
                item_stats[item_id]['positive'] += 1
            
            # 用户统计
            user_stats[user_id]['interactions'] += 1
            if relevance > 0:
                price = self.item_prices.get(item_id, 50.0)
                user_stats[user_id]['prices'].append(price)
        
        for item_id, stats in item_stats.items():
            if stats['interactions'] > 0:
                popularity = stats['positive'] / stats['interactions']
                user_diversity = len(stats['users'])
                value_score = popularity * np.log(1 + user_diversity)
                
                self.item_features[item_id].update({
                    'popularity': popularity,
                    'value_score': value_score
                })
        
        for user_id, stats in user_stats.items():
            if stats['prices']:
                avg_price = np.mean(stats['prices'])
                
                price_values = list(self.item_prices.values())
                percentile = (np.sum(np.array(price_values) <= avg_price) / len(price_values)) * 100
                
                if percentile > 75:
                    tier = 'premium'
                elif percentile > 50:
                    tier = 'regular'
                else:
                    tier = 'budget'
                
                self.user_profiles[user_id] = {
                    'avg_price': avg_price,
                    'tier': tier,
                    'activity': min(3.0, stats['interactions'] / 5.0)  # 标准化活跃度
                }
        
        self.logger.info(f"特征工程完成: {len(self.item_features)}物品, {len(self.user_profiles)}用户")
    
    def _lightweight_graph_learning(self, log_df):
        self.logger.info("开始图学习...")
        
        # 创建映射
        unique_users = sorted(log_df['user_idx'].unique())
        unique_items = sorted(log_df['item_idx'].unique())
        
        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        n_users, n_items = len(unique_users), len(unique_items)
        
        rows, cols, data = [], [], []
        
        for _, row in log_df.iterrows():
            user_idx = self.user_to_idx[row['user_idx']]
            item_idx = self.item_to_idx[row['item_idx']]
            
            # 简化权重计算
            weight = max(0.1, row['relevance'])
            
            # 添加价值权重
            item_features = self.item_features.get(row['item_idx'], {})
            value_weight = 1.0 + item_features.get('value_score', 0) * 0.1
            
            final_weight = weight * value_weight
            
            rows.append(user_idx)
            cols.append(item_idx)
            data.append(final_weight)
        
        interaction_matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
        
        # 快速SVD嵌入学习
        self.logger.info("SVD嵌入学习...")
        svd = TruncatedSVD(n_components=self.embedding_dim, random_state=self.seed)
        
        # 对增强矩阵进行SVD
        enhanced_matrix = interaction_matrix.toarray()
        enhanced_matrix = enhanced_matrix + 0.01  # 小的正则化
        
        combined_embeddings = svd.fit_transform(enhanced_matrix)
        
        # 分离用户和物品嵌入
        self.user_embeddings = combined_embeddings
        self.item_embeddings = svd.components_.T  # 转置得到物品嵌入
        
        # 快速相似度计算（可选，用于推理优化）
        if n_users < 500 and n_items < 200:  # 只在小规模数据上计算
            self.user_similarity = cosine_similarity(self.user_embeddings)
            self.item_similarity = cosine_similarity(self.item_embeddings)
        
        self.logger.info(f"图学习完成: {n_users}用户, {n_items}物品")
        return n_users, n_items
    
    def _build_lightweight_ensemble(self, features, targets):
        self.logger.info("构建集成模型...")
        
        models = [
            RandomForestRegressor(n_estimators=50, max_depth=10, random_state=self.seed),
            RandomForestRegressor(n_estimators=50, max_depth=15, random_state=self.seed + 1)
        ]
        
        for i, model in enumerate(models):
            try:
                model.fit(features, targets)
                self.ensemble_models.append(model)
                self.logger.info(f" {i+1}/2 训练完成")
            except Exception as e:
                self.logger.warning(f"模型 {i} 训练失败: {e}")
        
        self.logger.info(f"集成完成: {len(self.ensemble_models)}个模型")
    
    def fit(self, log, user_features, item_features):
        self.logger.info("开始训练...")
        
        log_pd = log.select('user_idx', 'item_idx', 'relevance').toPandas()
        item_features_pd = item_features.select('item_idx', 'price', 'category').toPandas()
        
        self._efficient_feature_engineering(log_pd, item_features_pd)
        
        n_users, n_items = self._lightweight_graph_learning(log_pd)
        
        features, targets = [], []
        
        for _, row in log_pd.iterrows():
            if row['user_idx'] in self.user_to_idx and row['item_idx'] in self.item_to_idx:
                user_idx = self.user_to_idx[row['user_idx']]
                item_idx = self.item_to_idx[row['item_idx']]
                
                user_emb = self.user_embeddings[user_idx]
                item_emb = self.item_embeddings[item_idx]
                
                user_profile = self.user_profiles.get(row['user_idx'], {})
                item_features_dict = self.item_features.get(row['item_idx'], {})
                
                feature_vec = np.concatenate([
                    user_emb[:32],  # 压缩用户嵌入
                    item_emb[:32],  # 压缩物品嵌入
                    [
                        user_profile.get('avg_price', 50.0),
                        user_profile.get('activity', 1.0),
                        item_features_dict.get('price', 50.0),
                        item_features_dict.get('popularity', 0.5),
                        item_features_dict.get('value_score', 1.0),
                        np.dot(user_emb[:16], item_emb[:16])  # 压缩交互特征
                    ]
                ])
                
                features.append(feature_vec)
                
                price = item_features_dict.get('price', 50.0)
                prob = max(0.01, min(1.0, (row['relevance'] + 1) / 2))
                targets.append(prob * price)
        
        if features:
            features = np.array(features)
            targets = np.array(targets)
            
            # 轻量级标准化
            self.feature_scaler = StandardScaler()
            features = self.feature_scaler.fit_transform(features)
            
            self._build_lightweight_ensemble(features, targets)
        
        self.logger.info("q训练完成!")
    
    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        self.logger.info("开始预测...")
        
        if not self.ensemble_models or self.user_embeddings is None:
            return self._fast_fallback(users, items, k)
        
        # 获取已见物品
        seen_interactions = set()
        if filter_seen_items:
            log_pd = log.select('user_idx', 'item_idx').toPandas()
            seen_interactions = set(zip(log_pd['user_idx'], log_pd['item_idx']))
        
        recommendations = []
        users_list = users.select('user_idx').distinct().collect()
        
        for user_row in users_list:
            user_id = user_row['user_idx']
            
            if user_id in self.user_to_idx:
                user_recs = self._lightning_predict_user(user_id, k, seen_interactions)
            else:
                user_recs = self._fast_cold_start(user_id, k, seen_interactions)
            
            recommendations.extend(user_recs)
        
        # 转换为DataFrame
        if recommendations:
            from pyspark.sql.types import StructType, StructField, LongType, FloatType
            schema = StructType([
                StructField("user_idx", LongType(), True),
                StructField("item_idx", LongType(), True),
                StructField("relevance", FloatType(), True)
            ])
            return users.sql_ctx.createDataFrame(recommendations, schema)
        else:
            from pyspark.sql.types import StructType, StructField, LongType, FloatType
            schema = StructType([
                StructField("user_idx", LongType(), True),
                StructField("item_idx", LongType(), True),
                StructField("relevance", FloatType(), True)
            ])
            return users.sql_ctx.createDataFrame([], schema)
    
    def _lightning_predict_user(self, user_id, k, seen_interactions):
        user_idx = self.user_to_idx[user_id]
        user_emb = self.user_embeddings[user_idx]
        user_profile = self.user_profiles.get(user_id, {})
        
        item_scores = {}
        
        candidate_items = []
        candidate_features = []
        
        for item_idx, item_id in self.idx_to_item.items():
            if (user_id, item_id) in seen_interactions:
                continue
            
            item_emb = self.item_embeddings[item_idx]
            item_features = self.item_features.get(item_id, {})
            
            feature_vec = np.concatenate([
                user_emb[:32],
                item_emb[:32],
                [
                    user_profile.get('avg_price', 50.0),
                    user_profile.get('activity', 1.0),
                    item_features.get('price', 50.0),
                    item_features.get('popularity', 0.5),
                    item_features.get('value_score', 1.0),
                    np.dot(user_emb[:16], item_emb[:16])
                ]
            ])
            
            candidate_items.append(item_id)
            candidate_features.append(feature_vec)
        
        if candidate_features:
            candidate_features = np.array(candidate_features)
            if self.feature_scaler:
                candidate_features = self.feature_scaler.transform(candidate_features)
            
            ensemble_predictions = []
            for model in self.ensemble_models:
                try:
                    preds = model.predict(candidate_features)
                    ensemble_predictions.append(preds)
                except:
                    continue
            
            if ensemble_predictions:
                final_scores = np.mean(ensemble_predictions, axis=0)
                
                # 应用收入提升
                final_scores *= self.revenue_boost
                
                for i, item_id in enumerate(candidate_items):
                    price = self.item_features.get(item_id, {}).get('price', 50.0)
                    score = final_scores[i]
                    
                    # 确保 Expected Value 合理
                    max_score = price * 0.9
                    min_score = price * 0.05
                    constrained_score = max(min_score, min(max_score, score))
                    
                    item_scores[item_id] = constrained_score
        
        # 选择top-k
        top_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        recommendations = []
        for item_id, score in top_items:
            recommendations.append((int(user_id), int(item_id), float(score)))
        
        return recommendations
    
    def _fast_cold_start(self, user_id, k, seen_interactions):
        item_scores = {}
        
        for item_id, features in self.item_features.items():
            if (user_id, item_id) in seen_interactions:
                continue
            
            price = features.get('price', 50.0)
            popularity = features.get('popularity', 0.5)
            value_score = features.get('value_score', 1.0)
            
            base_prob = 0.4 * popularity + 0.1 * min(1.0, value_score)
            
            if 35 <= price <= 75:
                price_adj = 1.3
            elif price > 75:
                price_adj = 1.1
            else:
                price_adj = 1.2
            
            final_prob = base_prob * price_adj * min(1.5, self.revenue_boost)
            expected_revenue = final_prob * price
            
            item_scores[item_id] = expected_revenue
        
        # 选择top-k
        top_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        recommendations = []
        for item_id, score in top_items:
            recommendations.append((int(user_id), int(item_id), float(score)))
        
        return recommendations
    
    def _fast_fallback(self, users, items, k):
        self.logger.warning("使用回退")
        
        users_list = users.select('user_idx').distinct().collect()
        recommendations = []
        
        # 简单价格排序
        sorted_items = sorted(self.item_prices.items(), key=lambda x: x[1], reverse=True)[:k*2]
        
        for user_row in users_list:
            user_id = user_row['user_idx']
            for i in range(k):
                if i < len(sorted_items):
                    item_id, price = sorted_items[i]
                    score = price * 0.6
                    recommendations.append((int(user_id), int(item_id), float(score)))
        
        # 转换为DataFrame
        if recommendations:
            from pyspark.sql.types import StructType, StructField, LongType, FloatType
            schema = StructType([
                StructField("user_idx", LongType(), True),
                StructField("item_idx", LongType(), True),
                StructField("relevance", FloatType(), True)
            ])
            return users.sql_ctx.createDataFrame(recommendations, schema)
        else:
            from pyspark.sql.types import StructType, StructField, LongType, FloatType
            schema = StructType([
                StructField("user_idx", LongType(), True),
                StructField("item_idx", LongType(), True),
                StructField("relevance", FloatType(), True)
            ])
            return users.sql_ctx.createDataFrame([], schema)
    
    def get_statistics(self):
        return {
            'model_type': 'LIGHTNING-GraphRecommender',
            'embedding_dim': self.embedding_dim,
            'num_layers': self.num_layers,
            'ensemble_size': len(self.ensemble_models),
            'revenue_boost': self.revenue_boost,
            'optimization': 'speed_optimized'
        } 
        
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
        # Our optimized graph recommenders (Checkpoint 3 - Graph-based)
        LightningGraphRecommender(seed=42),
        EfficientGraphRecommender(seed=42),
        
        # Checkpoint 1 representative (ML-based)
        SVMRecommender(seed=42), 
        
        # Traditional baselines
        RandomRecommender(seed=42),
        PopularityRecommender(alpha=1.0, seed=42),
        ContentBasedRecommender(similarity_threshold=0.0, seed=42),
        
        # Template for comparison
        MyRecommender(seed=42)
    ]
    recommender_names = ["LightningGraph", "EfficientGraph", "SVM", "Random", "Popularity", "ContentBased", "MyRecommender"]
    
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