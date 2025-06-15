import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from pyspark.sql import DataFrame, Window
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sklearn
from sim4rec.utils import pandas_to_spark
import xgboost as xgb
from sklearn.model_selection import train_test_split

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

class MyRecommender(BaseRecommender):
    def __init__(self, seed=None):
        super().__init__(seed)
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            early_stopping_rounds=10,
            eval_metric="logloss",
        )
        self.scalar = StandardScaler()

    def fit(self, log, user_features=None, item_features=None):
        if user_features and item_features:
            pd_log = log.join(
                user_features,
                on='user_idx'
            ).join(
                item_features,
                on='item_idx'
            ).drop(
                'user_idx', 'item_idx', '__iter'
            ).toPandas()

            pd_log = pd.get_dummies(pd_log)
            pd_log['price'] = self.scalar.fit_transform(pd_log[['price']])

            y = pd_log['relevance']
            X = pd_log.drop(['relevance'], axis=1)

            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=self.seed
            )

            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        cross = users.join(
            items
        ).drop('__iter').toPandas().copy()

        cross = pd.get_dummies(cross)
        cross['orig_price'] = cross['price']
        cross['price'] = self.scalar.transform(cross[['price']])

        # Predict probability of relevance == 1
        cross['prob'] = self.model.predict_proba(
            cross.drop(['user_idx', 'item_idx', 'orig_price'], axis=1)
        )[:, np.where(self.model.classes_ == 1)[0][0]]

        cross["relevance"] = cross["prob"] * cross["orig_price"]

        cross = cross.sort_values(by=['user_idx', 'relevance'], ascending=[True, False])
        cross = cross.groupby('user_idx').head(k)

        cross['price'] = cross['orig_price']

        return pandas_to_spark(cross)