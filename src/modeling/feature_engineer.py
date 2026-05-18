"""
Feature engineering pipeline.

Loads pre-computed embeddings, generates proxy labels via the
ProxyLabeler, merges them, and returns a scaled feature matrix
ready for XGBoost.
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import PROCESSED_DIR, EMBEDDINGS_DIR, MODELS_DIR
from src.modeling.proxy_labeler import ProxyLabeler

logger = logging.getLogger(__name__)


class FeatureEngineer:

    def __init__(self):
        self.embeddings_path = EMBEDDINGS_DIR / "ceo_embeddings.parquet"
        self.corpus_path = PROCESSED_DIR / "corpus_index.parquet"
        self.scaler_path = MODELS_DIR / "feature_scaler.joblib"
        self.labeler = ProxyLabeler()
        self.scaler = StandardScaler()

    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
        """
        Build (X, Y, feature_names).

        X uses only embedding dimensions -- linguistic features are
        reserved for proxy label generation and excluded here to
        prevent leakage.
        """
        if not self.embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings not found: {self.embeddings_path}")

        df_emb = pd.read_parquet(self.embeddings_path)
        logger.info("Loaded %d embeddings.", len(df_emb))

        emb_cols = [f"emb_{i}" for i in range(len(df_emb.iloc[0]["embedding"]))]
        emb_matrix = pd.DataFrame(df_emb["embedding"].tolist(), columns=emb_cols)
        emb_matrix["ceo_name"] = df_emb["ceo_name"]

        labels = self.labeler.build_labels(PROCESSED_DIR)
        merged = pd.merge(emb_matrix, labels, on="ceo_name", how="inner")
        logger.info("Merged dataset shape: %s", merged.shape)

        X_raw = merged[emb_cols]
        Y = merged[["ceo_name"] + list(self.labeler.lexicon.keys())]

        X_scaled = pd.DataFrame(self.scaler.fit_transform(X_raw), columns=emb_cols)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, self.scaler_path)

        X_scaled["ceo_name"] = merged["ceo_name"].values
        return X_scaled, Y, emb_cols

    def prepare_inference_data(self, text: str, embedding: np.ndarray) -> pd.DataFrame:
        """Scale a single embedding vector using the saved training scaler."""
        if not self.scaler_path.exists():
            raise FileNotFoundError("Scaler not found. Train the model first.")

        self.scaler = joblib.load(self.scaler_path)
        cols = self.scaler.feature_names_in_

        row = {col: 0.0 for col in cols}
        for i, val in enumerate(embedding):
            key = f"emb_{i}"
            if key in row:
                row[key] = val

        X_raw = pd.DataFrame([row])[cols]
        return pd.DataFrame(self.scaler.transform(X_raw), columns=cols)
