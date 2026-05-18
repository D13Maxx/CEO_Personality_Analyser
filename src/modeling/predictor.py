"""
Inference pipeline.

End-to-end prediction of Big Five personality traits from a
single transcript PDF. Wraps parsing, embedding, feature
engineering, and the trained XGBoost models.
"""

import logging
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass

import numpy as np
from xgboost import XGBRegressor

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import MODELS_DIR, TRAITS, SCORE_MIN, SCORE_MAX
from src.ingestion.transcript_parser import (
    extract_text_from_pdf, identify_speakers,
    parse_transcript, clean_financial_text,
)
from src.vectorization.embedding_generator import EmbeddingGenerator
from src.modeling.feature_engineer import FeatureEngineer

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    ceo_name: str
    word_count: int
    scores: Dict[str, float]
    percentiles: Dict[str, float]
    transcript_summary: dict
    warnings: list


class Predictor:

    def __init__(self):
        self.models_dir = MODELS_DIR
        self.models: Dict[str, XGBRegressor] = {}
        self.embedding_gen = EmbeddingGenerator()
        self.feature_eng = FeatureEngineer()
        self.corpus_stats = self._load_corpus_stats()

    def _load_models(self):
        if len(self.models) == len(TRAITS):
            return True
        for trait in TRAITS:
            path = self.models_dir / f"{trait}_xgb.json"
            if not path.exists():
                logger.error("Model not found: %s", path)
                return False
            model = XGBRegressor()
            model.load_model(str(path))
            self.models[trait] = model
        return True

    def _load_corpus_stats(self):
        path = self.models_dir / "cv_metrics.json"
        if not path.exists():
            return {}
        import json
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def _to_percentile(score):
        """Convert 1-7 score to percentile assuming N(4, 1)."""
        import scipy.stats as stats
        z = (score - 4.0) / 1.0
        return max(1.0, min(99.0, stats.norm.cdf(z) * 100))

    def predict_from_pdf(self, file_path, target_ceo_name=None):
        """Run full prediction on a PDF transcript."""
        file_path = Path(file_path)
        warnings = []

        if file_path.suffix.lower() != ".pdf":
            logger.error("Unsupported format. Only PDF is supported.")
            return None

        result = parse_transcript(file_path)
        if result is None:
            logger.error("Could not extract text from document.")
            return None

        if not result.executive_name or not result.speech_text:
            logger.error("Could not identify executive speech.")
            return None

        summary = {s: {"role": "unknown", "turns": 0, "total_words": 0}
                   for s in result.all_speakers}

        if result.word_count < 500:
            warnings.append(f"Low word count ({result.word_count}). Results may be unreliable.")

        return self.predict_from_text(
            result.executive_name, result.speech_text, summary, warnings)

    def predict_from_text(self, ceo_name, speech_text, summary=None, warnings=None):
        """Predict traits from pre-extracted speech text."""
        if not self._load_models():
            return None

        warnings = warnings or []
        summary = summary or {}

        embedding = self.embedding_gen.embed_single_transcript(speech_text)
        if embedding is None:
            logger.error("Failed to generate embedding.")
            return None

        try:
            features = self.feature_eng.prepare_inference_data(speech_text, embedding)
        except Exception as e:
            logger.error("Feature engineering failed: %s", e)
            return None

        scores, percentiles = {}, {}
        for trait in TRAITS:
            pred = float(self.models[trait].predict(features.values)[0])
            pred = max(SCORE_MIN, min(SCORE_MAX, pred))
            scores[trait] = pred
            percentiles[trait] = self._to_percentile(pred)

        return PredictionResult(
            ceo_name=ceo_name,
            word_count=len(speech_text.split()),
            scores=scores,
            percentiles=percentiles,
            transcript_summary=summary,
            warnings=warnings,
        )
