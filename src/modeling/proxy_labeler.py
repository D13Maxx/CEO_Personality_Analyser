"""
Proxy labeler for ground-truth generation.

Implements a heuristic scorer inspired by Mairesse (2007) and LIWC.
Maps raw transcript text to noisy 1-7 Big Five scores using
dictionary lookups and stylometric features. These serve as
training labels for the downstream XGBoost models.
"""

import json
import logging
from pathlib import Path
from typing import Dict

import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DICT_DIR, TRAITS, SCORE_MIN, SCORE_MAX
from src.vectorization.text_preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)


class ProxyLabeler:

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.lexicon = self._load_lexicon()
        self.word_categories = {
            "first_person_singular": {"i", "me", "my", "mine", "myself"},
            "first_person_plural": {"we", "us", "our", "ours", "ourselves"},
            "social": (self._lexicon_words("agreeableness", "positive") |
                       self._lexicon_words("extraversion", "positive")),
            "positive_emotion": (self._lexicon_words("agreeableness", "positive") |
                                 self._lexicon_words("extraversion", "positive")),
            "negative_emotion": self._lexicon_words("neuroticism", "positive"),
            "anger": {"angry", "mad", "furious", "rage", "hostile", "fight", "argue"},
            "certainty": {"always", "never", "certainly", "absolutely", "definite", "ensure"},
            "hedging": {"maybe", "perhaps", "might", "could", "seem", "appear", "somewhat"},
            "future_tense": {"will", "shall", "gonna", "going", "expect", "plan", "future"},
            "negation": {"not", "no", "never", "none", "neither", "nor", "cannot", "can't", "don't", "won't"},
            "articles": {"a", "an", "the"},
            "prepositions": {"in", "on", "at", "by", "for", "with", "about", "against",
                             "between", "into", "through"},
        }

    def _load_lexicon(self):
        path = DICT_DIR / "personality_lexicon.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        logger.warning("Lexicon not found at %s", path)
        return {t: {"positive": [], "negative": [], "features": []} for t in TRAITS}

    def _lexicon_words(self, trait, polarity):
        return set(self.lexicon.get(trait, {}).get(polarity, []))

    def compute_linguistic_features(self, text: str) -> Dict[str, float]:
        stats = self.preprocessor.extract_linguistic_stats(text)
        words = self.preprocessor.tokenize_words(text, remove_stopwords=False)
        wc = len(words)

        if wc == 0:
            features = {k: 0.0 for k in self.word_categories}
            features.update(stats)
            for t in TRAITS:
                features[f"{t}_pos_rate"] = 0.0
                features[f"{t}_neg_rate"] = 0.0
            return features

        cat_counts = {cat: 0 for cat in self.word_categories}
        trait_counts = {f"{t}_{p}": 0 for t in TRAITS for p in ("pos", "neg")}

        trait_word_sets = {}
        for t in TRAITS:
            trait_word_sets[f"{t}_pos"] = self._lexicon_words(t, "positive")
            trait_word_sets[f"{t}_neg"] = self._lexicon_words(t, "negative")

        for word in words:
            for cat, word_set in self.word_categories.items():
                if word in word_set:
                    cat_counts[cat] += 1
            for key, word_set in trait_word_sets.items():
                if word in word_set:
                    trait_counts[key] += 1

        # Rates per 1000 words
        features = stats.copy()
        for cat, count in cat_counts.items():
            features[f"{cat}_rate"] = (count / wc) * 1000
        for key, count in trait_counts.items():
            features[f"{key}_rate"] = (count / wc) * 1000

        features["negation_rate_inv"] = -features["negation_rate"]
        features["anger_words_rate_inv"] = -features.get("anger_rate", 0)
        features["words_per_turn"] = stats["avg_words_per_sentence"] * 3

        return features

    def score_text(self, text: str) -> Dict[str, float]:
        """Compute uncalibrated trait scores (need z-scoring across the corpus)."""
        features = self.compute_linguistic_features(text)
        scores = {}

        for trait in TRAITS:
            score = features.get(f"{trait}_pos_rate", 0) - features.get(f"{trait}_neg_rate", 0)

            for feat_name in self.lexicon.get(trait, {}).get("features", []):
                val = features.get(feat_name, 0)
                if "rate" in feat_name:
                    score += val * 0.5
                elif feat_name == "type_token_ratio":
                    score += val * 100
                elif feat_name == "avg_word_length":
                    score += val * 5
                else:
                    score += val * 0.1

            scores[trait] = score
        return scores

    def build_labels(self, processed_dir) -> pd.DataFrame:
        """
        Score the entire corpus and standardize to 1-7 using
        z-score normalization (mean=4, sd=1).
        """
        processed_dir = Path(processed_dir)
        index = processed_dir / "corpus_index.parquet"
        corpus = {}

        if index.exists():
            df = pd.read_parquet(index)
            for _, row in df[df["meets_threshold"] == True].iterrows():
                corpus[row["ceo_name"]] = row["text"]

        if not corpus:
            logger.error("No corpus found to label.")
            return pd.DataFrame()

        logger.info("Computing heuristic labels for %d CEOs...", len(corpus))
        raw_scores = []
        for name, text in corpus.items():
            scores = self.score_text(text)
            scores["ceo_name"] = name
            raw_scores.append(scores)

        df_raw = pd.DataFrame(raw_scores)
        result = pd.DataFrame({"ceo_name": df_raw["ceo_name"]})

        for trait in TRAITS:
            z = (df_raw[trait] - df_raw[trait].mean()) / (df_raw[trait].std() + 1e-9)
            result[trait] = (z + 4.0).clip(lower=SCORE_MIN, upper=SCORE_MAX)

        return result
