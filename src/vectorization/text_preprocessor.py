"""
Text preprocessor for semantic vectorization.

Handles cleaning, tokenization, chunking, and basic linguistic
statistics needed by the proxy labeler and embedding generator.
"""

import re
import logging
from pathlib import Path
from typing import List, Set

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords as nltk_stopwords

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DICT_DIR

logger = logging.getLogger(__name__)

for resource in ("tokenizers/punkt", "tokenizers/punkt_tab", "corpora/stopwords"):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split("/")[-1], quiet=True)


class TextPreprocessor:

    def __init__(self):
        self.stop_words = self._build_stopword_set()

    def _build_stopword_set(self) -> Set[str]:
        words = set(nltk_stopwords.words("english"))
        custom = DICT_DIR / "stopwords.txt"
        if custom.exists():
            with open(custom, "r", encoding="utf-8") as f:
                words.update(line.strip().lower() for line in f if line.strip())
        return words

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'[\n\t\r]+', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def tokenize_sentences(self, text: str) -> List[str]:
        return sent_tokenize(self.clean_text(text))

    def tokenize_words(self, text: str, remove_stopwords: bool = True) -> List[str]:
        alpha_only = re.sub(r'[^a-z\s]', ' ', text.lower())
        tokens = word_tokenize(alpha_only)
        if remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words and len(t) > 1]
        return tokens

    def chunk_text(self, text: str, chunk_size: int = 5) -> List[str]:
        """Group sentences into chunks for transformer encoding."""
        sentences = self.tokenize_sentences(text)
        return [" ".join(sentences[i:i + chunk_size])
                for i in range(0, len(sentences), chunk_size)]

    def extract_linguistic_stats(self, text: str) -> dict:
        sentences = self.tokenize_sentences(text)
        words = word_tokenize(re.sub(r'[^a-z\s]', ' ', text.lower()))
        wc = len(words)
        sc = len(sentences)

        if wc == 0:
            return {"word_count": 0, "sentence_count": 0,
                    "avg_words_per_sentence": 0, "avg_word_length": 0,
                    "type_token_ratio": 0}

        return {
            "word_count": wc,
            "sentence_count": sc,
            "avg_words_per_sentence": wc / sc if sc else 0,
            "avg_word_length": sum(len(w) for w in words) / wc,
            "type_token_ratio": len(set(words)) / wc,
        }
