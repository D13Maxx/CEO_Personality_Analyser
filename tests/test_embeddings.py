"""Tests for the embedding generator and text preprocessor."""

import pytest
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.vectorization.text_preprocessor import TextPreprocessor
from src.vectorization.embedding_generator import EmbeddingGenerator
from config import EMBEDDING_DIM


def test_clean_text_lowercases_and_strips():
    prep = TextPreprocessor()
    text = "Hello WORLD! This is a test... with numbers 123 and \n newlines."
    cleaned = prep.clean_text(text)

    assert "hello world!" in cleaned
    assert "newlines" in cleaned
    assert "\n" not in cleaned


def test_sentence_and_word_tokenization():
    prep = TextPreprocessor()
    text = "We expect revenue to grow. The margins are stable."

    sentences = prep.tokenize_sentences(text)
    assert len(sentences) == 2

    words = prep.tokenize_words(sentences[0], remove_stopwords=True)
    assert "expect" in words
    assert "grow" in words
    assert "we" not in words  # stopword


def test_embedding_with_mock_model():
    gen = EmbeddingGenerator()

    class MockModel:
        def encode(self, chunks, show_progress_bar=False, convert_to_numpy=True):
            return np.ones((len(chunks), EMBEDDING_DIM)) * 2.0

    gen.model = MockModel()

    vec = gen.embed_single_transcript("Revenue and growth margin.")
    assert vec is not None
    assert vec.shape == (EMBEDDING_DIM,)
    assert np.isclose(vec[0], 2.0)

    # Empty text should return zero vector
    vec_empty = gen.embed_single_transcript("")
    assert np.all(vec_empty == 0)
