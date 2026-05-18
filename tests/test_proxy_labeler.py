"""Tests for the proxy labeler."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.modeling.proxy_labeler import ProxyLabeler


@pytest.fixture
def labeler():
    return ProxyLabeler()


def test_empty_text_returns_zeros(labeler):
    feats = labeler.compute_linguistic_features("")
    assert feats["word_count"] == 0
    assert feats["sentence_count"] == 0
    assert feats["type_token_ratio"] == 0.0


def test_detects_pronouns_and_emotion(labeler):
    text = "We are very excited and happy to announce this new partnership. It will drive growth."
    feats = labeler.compute_linguistic_features(text)

    assert feats["word_count"] > 0
    assert feats["sentence_count"] == 2
    assert feats["first_person_plural_rate"] > 0
    assert feats["positive_emotion_rate"] > 0
    assert feats["anger_words_rate_inv"] == 0.0


def test_relative_trait_scores(labeler):
    extraverted = "We are so happy and excited to work together with our amazing community!"
    neurotic = "I am very anxious and worried about this terrible crisis. I fear it will be difficult."

    scores_e = labeler.score_text(extraverted)
    scores_n = labeler.score_text(neurotic)

    # Extraverted text should rank higher on extraversion than neuroticism
    assert scores_e["extraversion"] > scores_e["neuroticism"]
    # Neurotic text should rank higher on neuroticism than the extraverted text
    assert scores_n["neuroticism"] > scores_e["neuroticism"]
