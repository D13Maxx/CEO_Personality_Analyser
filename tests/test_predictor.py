"""Tests for the predictor percentile conversion."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.modeling.predictor import Predictor


def test_percentile_at_mean():
    pred = Predictor()
    # 4.0 is the population mean; should be ~50th percentile
    assert 48.0 <= pred._to_percentile(4.0) <= 52.0


def test_percentile_one_sd_above():
    pred = Predictor()
    # 5.0 is +1 SD; should be ~84th percentile
    assert 82.0 <= pred._to_percentile(5.0) <= 86.0


def test_percentile_one_sd_below():
    pred = Predictor()
    # 3.0 is -1 SD; should be ~16th percentile
    assert 14.0 <= pred._to_percentile(3.0) <= 18.0


def test_percentile_extreme_bounds():
    pred = Predictor()
    assert pred._to_percentile(7.0) == 99.0
    assert pred._to_percentile(1.0) == 1.0
