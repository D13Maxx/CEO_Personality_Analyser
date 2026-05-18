"""
Global configuration for the CEO Personality Analyzer.
Paths, hyperparameters, and constants.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
IS_COLAB = "google.colab" in sys.modules

if IS_COLAB:
    DATA_DIR = Path("/content/drive/MyDrive/CEO_Project/data")
else:
    DATA_DIR = Path("G:\\My Drive\\CEO_Project\\data")

RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
MODELS_DIR = DATA_DIR / "models"
DICT_DIR = PROJECT_ROOT / "dictionaries"

PROCESSED_SCREENER_DIR = PROCESSED_DIR / "screener"
SCREENER_PDF_DIR = RAW_PDF_DIR / "screener"

# Big Five traits

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

TRAIT_LABELS = {
    "openness": "Openness to Experience",
    "conscientiousness": "Conscientiousness",
    "extraversion": "Extraversion",
    "agreeableness": "Agreeableness",
    "neuroticism": "Neuroticism",
}

TRAIT_COLORS = {
    "openness": "#8B5CF6",
    "conscientiousness": "#3B82F6",
    "extraversion": "#F59E0B",
    "agreeableness": "#10B981",
    "neuroticism": "#EF4444",
}

SCORE_MIN = 1.0
SCORE_MAX = 7.0
SCORE_MIDPOINT = 4.0

# Screener.in scraper

SCREENER_BASE_URL = "https://www.screener.in"
SCREENER_SEARCH_API = f"{SCREENER_BASE_URL}/api/company/search/"
SCREENER_MIN_SLEEP = 4
SCREENER_MAX_SLEEP = 8
SCREENER_PDF_MIN_SLEEP = 2
SCREENER_PDF_MAX_SLEEP = 4
SCREENER_BATCH_PAUSE_EVERY = 20
SCREENER_BATCH_PAUSE_MIN = 30
SCREENER_BATCH_PAUSE_MAX = 60
SCREENER_MAX_RETRIES = 3
COMPANIES_FILE = PROJECT_ROOT / "companies.txt"

# PDF parsing

MIN_PAGE_TEXT_LENGTH = 50
MAX_HEADER_LINES = 3
MAX_FOOTER_LINES = 2

# Corpus thresholds

MIN_WORDS_PER_CEO = 5000

# Transformer embeddings

TRANSFORMER_MODEL_NAME = "all-mpnet-base-v2"
EMBEDDING_DIM = 768
MAX_SEQ_LENGTH = 384

# XGBoost defaults (Optuna overrides these per trait)

XGB_MAX_DEPTH = 6
XGB_N_ESTIMATORS = 300
XGB_LEARNING_RATE = 0.05
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8
XGB_REG_ALPHA = 0.1
XGB_REG_LAMBDA = 1.0
XGB_CV_FOLDS = 5

# Optuna tuning

OPTUNA_N_TRIALS = 50
OPTUNA_CV_FOLDS = 5
OPTUNA_TIMEOUT = 300  # seconds per trait

# Streamlit UI

APP_TITLE = "Socrates"
APP_SUBTITLE = "Loquere ut te videam"
APP_ICON = "\U0001f3db\ufe0f"
MAX_UPLOAD_SIZE_MB = 50

# PDF report

REPORT_TITLE = "CEO Personality Profile"
REPORT_FONT = "Helvetica"
REPORT_MARGIN = 15