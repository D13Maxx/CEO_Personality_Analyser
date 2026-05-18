"""Model training runner (with Optuna tuning)."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from src.modeling.trainer import train_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("=== Starting Model Training ===")
    try:
        train_models()
        logger.info("=== Training Complete ===")
    except Exception as e:
        logger.error("Training failed: %s", e)


if __name__ == "__main__":
    main()
