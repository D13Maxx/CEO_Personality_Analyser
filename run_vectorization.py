"""Embedding generation runner."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from src.vectorization.embedding_generator import generate_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("=== Starting Vectorization ===")
    try:
        generate_pipeline()
        logger.info("=== Vectorization Complete ===")
    except Exception as e:
        logger.error("Vectorization failed: %s", e)


if __name__ == "__main__":
    main()
