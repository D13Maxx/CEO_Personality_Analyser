"""
Corpus builder runner.
Parses all PDFs, identifies executives via management hierarchy,
scrubs financial text, aggregates speech, and saves the corpus.
"""

import argparse
import logging
from pathlib import Path

from src.ingestion.corpus_builder import build_corpus

import sys
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from config import SCREENER_PDF_DIR, PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build CEO Speech Corpus from Transcripts.")
    parser.add_argument("--raw-dir", type=str, default=str(SCREENER_PDF_DIR))
    parser.add_argument("--out-dir", type=str, default=str(PROCESSED_DIR))
    parser.add_argument("--local-copy", action="store_true",
                        help="Copy PDFs to Colab local disk before parsing")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_true")

    args = parser.parse_args()
    resume = args.resume and not args.no_resume

    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        logger.error("PDF directory not found: %s", raw_dir)
        return

    logger.info("Raw: %s | Out: %s | Local-copy: %s | Resume: %s",
                raw_dir, args.out_dir, args.local_copy, resume)

    stats = build_corpus(raw_dir=raw_dir, out_dir=args.out_dir,
                         local_copy=args.local_copy, resume=resume)

    logger.info("=== Complete ===")
    for k, v in stats.items():
        logger.info("  %s: %s", k.replace("_", " ").title(), v)


if __name__ == "__main__":
    main()
