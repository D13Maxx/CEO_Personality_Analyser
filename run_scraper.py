"""
Screener.in transcript scraper runner.
Pauses Google Drive sync during heavy I/O to prevent contention.
"""

import argparse
import logging
from pathlib import Path

from src.ingestion.screener_scraper import ScreenerScraper
from src.ingestion.gdrive_pause import gdrive_paused

import sys
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from config import COMPANIES_FILE, SCREENER_PDF_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Scrape Transcripts from Screener.in")
    parser.add_argument("--companies", type=str, default=str(COMPANIES_FILE))
    parser.add_argument("--start-year", type=int, default=2022)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--dest", type=str, default=None)

    args = parser.parse_args()

    scraper = ScreenerScraper(
        dest_dir=Path(args.dest) if args.dest else None,
        start_year=args.start_year,
        end_year=args.end_year,
    )

    try:
        with gdrive_paused() as was_paused:
            if was_paused:
                print("[OK] Google Drive sync paused.")
            else:
                print("[!] Could not auto-pause Drive. Pause manually if freezing occurs.")

            result = scraper.scrape_all_companies(companies_file=Path(args.companies))
            scraper.save_manifest(result)

    except KeyboardInterrupt:
        print("\nInterrupted. Progress checkpointed; re-run to resume.")


if __name__ == "__main__":
    main()
