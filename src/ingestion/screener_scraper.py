"""
Screener.in Earnings Call Transcript Scraper.

Searches for companies on screener.in, discovers transcript PDF links
from the Documents section, and downloads PDFs for the specified year range.

Features:
  - Name → ticker resolution via Screener's search API.
  - Date-filtered transcript extraction (default: 2022-2025).
  - Anti-ban measures: random delays, batch pauses, exponential backoff.
  - Checkpoint/resume support for long-running sessions.
  - Final JSON report with per-company success/failure stats.
"""

import os
import re
import time
import json
import random
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Tuple
from difflib import SequenceMatcher

import requests
from bs4 import BeautifulSoup

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    SCREENER_BASE_URL, SCREENER_SEARCH_API,
    SCREENER_MIN_SLEEP, SCREENER_MAX_SLEEP,
    SCREENER_PDF_MIN_SLEEP, SCREENER_PDF_MAX_SLEEP,
    SCREENER_BATCH_PAUSE_EVERY, SCREENER_BATCH_PAUSE_MIN,
    SCREENER_BATCH_PAUSE_MAX, SCREENER_MAX_RETRIES,
    SCREENER_PDF_DIR, COMPANIES_FILE,
)

logger = logging.getLogger(__name__)


# ============================================================
# Data Classes
# ============================================================

@dataclass
class TranscriptInfo:
    """A single transcript discovered on screener.in."""
    company_name: str
    ticker: str
    date_label: str        # e.g. "Jan 2023"
    year: int
    month: int
    pdf_url: str
    downloaded: bool = False
    local_path: str = ""
    error: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class CompanyResult:
    """Result of scraping a single company."""
    company_name: str
    ticker: str = ""
    screener_url: str = ""
    status: str = "pending"  # pending, success, no_transcripts, not_found, error
    transcripts_found: int = 0
    transcripts_downloaded: int = 0
    error: str = ""
    transcripts: List[TranscriptInfo] = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        return d


# Minimum number of transcripts to keep a company's PDFs
MIN_TRANSCRIPTS = 5


@dataclass
class ScrapeResult:
    """Overall scraping session summary."""
    total_companies: int = 0
    companies_resolved: int = 0
    companies_with_transcripts: int = 0
    total_transcripts_found: int = 0
    total_pdfs_downloaded: int = 0
    companies_not_found: List[str] = field(default_factory=list)
    companies_no_transcripts: List[str] = field(default_factory=list)
    companies_with_errors: List[str] = field(default_factory=list)
    companies_discarded: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# ============================================================
# Month Parsing
# ============================================================

MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def parse_screener_date(date_text: str) -> Tuple[int, int]:
    """
    Parse screener.in date like 'Jan 2023' → (2023, 1).
    Returns (0, 0) if parsing fails.
    """
    date_text = date_text.strip()
    parts = date_text.split()
    if len(parts) < 2:
        return 0, 0

    month_str = parts[0].lower()[:3]
    month = MONTH_MAP.get(month_str, 0)

    try:
        year = int(parts[-1])
    except ValueError:
        return 0, 0

    return year, month


# ============================================================
# Scraper
# ============================================================

class ScreenerScraper:
    """
    Scrapes transcript PDFs from screener.in for a list of companies.
    """

    CHECKPOINT_FILE = "_screener_checkpoint.json"

    # Standard browser headers to avoid bot detection
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }

    def __init__(
        self,
        dest_dir: Optional[Path] = None,
        start_year: int = 2022,
        end_year: int = 2025,
    ):
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self.dest_dir = Path(dest_dir or SCREENER_PDF_DIR)
        self.dest_dir.mkdir(parents=True, exist_ok=True)
        self.start_year = start_year
        self.end_year = end_year

        # Cache existing filenames to skip re-downloads
        self._existing_files = self._load_existing_files()

        # Checkpoint state
        self._checkpoint_path = self.dest_dir / self.CHECKPOINT_FILE
        self._checkpoint: Dict = self._load_checkpoint()

    # ----------------------------------------------------------
    # Filesystem helpers
    # ----------------------------------------------------------

    def _load_existing_files(self) -> set:
        """Cache existing filenames from download directory (once)."""
        if not self.dest_dir.exists():
            return set()
        try:
            cached = set(os.listdir(self.dest_dir))
            logger.info(f"Cached {len(cached)} existing filenames from {self.dest_dir}")
            return cached
        except Exception as e:
            logger.warning(f"Could not cache directory listing: {e}")
            return set()

    # ----------------------------------------------------------
    # Checkpoint helpers
    # ----------------------------------------------------------

    def _load_checkpoint(self) -> dict:
        """Load checkpoint from disk, or return a fresh one."""
        if self._checkpoint_path.exists():
            try:
                data = json.loads(self._checkpoint_path.read_text(encoding="utf-8"))
                done = len(data.get("completed_companies", []))
                logger.info(f"Resumed checkpoint: {done} companies already processed.")
                return data
            except Exception as e:
                logger.warning(f"Corrupt checkpoint, starting fresh: {e}")
        return {
            "completed_companies": [],   # list of company names already done
            "company_results": {},       # name → CompanyResult dict
            "started_at": datetime.now().isoformat(),
        }

    def _save_checkpoint(self):
        """Persist checkpoint to disk."""
        self._checkpoint["updated_at"] = datetime.now().isoformat()
        self._checkpoint_path.write_text(
            json.dumps(self._checkpoint, indent=2, default=str),
            encoding="utf-8",
        )

    # ----------------------------------------------------------
    # HTTP helpers
    # ----------------------------------------------------------

    def _random_sleep(self, min_s=None, max_s=None):
        """Sleep for a random duration."""
        min_s = min_s or SCREENER_MIN_SLEEP
        max_s = max_s or SCREENER_MAX_SLEEP
        delay = random.uniform(min_s, max_s)
        logger.debug(f"Sleeping {delay:.1f}s...")
        time.sleep(delay)

    def _get(self, url: str, retries: int = None, is_pdf: bool = False) -> Optional[requests.Response]:
        """GET with retry + exponential backoff. Returns Response or None."""
        retries = retries or SCREENER_MAX_RETRIES
        for attempt in range(retries):
            if is_pdf:
                self._random_sleep(SCREENER_PDF_MIN_SLEEP, SCREENER_PDF_MAX_SLEEP)
            else:
                self._random_sleep()
            try:
                r = self.session.get(url, timeout=30)
                if r.status_code == 404:
                    logger.warning(f"404 Not Found: {url}")
                    return None
                if r.status_code == 429:
                    wait = 2 ** (attempt + 2) + random.uniform(5, 15)
                    logger.warning(f"Rate limited (429). Waiting {wait:.0f}s...")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                return r
            except Exception as e:
                wait = 2 ** attempt + random.uniform(0, 2)
                logger.warning(
                    f"Attempt {attempt+1}/{retries} failed for {url}: {e}. "
                    f"Retry in {wait:.1f}s"
                )
                time.sleep(wait)
        return None

    # ----------------------------------------------------------
    # Step 1: Resolve company name → screener.in URL
    # ----------------------------------------------------------

    def _resolve_company(self, company_name: str) -> Optional[Tuple[str, str]]:
        """
        Search screener.in API to find the company's ticker and URL.

        Returns:
            (ticker, url_path) or None if not found.
        """
        # Clean up the company name for searching
        search_name = company_name.strip()
        # Remove "Ltd." / "Ltd" suffix for better matching
        search_query = re.sub(r'\s+Ltd\.?$', '', search_name, flags=re.IGNORECASE).strip()

        url = f"{SCREENER_SEARCH_API}?q={requests.utils.quote(search_query)}"
        resp = self._get(url)
        if not resp:
            return None

        try:
            results = resp.json()
        except Exception:
            logger.error(f"Invalid JSON from search API for '{company_name}'")
            return None

        if not results:
            # Try with a shorter query (first two words)
            words = search_query.split()
            if len(words) > 2:
                short_query = " ".join(words[:2])
                url = f"{SCREENER_SEARCH_API}?q={requests.utils.quote(short_query)}"
                resp = self._get(url)
                if resp:
                    try:
                        results = resp.json()
                    except Exception:
                        pass

        if not results:
            return None

        # Find best match using fuzzy matching
        best_match = None
        best_score = 0.0
        name_lower = company_name.lower()

        for r in results:
            r_name = r.get("name", "").lower()
            score = SequenceMatcher(None, name_lower, r_name).ratio()
            if score > best_score:
                best_score = score
                best_match = r

        # Accept if score is reasonably high, or if it's the only result
        if best_match and (best_score > 0.4 or len(results) == 1):
            url_path = best_match["url"]
            # Extract ticker from URL: /company/INFY/consolidated/ → INFY
            ticker_match = re.search(r'/company/([^/]+)/', url_path)
            ticker = ticker_match.group(1) if ticker_match else best_match.get("name", "UNKNOWN")
            return ticker, url_path

        return None

    # ----------------------------------------------------------
    # Step 2: Get transcript links from company page
    # ----------------------------------------------------------

    def _get_transcript_links(self, url_path: str) -> List[Tuple[str, int, int, str]]:
        """
        Fetch company page and extract transcript PDF links.

        Returns:
            List of (date_label, year, month, pdf_url) tuples.
        """
        # Build full URL
        page_url = f"{SCREENER_BASE_URL}{url_path}"
        resp = self._get(page_url)
        if not resp:
            return []

        soup = BeautifulSoup(resp.text, "html.parser")

        # Find the documents section
        docs_section = soup.find("section", id="documents")
        if not docs_section:
            # Try alternative: div with id documents
            docs_section = soup.find("div", id="documents")
        if not docs_section:
            logger.warning(f"No documents section found on {page_url}")
            return []

        transcripts = []

        # Find all document rows/entries
        # Screener structure: each row has a date and concall links
        # Look for links with class "concall-link" and text/title containing "Transcript"
        rows = docs_section.find_all("div", class_="documents-concall") or \
               docs_section.find_all("li") or \
               docs_section.find_all("div")

        # Strategy: find all transcript links in the documents section
        all_links = docs_section.find_all("a", class_="concall-link")

        if not all_links:
            # Broader search: any link with "transcript" in text
            all_links = docs_section.find_all("a", href=True)
            all_links = [
                a for a in all_links
                if "transcript" in (a.get_text(strip=True).lower())
                or "transcript" in (a.get("title", "").lower())
            ]

        for link in all_links:
            link_text = link.get_text(strip=True).lower()
            link_title = link.get("title", "").lower()

            # Only want "Transcript" links (not "AI Summary", "PPT", "REC")
            if "transcript" not in link_text and "transcript" not in link_title:
                continue

            href = link.get("href", "")
            if not href:
                continue

            # Make absolute URL
            if href.startswith("/"):
                href = f"{SCREENER_BASE_URL}{href}"
            elif not href.startswith("http"):
                continue

            # Find the associated date: look at parent/sibling elements
            date_label = self._find_date_for_link(link)
            year, month = parse_screener_date(date_label)

            if year == 0:
                # Try to extract year from nearby text
                parent_text = link.parent.get_text(strip=True) if link.parent else ""
                for y in range(2020, 2030):
                    if str(y) in parent_text:
                        year = y
                        break

            transcripts.append((date_label, year, month, href))

        return transcripts

    def _find_date_for_link(self, link_tag) -> str:
        """
        Walk up the DOM from a transcript link to find the associated date text.
        Screener.in typically has the date in a sibling or parent element.
        """
        # Check parent and grandparent for date text
        for ancestor in [link_tag.parent, link_tag.parent.parent if link_tag.parent else None]:
            if not ancestor:
                continue
            # Look for text like "Jan 2023", "Oct 2024", etc.
            text = ancestor.get_text(" ", strip=True)
            # Match month-year pattern
            date_match = re.search(
                r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})',
                text,
                re.IGNORECASE,
            )
            if date_match:
                return f"{date_match.group(1)} {date_match.group(2)}"

        # Fallback: look at previous siblings
        prev = link_tag.find_previous(string=re.compile(
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}',
            re.IGNORECASE,
        ))
        if prev:
            m = re.search(
                r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})',
                str(prev),
                re.IGNORECASE,
            )
            if m:
                return f"{m.group(1)} {m.group(2)}"

        return "Unknown"

    # ----------------------------------------------------------
    # Step 3: Download PDF
    # ----------------------------------------------------------

    def _make_filename(self, company_name: str, date_label: str, year: int, month: int) -> str:
        """Generate a safe filename for the downloaded PDF using the full company name."""
        # Sanitize company name for filesystem: remove special chars, replace spaces
        safe_name = re.sub(r'[<>:"/\\|?*]', '', company_name)  # remove illegal chars
        safe_name = re.sub(r'[&]', 'and', safe_name)            # & → and
        safe_name = safe_name.replace(' ', '_').replace('.', '').strip('_')
        month_str = f"{month:02d}" if month else "00"
        safe_date = f"{year}_{month_str}"
        return f"{safe_name}_{safe_date}_{date_label.replace(' ', '_')}.pdf"

    def _download_pdf(self, pdf_url: str, filename: str) -> bool:
        """Download a single PDF to the destination directory."""
        # Check if already downloaded
        if filename in self._existing_files:
            logger.debug(f"Already exists: {filename}")
            return True

        resp = self._get(pdf_url, is_pdf=True)
        if not resp:
            return False

        # Validate it's a PDF (or at least a non-empty response)
        content = resp.content
        if len(content) < 500:
            logger.warning(f"File too small ({len(content)} bytes), likely not a PDF: {pdf_url}")
            return False

        fp = self.dest_dir / filename
        fp.write_bytes(content)
        self._existing_files.add(filename)
        logger.info(f"Downloaded: {filename} ({len(content):,} bytes)")
        return True

    # ----------------------------------------------------------
    # Main Orchestrator
    # ----------------------------------------------------------

    def scrape_all_companies(
        self,
        companies_file: Optional[Path] = None,
    ) -> ScrapeResult:
        """
        Main entry point: read company list, resolve tickers, find transcripts,
        download PDFs.

        Args:
            companies_file: Path to text file with one company name per line.

        Returns:
            ScrapeResult with summary statistics.
        """
        companies_file = Path(companies_file or COMPANIES_FILE)
        if not companies_file.exists():
            raise FileNotFoundError(f"Companies file not found: {companies_file}")

        # Read and deduplicate company names
        raw_lines = companies_file.read_text(encoding="utf-8").splitlines()
        companies = []
        seen = set()
        for line in raw_lines:
            name = line.strip()
            if name and name.lower() not in seen:
                companies.append(name)
                seen.add(name.lower())

        logger.info(f"=== Screener.in Transcript Scraper ===")
        logger.info(f"Companies: {len(companies)} | Years: {self.start_year}-{self.end_year}")
        logger.info(f"Destination: {self.dest_dir}")

        result = ScrapeResult(total_companies=len(companies))

        # Get already-completed companies from checkpoint
        completed = set(self._checkpoint.get("completed_companies", []))
        pending = [c for c in companies if c not in completed]

        if completed:
            logger.info(f"Resuming: {len(completed)} companies already processed, {len(pending)} remaining.")

        try:
            from tqdm import tqdm
            company_iter = tqdm(pending, desc="Scraping companies", initial=len(completed), total=len(companies))
        except ImportError:
            company_iter = pending

        for idx, company_name in enumerate(company_iter):
            comp_result = self._process_company(company_name)

            # Update result counters
            if comp_result.status == "success":
                result.companies_with_transcripts += 1
                result.companies_resolved += 1
            elif comp_result.status == "discarded":
                result.companies_discarded.append(company_name)
                result.companies_resolved += 1
            elif comp_result.status == "no_transcripts":
                result.companies_no_transcripts.append(company_name)
                result.companies_resolved += 1
            elif comp_result.status == "not_found":
                result.companies_not_found.append(company_name)
            else:
                result.companies_with_errors.append(company_name)

            result.total_transcripts_found += comp_result.transcripts_found
            result.total_pdfs_downloaded += comp_result.transcripts_downloaded

            # Save to checkpoint
            self._checkpoint["completed_companies"].append(company_name)
            self._checkpoint["company_results"][company_name] = comp_result.to_dict()
            self._save_checkpoint()

            # Batch pause: after every N companies, take a longer break
            companies_done = len(self._checkpoint["completed_companies"])
            if companies_done % SCREENER_BATCH_PAUSE_EVERY == 0 and idx < len(pending) - 1:
                pause = random.uniform(SCREENER_BATCH_PAUSE_MIN, SCREENER_BATCH_PAUSE_MAX)
                logger.info(f"Batch pause ({companies_done} companies done). Sleeping {pause:.0f}s...")
                time.sleep(pause)

        # Final report
        self._save_report(result, companies)
        return result

    def _process_company(self, company_name: str) -> CompanyResult:
        """Process a single company: resolve → find transcripts → download."""
        comp = CompanyResult(company_name=company_name)

        # Step 1: Resolve name → ticker
        logger.info(f"Processing: {company_name}")
        resolved = self._resolve_company(company_name)
        if not resolved:
            comp.status = "not_found"
            comp.error = "Could not find company on screener.in"
            logger.warning(f"  [X] Not found on screener.in: {company_name}")
            return comp

        ticker, url_path = resolved
        comp.ticker = ticker
        comp.screener_url = f"{SCREENER_BASE_URL}{url_path}"
        logger.info(f"  [OK] Resolved: {company_name} -> {ticker} ({url_path})")

        # Step 2: Get transcript links
        try:
            all_transcripts = self._get_transcript_links(url_path)
        except Exception as e:
            comp.status = "error"
            comp.error = f"Failed to fetch transcripts: {e}"
            logger.error(f"  [X] Error fetching transcripts: {e}")
            return comp

        # Filter by year range
        filtered = [
            (date_label, year, month, pdf_url)
            for date_label, year, month, pdf_url in all_transcripts
            if self.start_year <= year <= self.end_year
        ]

        comp.transcripts_found = len(filtered)
        logger.info(f"  Found {len(all_transcripts)} total transcripts, {len(filtered)} in {self.start_year}-{self.end_year}")

        if not filtered:
            comp.status = "no_transcripts"
            if all_transcripts:
                comp.error = f"Found {len(all_transcripts)} transcripts but none in {self.start_year}-{self.end_year}"
            else:
                comp.error = "No transcript links found on page"
            return comp

        # Step 3: Download PDFs
        downloaded = 0
        for date_label, year, month, pdf_url in filtered:
            filename = self._make_filename(company_name, date_label, year, month)
            transcript = TranscriptInfo(
                company_name=company_name,
                ticker=ticker,
                date_label=date_label,
                year=year,
                month=month,
                pdf_url=pdf_url,
            )

            try:
                success = self._download_pdf(pdf_url, filename)
                if success:
                    transcript.downloaded = True
                    transcript.local_path = str(self.dest_dir / filename)
                    downloaded += 1
                else:
                    transcript.error = "Download failed"
            except Exception as e:
                transcript.error = str(e)
                logger.warning(f"    [X] Download error: {e}")

            comp.transcripts.append(transcript)

        comp.transcripts_downloaded = downloaded

        # Discard companies with fewer than MIN_TRANSCRIPTS PDFs
        if 0 < downloaded < MIN_TRANSCRIPTS:
            logger.warning(
                f"  [!] Only {downloaded} PDFs for {ticker} (min {MIN_TRANSCRIPTS}). "
                f"Discarding downloaded files."
            )
            # Delete the downloaded PDFs
            for t in comp.transcripts:
                if t.downloaded and t.local_path:
                    try:
                        fp = Path(t.local_path)
                        if fp.exists():
                            fp.unlink()
                            self._existing_files.discard(fp.name)
                    except Exception as e:
                        logger.warning(f"    Could not delete {t.local_path}: {e}")
                    t.downloaded = False
                    t.local_path = ""
            comp.status = "discarded"
            comp.error = f"Only {downloaded} transcripts (minimum is {MIN_TRANSCRIPTS})"
            comp.transcripts_downloaded = 0
        elif downloaded > 0:
            comp.status = "success"
            if downloaded < len(filtered):
                comp.error = f"Only {downloaded}/{len(filtered)} PDFs downloaded"
        else:
            comp.status = "error"

        logger.info(f"  Downloaded {downloaded}/{len(filtered)} PDFs for {ticker}")
        return comp

    # ----------------------------------------------------------
    # Report
    # ----------------------------------------------------------

    def _save_report(self, result: ScrapeResult, companies: List[str]):
        """Save final JSON report."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "year_range": f"{self.start_year}-{self.end_year}",
            "summary": {
                "total_companies": result.total_companies,
                "companies_resolved": result.companies_resolved,
                "companies_with_transcripts": result.companies_with_transcripts,
                "total_transcripts_found": result.total_transcripts_found,
                "total_pdfs_downloaded": result.total_pdfs_downloaded,
            },
            "companies_not_found": result.companies_not_found,
            "companies_no_transcripts": result.companies_no_transcripts,
            "companies_with_errors": result.companies_with_errors,
            "per_company": self._checkpoint.get("company_results", {}),
        }

        report_path = self.dest_dir / "screener_scrape_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info(f"Report saved to {report_path}")

        # Print summary
        print(f"\n{'='*60}")
        print(f"  Screener.in Transcript Scraper -- Summary")
        print(f"{'='*60}")
        print(f"  Companies searched:        {result.total_companies}")
        print(f"  Companies resolved:        {result.companies_resolved}")
        print(f"  Companies w/ transcripts:  {result.companies_with_transcripts}")
        print(f"  Total transcripts found:   {result.total_transcripts_found}")
        print(f"  Total PDFs downloaded:     {result.total_pdfs_downloaded}")
        print(f"{'='*60}")

        if result.companies_not_found:
            print(f"\n[!] Companies NOT found on screener.in ({len(result.companies_not_found)}):")
            for c in result.companies_not_found:
                print(f"    - {c}")

        if result.companies_discarded:
            print(f"\n[!] Companies DISCARDED (fewer than {MIN_TRANSCRIPTS} transcripts, {len(result.companies_discarded)}):")
            for c in result.companies_discarded:
                print(f"    - {c}")

        if result.companies_no_transcripts:
            print(f"\n[!] Companies with NO transcripts in {self.start_year}-{self.end_year} ({len(result.companies_no_transcripts)}):")
            for c in result.companies_no_transcripts:
                print(f"    - {c}")

    def save_manifest(self, result: ScrapeResult, filepath=None):
        """Save a manifest compatible with the old scraper interface."""
        fp = Path(filepath or (self.dest_dir / "scrape_manifest.json"))
        fp.write_text(json.dumps({
            "scraped_at": datetime.now().isoformat(),
            "total_found": result.total_transcripts_found,
            "pdfs_downloaded": result.total_pdfs_downloaded,
            "errors": result.errors,
            "companies_not_found": result.companies_not_found,
            "companies_no_transcripts": result.companies_no_transcripts,
            "companies_discarded": result.companies_discarded,
        }, indent=2, default=str), encoding="utf-8")
        logger.info(f"Saved manifest to {fp}")


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Screener.in Transcript Scraper")
    parser.add_argument(
        "--companies", type=str, default=str(COMPANIES_FILE),
        help=f"Path to companies text file (default: {COMPANIES_FILE})"
    )
    parser.add_argument(
        "--start-year", type=int, default=2022,
        help="Start year for transcript filter (default: 2022)"
    )
    parser.add_argument(
        "--end-year", type=int, default=2025,
        help="End year for transcript filter (default: 2025)"
    )
    parser.add_argument(
        "--dest", type=str, default=None,
        help=f"Destination directory for PDFs (default: {SCREENER_PDF_DIR})"
    )
    args = parser.parse_args()

    scraper = ScreenerScraper(
        dest_dir=Path(args.dest) if args.dest else None,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    result = scraper.scrape_all_companies(
        companies_file=Path(args.companies),
    )
    scraper.save_manifest(result)
