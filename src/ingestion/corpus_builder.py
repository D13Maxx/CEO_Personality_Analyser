"""
Corpus builder for the CEO Personality Analyzer.

Processes all PDF transcripts in a directory, aggregates speech per
executive across multiple files, applies the minimum word threshold,
and saves the final corpus as JSON + Parquet.

Uses JSONL append-only checkpointing so processing can be safely
interrupted and resumed. Optional Colab local-copy mode avoids
the Drive FUSE bottleneck.
"""

import os
import json
import logging
import shutil
from pathlib import Path
from typing import Dict
from datetime import datetime
from collections import defaultdict

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import SCREENER_PDF_DIR, PROCESSED_SCREENER_DIR, MIN_WORDS_PER_CEO
from src.ingestion.transcript_parser import parse_transcript

logger = logging.getLogger(__name__)

CHECKPOINT_FILE = "_parse_results.jsonl"


class CorpusBuilder:
    """Builds an aggregated executive speech corpus from raw PDF transcripts."""

    def __init__(self, raw_dir=SCREENER_PDF_DIR, out_dir=PROCESSED_SCREENER_DIR):
        self.raw_dir = Path(raw_dir)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.out_dir / CHECKPOINT_FILE

        self.exec_data: Dict[str, Dict] = defaultdict(lambda: {
            "executive_name": "",
            "executive_title": "",
            "company_names": set(),
            "quarters": set(),
            "total_words": 0,
            "text_segments": [],
            "source_files": [],
        })

    def _load_checkpoint(self):
        """Rebuild in-memory state from the JSONL checkpoint."""
        already_done = set()
        stats = self._fresh_stats()

        if not self.checkpoint_path.exists():
            return already_done, stats

        with open(self.checkpoint_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Corrupt checkpoint line %d, skipping", line_num)
                    continue

                already_done.add(rec.get("filename", ""))
                stats["total_files_processed"] += 1

                if not rec.get("executive_name") or not rec.get("speech_text"):
                    stats["files_no_match"] += 1
                    continue

                stats["files_with_speech"] += 1
                stats["total_words_extracted"] += rec.get("word_count", 0)

                name = rec["executive_name"]
                entry = self.exec_data[name]
                entry["executive_name"] = name
                entry["executive_title"] = rec.get("executive_title", "")
                entry["company_names"].add(rec.get("company_name", ""))
                entry["quarters"].add(rec.get("date_str", "UNKNOWN"))
                entry["total_words"] += rec.get("word_count", 0)
                entry["text_segments"].append(rec.get("speech_text", ""))
                entry["source_files"].append(rec["filename"])

        logger.info("Checkpoint: %d files done, %d executives accumulated",
                     len(already_done), len(self.exec_data))
        return already_done, stats

    def _write_checkpoint(self, filename, result, date_str):
        if result is None or not result.executive_name or not result.speech_text:
            rec = {"filename": filename, "executive_name": "", "speech_text": ""}
        else:
            rec = {
                "filename": filename,
                "executive_name": " ".join(w.capitalize() for w in result.executive_name.split()),
                "executive_title": result.executive_title,
                "company_name": result.company_name,
                "speech_text": result.speech_text,
                "word_count": result.word_count,
                "date_str": date_str,
            }
        with open(self.checkpoint_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    @staticmethod
    def _fresh_stats():
        return {
            "total_files_processed": 0,
            "files_with_speech": 0,
            "files_no_match": 0,
            "total_execs_found": 0,
            "execs_meeting_threshold": 0,
            "total_words_extracted": 0,
        }

    @staticmethod
    def copy_to_local(source_dir, local_dir="/content/local_pdfs"):
        """Copy PDFs from Drive to Colab local disk (avoids FUSE latency)."""
        local = Path(local_dir)
        local.mkdir(parents=True, exist_ok=True)

        source_pdfs = [f for f in os.listdir(source_dir) if f.lower().endswith(".pdf")]
        if len(list(local.glob("*.pdf"))) >= len(source_pdfs):
            logger.info("Local copy already complete (%d PDFs)", len(source_pdfs))
            return local

        logger.info("Copying %d PDFs to %s ...", len(source_pdfs), local)
        for name in tqdm(source_pdfs, desc="Copying"):
            dst = local / name
            if not dst.exists():
                shutil.copy2(str(Path(source_dir) / name), str(dst))
        return local

    def process_directory(self, resume=True):
        """Process all PDFs and build the corpus."""
        try:
            all_names = os.listdir(self.raw_dir)
        except OSError as e:
            logger.error("Cannot list %s: %s", self.raw_dir, e)
            return {}

        pdf_files = sorted(self.raw_dir / n for n in all_names if n.lower().endswith(".pdf"))
        logger.info("Found %d PDFs in %s", len(pdf_files), self.raw_dir)

        already_done, stats = self._load_checkpoint() if resume else (set(), self._fresh_stats())

        remaining = [p for p in pdf_files if p.name not in already_done]
        logger.info("Remaining: %d / %d", len(remaining), len(pdf_files))

        if not remaining:
            self._save_corpus(stats)
            return stats

        for pdf_path in tqdm(remaining, desc="Processing"):
            self._process_one(pdf_path, stats)

        self._save_corpus(stats)
        logger.info("Processing complete.")
        return stats

    def _process_one(self, pdf_path, stats):
        stats["total_files_processed"] += 1

        try:
            result = parse_transcript(pdf_path)
        except Exception as e:
            logger.error("Exception parsing %s: %s", pdf_path.name, e)
            self._write_checkpoint(pdf_path.name, None, "UNKNOWN")
            return

        # Guess year from filename tokens like '2024'
        year = "UNKNOWN"
        for token in pdf_path.stem.split("_"):
            if len(token) == 4 and token.isdigit():
                year = token
                break

        self._write_checkpoint(pdf_path.name, result, year)

        if result is None or not result.executive_name or not result.speech_text:
            stats["files_no_match"] += 1
            return

        normalized_name = " ".join(w.capitalize() for w in result.executive_name.split())
        stats["files_with_speech"] += 1
        stats["total_words_extracted"] += result.word_count

        entry = self.exec_data[normalized_name]
        entry["executive_name"] = normalized_name
        entry["executive_title"] = result.executive_title
        entry["company_names"].add(result.company_name)
        entry["quarters"].add(year)
        entry["total_words"] += result.word_count
        entry["text_segments"].append(result.speech_text)
        entry["source_files"].append(pdf_path.name)

    def _save_corpus(self, stats):
        passing = 0
        records = []

        for name, entry in self.exec_data.items():
            stats["total_execs_found"] = stats.get("total_execs_found", 0) + 1
            combined_text = "\n\n".join(entry["text_segments"])

            rec = {
                "ceo_name": name,
                "executive_title": entry.get("executive_title", ""),
                "company_names": list(entry["company_names"]),
                "quarters_analyzed": len(entry["quarters"]),
                "total_words": entry["total_words"],
                "meets_threshold": entry["total_words"] >= MIN_WORDS_PER_CEO,
                "text": combined_text,
                "source_files": entry["source_files"],
            }
            records.append(rec)

            if rec["meets_threshold"]:
                passing += 1
                safe_name = "".join(c if c.isalnum() else "_" for c in name).strip("_")
                with open(self.out_dir / f"{safe_name}.json", "w", encoding="utf-8") as f:
                    json.dump(rec, f, indent=2)

        stats["execs_meeting_threshold"] = passing

        if records:
            df = pd.DataFrame(records)
            pq.write_table(pa.Table.from_pandas(df), self.out_dir / "corpus_index.parquet")
            with open(self.out_dir / "corpus_summary.json", "w", encoding="utf-8") as f:
                json.dump({"generated_at": datetime.now().isoformat(),
                           "statistics": stats,
                           "min_words_required": MIN_WORDS_PER_CEO}, f, indent=2)

        logger.info("Corpus: %d executives met %d-word threshold (%d words, %d files)",
                     passing, MIN_WORDS_PER_CEO,
                     stats.get("total_words_extracted", 0),
                     stats.get("files_with_speech", 0))


def build_corpus(raw_dir=None, out_dir=None, local_copy=False, resume=True):
    raw = Path(raw_dir or SCREENER_PDF_DIR)
    out = Path(out_dir or PROCESSED_SCREENER_DIR)
    if local_copy:
        raw = CorpusBuilder.copy_to_local(raw)
    return CorpusBuilder(raw_dir=raw, out_dir=out).process_directory(resume=resume)
