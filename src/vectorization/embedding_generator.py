"""
Document embedding generator using Sentence Transformers.

Produces a single dense vector per executive by chunking their
aggregated speech, encoding each chunk, and mean-pooling.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import PROCESSED_DIR, EMBEDDINGS_DIR, TRANSFORMER_MODEL_NAME, EMBEDDING_DIM
from src.vectorization.text_preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)


class EmbeddingGenerator:

    def __init__(self, processed_dir=PROCESSED_DIR, output_dir=EMBEDDINGS_DIR):
        self.processed_dir = Path(processed_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.output_dir / "ceo_embeddings.parquet"
        self.preprocessor = TextPreprocessor()
        self.model = None

    def _ensure_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading model: %s", TRANSFORMER_MODEL_NAME)
            self.model = SentenceTransformer(TRANSFORMER_MODEL_NAME)

    def load_corpus(self) -> Dict[str, str]:
        """Load valid executive transcripts from the corpus index."""
        index = self.processed_dir / "corpus_index.parquet"
        if index.exists():
            df = pd.read_parquet(index)
            valid = df[df["meets_threshold"] == True]
            return {row["ceo_name"]: row["text"] for _, row in valid.iterrows()}

        corpus = {}
        for p in self.processed_dir.glob("*.json"):
            if p.name == "corpus_summary.json":
                continue
            try:
                with open(p, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    if d.get("meets_threshold"):
                        corpus[d["ceo_name"]] = d["text"]
            except Exception as e:
                logger.error("Error loading %s: %s", p, e)
        return corpus

    def generate_embeddings(self) -> Optional[pd.DataFrame]:
        corpus = self.load_corpus()
        if not corpus:
            logger.error("No corpus found.")
            return None

        self._ensure_model()
        records = []

        logger.info("Encoding %d executives ...", len(corpus))
        for name, text in tqdm(corpus.items(), desc="Encoding"):
            chunks = self.preprocessor.chunk_text(text, chunk_size=5)
            if not chunks:
                vec = np.zeros(EMBEDDING_DIM)
            else:
                chunk_vecs = self.model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
                vec = np.mean(chunk_vecs, axis=0)
            records.append({"ceo_name": name, "embedding": vec.tolist()})

        df = pd.DataFrame(records)
        pq.write_table(pa.Table.from_pandas(df), self.output_path)
        logger.info("Saved embeddings to %s", self.output_path)
        return df

    def embed_single_transcript(self, text: str) -> Optional[np.ndarray]:
        """Embed one transcript at inference time."""
        self._ensure_model()
        chunks = self.preprocessor.chunk_text(text, chunk_size=5)
        if not chunks:
            return np.zeros(EMBEDDING_DIM)
        chunk_vecs = self.model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
        return np.mean(chunk_vecs, axis=0)


def generate_pipeline():
    EmbeddingGenerator().generate_embeddings()
