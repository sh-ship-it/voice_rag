"""Index-build script for the voice-enabled RAG pipeline.

Usage
-----
Activate the .venv, then run from the project root::

    python -m pipeline.build_index

or with overrides::

    DATASET_ROWS=500 python -m pipeline.build_index

What it does
------------
1. Streams ``ai4bharat/MSMARCO-XI`` (Hindi, "hi" config), taking the first
   ``DATASET_ROWS`` rows from the train split.
2. Flattens all passage strings across those rows, deduplicates them, and
   treats the resulting corpus as the unit to be chunked.
3. Runs all three chunking strategies (fixed_size, semantic, small_to_big)
   via ``pipeline.chunking.run_all_chunkers``.
4. Embeds every chunk's ``text`` field with ``intfloat/multilingual-e5-small``
   using the ``"passage: "`` prefix E5 models require.
5. Builds:
      - a FAISS ``IndexHNSWFlat`` (M=16, efConstruction=100, cosine via IP on
        L2-normalised vectors) for dense retrieval.
      - a ``BM25Okapi`` index (rank_bm25) over raw chunk text.
6. Persists to ``/index/``:
      - ``faiss_hnswflat.index``     -- FAISS binary index
      - ``bm25.pkl``                 -- pickled BM25 object
      - ``chunk_metadata.pkl``       -- ``Dict[str, Chunk]`` (chunk_id -> Chunk)
      - ``chunk_list.pkl``           -- ``List[Chunk]`` preserving order
7. Prints a corpus stats table at the end.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

# Suppress the harmless Windows symlinks warning from huggingface_hub
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np

# ---------------------------------------------------------------------------
# Project-local imports (these are available after ``pip install -r requirements.txt``)
# ---------------------------------------------------------------------------

# Guard: make sure we are running from the project root so relative imports resolve.
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from pipeline.config import get_settings
from pipeline.chunking import run_all_chunkers
from pipeline.schemas import Chunk

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_ID = "ai4bharat/MSMARCO-XI"
LANGUAGE = "hi"
EMBEDDING_PREFIX = "passage: "

# Language -> parquet filename mapping
# MSMARCO-XI contains validation splits (~440MB for Hindi, 97k rows) and train splits (~3.7GB)
# We use validation split which has ample rows (>97,000) for the 3,000 row requirement and downloads in ~2-3 mins.
LANG_TO_FILE: dict = {
    "hi": "validation/hinval.parquet",
    "bn": "validation/benval.parquet",
    "gu": "validation/gujval.parquet",
    "kn": "validation/kanval.parquet",
    "ml": "validation/malval.parquet",
    "mr": "validation/marval.parquet",
    "ne": "validation/nepval.parquet",
    "or": "validation/orival.parquet",
    "pa": "validation/panval.parquet",
    "sa": "validation/sanval.parquet",
    "ta": "validation/tamval.parquet",
    "te": "validation/telval.parquet",
    "ur": "validation/urdval.parquet",
}

# ---------------------------------------------------------------------------
# Step 1 -- Streaming dataset fetch
# ---------------------------------------------------------------------------

def _download_parquet(url: str, local_path: Path, hf_token: Optional[str] = None) -> None:
    """Download a remote parquet file with resume support using curl or urllib."""
    import subprocess
    import shutil

    if shutil.which("curl.exe") or shutil.which("curl"):
        curl_bin = shutil.which("curl.exe") or "curl"
        cmd = [curl_bin, "-L", "-C", "-", "--retry", "5", "--retry-delay", "2", "-o", str(local_path), url]
        if hf_token:
            cmd.extend(["-H", f"Authorization: Bearer {hf_token}"])
        print(f"[data]  Downloading via curl: {url} -> {local_path}")
        res = subprocess.run(cmd)
        if res.returncode != 0:
            raise RuntimeError(f"curl download failed with exit code {res.returncode}")
    else:
        import urllib.request
        print(f"[data]  Downloading via urllib: {url} -> {local_path}")
        req = urllib.request.Request(url)
        if hf_token:
            req.add_header("Authorization", f"Bearer {hf_token}")
        with urllib.request.urlopen(req) as resp, open(local_path, "wb") as out:
            shutil.copyfileobj(resp, out)


def load_passages(
    dataset_id: str,
    language: str,
    n_rows: int,
    data_dir: Path = Path("./data"),
    hf_token: Optional[str] = None,
) -> List[str]:
    """Download the language-specific parquet and stream passages.

    Strategy
    --------
    1. Fetch the language shard from ``ai4bharat/MSMARCO-XI`` into ``data_dir``
       using curl with resume support.
    2. Stream the local parquet with ``load_dataset("parquet", ...)``.
    3. Flatten ``passage_text`` or ``Translated_passages``, deduplicate, and
       return unique passages across the first ``n_rows`` rows.
    """
    from datasets import load_dataset  # type: ignore

    parquet_repo_path = LANG_TO_FILE.get(language)
    if not parquet_repo_path:
        raise ValueError(
            f"Unknown language code {language!r}. "
            f"Known codes: {sorted(LANG_TO_FILE.keys())}"
        )

    print(f"\n{'='*60}")
    print(f"[data]  Dataset : {dataset_id}")
    print(f"[data]  Language: {language!r}  ->  {parquet_repo_path}")
    print(f"[data]  Max rows: {n_rows}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Step 1 -- Download parquet to data_dir (cached on repeat runs)
    # ------------------------------------------------------------------
    data_dir.mkdir(parents=True, exist_ok=True)
    local_parquet = data_dir / parquet_repo_path.replace("/", "_")

    if local_parquet.exists() and local_parquet.stat().st_size > 1_000_000:
        size_mb = local_parquet.stat().st_size / 1e6
        print(f"[data]  Cache hit: {local_parquet}  ({size_mb:.1f} MB)")
    else:
        url = f"https://huggingface.co/datasets/{dataset_id}/resolve/main/{parquet_repo_path}"
        t0 = time.perf_counter()
        _download_parquet(url, local_parquet, hf_token=hf_token)
        elapsed = time.perf_counter() - t0
        size_mb = local_parquet.stat().st_size / 1e6
        print(f"[data]  Downloaded {size_mb:.1f} MB in {elapsed:.1f}s\n")

    # ------------------------------------------------------------------
    # Step 2 -- Stream from local parquet
    # ------------------------------------------------------------------
    print(f"[data]  Streaming passages from local parquet ...")
    ds = load_dataset("parquet", data_files={"train": str(local_parquet)}, split="train", streaming=True)

    seen: dict = {}
    row_count = 0

    for row in ds:
        if row_count >= n_rows:
            break

        passages_field = row.get("passages", {})
        texts: List[str] = []
        if isinstance(passages_field, dict):
            texts = (
                passages_field.get("passage_text")
                or passages_field.get("Translated_passages")
                or passages_field.get("English_passages")
                or []
            )
        elif isinstance(passages_field, list):
            texts = [
                p if isinstance(p, str) else (
                    p.get("passage_text") or p.get("Translated_passages") or "" if isinstance(p, dict) else ""
                )
                for p in passages_field
            ]

        for p in texts:
            if isinstance(p, str):
                p_stripped = p.strip()
                if p_stripped and p_stripped not in seen:
                    seen[p_stripped] = None

        row_count += 1
        if row_count % 500 == 0 or row_count == n_rows:
            print(f"  ... {row_count}/{n_rows} rows  |  unique passages: {len(seen)}")

    unique_passages = list(seen.keys())
    print(f"\n[data]  Done.  Rows: {row_count}  |  Unique passages: {len(unique_passages)}\n")
    return unique_passages


# ---------------------------------------------------------------------------
# Step 2 -- Embedding with multilingual-e5-small
# ---------------------------------------------------------------------------


def load_embedding_model(model_name: str):
    """Load and return a SentenceTransformer model."""
    from sentence_transformers import SentenceTransformer  # type: ignore
    print(f"[embed] Loading model {model_name!r} ...")
    model = SentenceTransformer(model_name)
    print(f"[embed] Model loaded.  Embedding dim = {model.get_sentence_embedding_dimension()}")
    return model


def embed_chunks(
    chunks: List[Chunk],
    model,
    batch_size: int = 256,
    prefix: str = EMBEDDING_PREFIX,
) -> np.ndarray:
    """Encode all chunk texts in batches; returns float32 array (N, D).

    Vectors are **L2-normalised** so that inner-product search in FAISS
    is equivalent to cosine similarity.
    """
    texts = [f"{prefix}{c.text}" for c in chunks]
    n = len(texts)
    print(f"[embed] Encoding {n} chunks in batches of {batch_size} ...")
    t0 = time.perf_counter()

    all_vecs: List[np.ndarray] = []
    for start in range(0, n, batch_size):
        batch = texts[start : start + batch_size]
        vecs = model.encode(
            batch,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        all_vecs.append(np.array(vecs, dtype=np.float32))
        done = min(start + batch_size, n)
        if done % (batch_size * 4) == 0 or done == n:
            elapsed = time.perf_counter() - t0
            print(f"  ... {done}/{n} chunks embedded  ({elapsed:.1f}s)")

    embeddings = np.vstack(all_vecs)
    elapsed = time.perf_counter() - t0
    print(f"[embed] Done in {elapsed:.2f}s.  Shape: {embeddings.shape}\n")
    return embeddings


# ---------------------------------------------------------------------------
# Step 3 -- FAISS HNSW index
# ---------------------------------------------------------------------------

def build_faiss_index(
    embeddings: np.ndarray,
    m: int = 16,
    ef_construction: int = 100,
) -> "faiss.IndexHNSWFlat":  # type: ignore[name-defined]
    """Build and populate a FAISS IndexHNSWFlat for cosine (IP) search.

    Vectors must already be L2-normalised (so inner-product == cosine).
    HNSW does not support remove_ids; the mapping between row ordinal and
    chunk_id is maintained externally via ``chunk_list``.

    Parameters
    ----------
    embeddings:
        L2-normalised float32 array of shape (N, D).
    m:
        HNSW ``M`` hyper-parameter (number of neighbours per layer).
    ef_construction:
        HNSW ``efConstruction`` (graph quality / build cost tradeoff).
    """
    import faiss  # type: ignore

    n, d = embeddings.shape
    print(f"[faiss] Building IndexHNSWFlat  M={m}  efConstruction={ef_construction}  n={n}  d={d} ...")
    t0 = time.perf_counter()

    # faiss.METRIC_INNER_PRODUCT -- since vecs are L2-normed, IP = cosine.
    index = faiss.IndexHNSWFlat(d, m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.add(embeddings)

    elapsed = time.perf_counter() - t0
    print(f"[faiss] Index built in {elapsed:.2f}s.  Total vectors: {index.ntotal}\n")
    return index


# ---------------------------------------------------------------------------
# Step 4 -- BM25 index
# ---------------------------------------------------------------------------

def build_bm25_index(chunks: List[Chunk]):
    """Build a BM25Okapi index over raw (non-prefixed) chunk texts.

    Tokenisation is simple whitespace splitting -- consistent with the
    ``_token_count`` helper in chunking.py.
    """
    from rank_bm25 import BM25Okapi  # type: ignore

    print(f"[bm25]  Tokenising {len(chunks)} chunks ...")
    t0 = time.perf_counter()
    corpus = [c.text.split() for c in chunks]
    bm25 = BM25Okapi(corpus)
    elapsed = time.perf_counter() - t0
    print(f"[bm25]  Index built in {elapsed:.2f}s.\n")
    return bm25


# ---------------------------------------------------------------------------
# Step 5 -- Persistence
# ---------------------------------------------------------------------------

def persist_index(
    index_dir: Path,
    faiss_index,
    bm25_index,
    chunks: List[Chunk],
) -> None:
    """Write FAISS index, BM25 index, and chunk metadata to ``index_dir``."""
    import faiss  # type: ignore

    index_dir.mkdir(parents=True, exist_ok=True)

    # FAISS binary
    faiss_path = index_dir / "faiss_hnswflat.index"
    faiss.write_index(faiss_index, str(faiss_path))
    print(f"[save]  FAISS index  -> {faiss_path}  ({faiss_path.stat().st_size / 1e6:.1f} MB)")

    # BM25
    bm25_path = index_dir / "bm25.pkl"
    with open(bm25_path, "wb") as fh:
        pickle.dump(bm25_index, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[save]  BM25 index   -> {bm25_path}  ({bm25_path.stat().st_size / 1e6:.1f} MB)")

    # chunk_list (ordered, matches FAISS row ordinal)
    chunk_list_path = index_dir / "chunk_list.pkl"
    with open(chunk_list_path, "wb") as fh:
        pickle.dump(chunks, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[save]  chunk_list   -> {chunk_list_path}  ({chunk_list_path.stat().st_size / 1e6:.1f} MB)")

    # chunk_metadata mapping  chunk_id -> Chunk
    metadata_path = index_dir / "chunk_metadata.pkl"
    chunk_map: Dict[str, Chunk] = {c.chunk_id: c for c in chunks}
    with open(metadata_path, "wb") as fh:
        pickle.dump(chunk_map, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[save]  chunk_map    -> {metadata_path}  ({metadata_path.stat().st_size / 1e6:.1f} MB)")

    # Human-readable stats JSON (for quick inspection)
    stats: Dict[str, object] = {
        "total_chunks": len(chunks),
        "by_strategy": {},
    }
    per_strategy: Dict[str, List[int]] = defaultdict(list)
    for c in chunks:
        if c.chunk_strategy and c.token_count is not None:
            per_strategy[c.chunk_strategy].append(c.token_count)
    for strategy, counts in per_strategy.items():
        stats["by_strategy"][strategy] = {  # type: ignore[index]
            "count": len(counts),
            "avg_tokens": round(sum(counts) / len(counts), 1) if counts else 0,
        }
    stats_path = index_dir / "build_stats.json"
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
    print(f"[save]  stats JSON   -> {stats_path}\n")


# ---------------------------------------------------------------------------
# Step 6 -- Print corpus stats
# ---------------------------------------------------------------------------

def print_stats(
    passages: List[str],
    chunks: List[Chunk],
    total_time_s: float,
) -> None:
    per_strategy: Dict[str, List[int]] = defaultdict(list)
    for c in chunks:
        if c.chunk_strategy and c.token_count is not None:
            per_strategy[c.chunk_strategy].append(c.token_count)

    print("\n" + "=" * 60)
    print("  CORPUS & INDEX BUILD STATS")
    print("=" * 60)
    print(f"  Source passages (deduplicated) : {len(passages):>8,}")
    print(f"  Total chunks (all strategies)  : {len(chunks):>8,}")
    print(f"  Total build time               : {total_time_s:>8.1f}s")
    print()
    print(f"  {'Strategy':<20} {'Chunks':>8}  {'Avg tokens':>10}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*10}")
    for strategy in ("fixed_size", "semantic", "small_to_big"):
        counts = per_strategy.get(strategy, [])
        avg = sum(counts) / len(counts) if counts else 0.0
        print(f"  {strategy:<20} {len(counts):>8,}  {avg:>10.1f}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    settings = get_settings()
    settings.ensure_directories()

    total_t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Stream dataset
    # ------------------------------------------------------------------
    passages = load_passages(
        dataset_id=settings.dataset_name,
        language=LANGUAGE,
        n_rows=settings.dataset_rows,
        data_dir=settings.data_dir,
        hf_token=settings.hf_token,
    )

    if not passages:
        print("[ERROR] No passages found.  Check dataset config and HF_TOKEN.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Load embedding model (shared across embedding + semantic chunker)
    # ------------------------------------------------------------------
    model = load_embedding_model(settings.embedding_model_name)

    # ------------------------------------------------------------------
    # 3. Chunk all passages with all three strategies
    # ------------------------------------------------------------------
    print(f"\n[chunk] Running all three chunkers over {len(passages)} passages ...\n")
    chunk_t0 = time.perf_counter()
    chunks = run_all_chunkers(passages, language=LANGUAGE, semantic_model=model)
    chunk_elapsed = time.perf_counter() - chunk_t0
    print(f"\n[chunk] Done.  {len(chunks)} total chunks in {chunk_elapsed:.1f}s.\n")

    # ------------------------------------------------------------------
    # 4. Embed all chunks
    # ------------------------------------------------------------------
    embeddings = embed_chunks(
        chunks,
        model=model,
        batch_size=settings.index_build_batch_size,
    )

    # ------------------------------------------------------------------
    # 5. Build FAISS + BM25 indices
    # ------------------------------------------------------------------
    faiss_index = build_faiss_index(
        embeddings,
        m=settings.hnsw_m,
        ef_construction=settings.hnsw_ef_construction,
    )
    bm25_index = build_bm25_index(chunks)

    # ------------------------------------------------------------------
    # 6. Persist
    # ------------------------------------------------------------------
    print("[save]  Writing index artefacts ...")
    persist_index(settings.index_dir, faiss_index, bm25_index, chunks)

    total_elapsed = time.perf_counter() - total_t0

    # ------------------------------------------------------------------
    # 7. Stats
    # ------------------------------------------------------------------
    print_stats(passages, chunks, total_elapsed)


if __name__ == "__main__":
    main()
