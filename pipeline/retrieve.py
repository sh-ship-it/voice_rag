"""Dense (FAISS), Sparse (BM25), and Hybrid retrieval engine with Reciprocal Rank Fusion (RRF).

Loads pre-built indices from /index/ at module startup for instant warm query execution.
Resolves small-to-big chunk strategies to parent passages for expanded LLM context.
"""

from __future__ import annotations

import os
import pickle
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from pipeline.config import get_settings
from pipeline.normalize import tokenize_for_bm25, normalize_text
from pipeline.schemas import (
    Chunk,
    RetrievalResult,
    RetrievalStrategy,
    ScoredChunk,
)

# ---------------------------------------------------------------------------
# Global index state container
# ---------------------------------------------------------------------------

class IndexRegistry:
    """Manages warm in-memory index structures (FAISS, BM25, and metadata)."""

    def __init__(self, index_dir: Optional[Path] = None) -> None:
        self.settings = get_settings()
        self.index_dir = index_dir or self.settings.index_dir
        self.faiss_index = None
        self.bm25_index = None
        self.chunk_list: List[Chunk] = []
        self.chunk_map: Dict[str, Chunk] = {}
        self.is_loaded = False
        self.load_indices()

    def load_indices(self) -> bool:
        """Load FAISS index, BM25 index, and chunk mappings from disk."""
        faiss_path = self.index_dir / "faiss_hnswflat.index"
        bm25_path = self.index_dir / "bm25.pkl"
        chunk_list_path = self.index_dir / "chunk_list.pkl"
        chunk_map_path = self.index_dir / "chunk_metadata.pkl"

        # Check if index files exist in nested subfolder
        nested_dir = self.index_dir / "index"
        if (nested_dir / "faiss_hnswflat.index").exists():
            faiss_path = nested_dir / "faiss_hnswflat.index"
            bm25_path = nested_dir / "bm25.pkl"
            chunk_list_path = nested_dir / "chunk_list.pkl"
            chunk_map_path = nested_dir / "chunk_metadata.pkl"

        if not (faiss_path.exists() and bm25_path.exists() and chunk_list_path.exists()):
            # Auto-download 380k index from Hugging Face Dataset
            try:
                import os
                from huggingface_hub import snapshot_download
                token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
                repo_id = os.environ.get("INDEX_DATASET_REPO", "shubham918748/voice-rag-index")
                print(f"[pipeline.retrieve] Auto-downloading 380k index from Hugging Face Dataset '{repo_id}' ...", flush=True)
                download_dir = snapshot_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    local_dir=str(self.index_dir),
                    token=token,
                )
                if (nested_dir / "faiss_hnswflat.index").exists():
                    faiss_path = nested_dir / "faiss_hnswflat.index"
                    bm25_path = nested_dir / "bm25.pkl"
                    chunk_list_path = nested_dir / "chunk_list.pkl"
                    chunk_map_path = nested_dir / "chunk_metadata.pkl"
                elif (self.index_dir / "faiss_hnswflat.index").exists():
                    faiss_path = self.index_dir / "faiss_hnswflat.index"
                    bm25_path = self.index_dir / "bm25.pkl"
                    chunk_list_path = self.index_dir / "chunk_list.pkl"
                    chunk_map_path = self.index_dir / "chunk_metadata.pkl"
            except Exception as dl_err:
                print(f"[pipeline.retrieve] Auto-download notice: {dl_err}", flush=True)

        if not (faiss_path.exists() and bm25_path.exists() and chunk_list_path.exists()):
            self._init_in_memory_fallback_index()
            return True

        try:
            import faiss
            import sys
            import __main__
            from pipeline.schemas import Chunk, ChunkMetadata

            # Ensure compatibility with pickles generated in Kaggle notebooks (__main__.Chunk)
            if not hasattr(__main__, "Chunk"):
                setattr(__main__, "Chunk", Chunk)
            if not hasattr(__main__, "ChunkMetadata"):
                setattr(__main__, "ChunkMetadata", ChunkMetadata)

            class NotebookCompatibleUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if name in ("Chunk", "ChunkMetadata"):
                        from pipeline import schemas
                        return getattr(schemas, name)
                    return super().find_class(module, name)

            print(f"[pipeline.retrieve] Loading FAISS index from {faiss_path} ...", flush=True)
            self.faiss_index = faiss.read_index(str(faiss_path))
            if hasattr(self.faiss_index, "hnsw"):
                self.faiss_index.hnsw.efSearch = 64

            print(f"[pipeline.retrieve] Loading BM25 index from {bm25_path} ...", flush=True)
            with open(bm25_path, "rb") as f:
                self.bm25_index = NotebookCompatibleUnpickler(f).load()

            print(f"[pipeline.retrieve] Loading chunk metadata from {chunk_list_path} ...", flush=True)
            with open(chunk_list_path, "rb") as f:
                self.chunk_list = NotebookCompatibleUnpickler(f).load()

            if chunk_map_path.exists():
                try:
                    with open(chunk_map_path, "rb") as f:
                        self.chunk_map = NotebookCompatibleUnpickler(f).load()
                except Exception:
                    self.chunk_map = {c.chunk_id: c for c in self.chunk_list}
            else:
                self.chunk_map = {c.chunk_id: c for c in self.chunk_list}

            self.is_loaded = True
            print(f"[pipeline.retrieve] Loaded {len(self.chunk_list):,} chunks into warm retrieval memory.", flush=True)
            if not self.bm25_index and len(self.chunk_list) <= 50000:
                self._build_inverted_bm25()
            return True
        except Exception as e:
            print(f"[pipeline.retrieve] Warning: Failed to load index files: {e}", flush=True)
            self._init_in_memory_fallback_index()
            return True

    def _build_inverted_bm25(self):
        """Build in-memory inverted index for sub-millisecond BM25 sparse search."""
        import math
        from collections import defaultdict

        N = len(self.chunk_list)
        if N == 0:
            return
        self.doc_lens = [len(c.text.split()) for c in self.chunk_list]
        self.avgdl = sum(self.doc_lens) / max(1, N)
        self.k1 = 1.5
        self.b = 0.75

        inv = defaultdict(dict)
        for doc_id, c in enumerate(self.chunk_list):
            # Apply Hindi-aware NFC normalization before tokenizing
            tokens = tokenize_for_bm25(c.text)
            counts = defaultdict(int)
            for t in tokens:
                counts[t] += 1
            for t, cnt in counts.items():
                inv[t][doc_id] = cnt

        self.inv_index = dict(inv)
        # Recompute doc_lens using the new tokenizer for BM25 consistency
        self.doc_lens = [len(tokenize_for_bm25(c.text)) for c in self.chunk_list]
        self.avgdl = sum(self.doc_lens) / max(1, len(self.doc_lens))
        self.idf = {}
        for t, doc_dict in self.inv_index.items():
            df = len(doc_dict)
            self.idf[t] = math.log(1 + (len(self.chunk_list) - df + 0.5) / (df + 0.5))

    def fast_bm25_search(self, tokens: List[str], top_k: int = 10) -> List[Tuple[int, float]]:
        """Perform sub-5ms inverted BM25 search with Hindi-aware NFC normalized tokens."""
        if not hasattr(self, "inv_index") or not self.inv_index:
            if self.bm25_index:
                scores = self.bm25_index.get_scores(tokens)
                top_idx = np.argsort(-scores)[:top_k]
                return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0]
            return []

        scores: Dict[int, float] = {}
        for t in tokens:
            # Apply same normalization as indexing
            t_norm = t.lower() if t.isascii() else t
            if t_norm not in self.inv_index:
                continue
            term_idf = self.idf[t_norm]
            for doc_id, freq in self.inv_index[t_norm].items():
                dl = self.doc_lens[doc_id]
                denom = freq + self.k1 * (1 - self.b + self.b * (dl / self.avgdl))
                scores[doc_id] = scores.get(doc_id, 0.0) + term_idf * (freq * (self.k1 + 1.0)) / denom

        if not scores:
            return []
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def _init_in_memory_fallback_index(self):
        """Initialize high-quality in-memory index for immediate testing while disk index builds."""
        if self.is_loaded and self.faiss_index is not None and self.bm25_index is not None:
            return

        import faiss
        from rank_bm25 import BM25Okapi
        from pipeline.embed import embed_query

        sample_passages = [
            ("chunk_fixed_p1", "नई दिल्ली भारत की आधिकारिक राजधानी है। यह भारत सरकार के तीनों अंगों - कार्यपालिका, विधायिका और न्यायपालिका का केंद्र है। राष्ट्रपति भवन, संसद भवन और सर्वोच्च न्यायालय यहीं स्थित हैं।", "भारत की राजधानी नई दिल्ली का इतिहास और प्रशासनिक महत्व।"),
            ("chunk_fixed_p2", "New Delhi is the official capital of India. It serves as the seat of all three branches of the Government of India: Executive, Legislative, and Judiciary, including Rashtrapati Bhavan and Parliament.", "Capital of India overview in English."),
            ("chunk_fixed_p3", "कंप्यूटर और इंटरनेट आधुनिक जीवन के सबसे महत्वपूर्ण साधन हैं। इंटरनेट के माध्यम से दुनिया भर की जानकारी, डिजिटल शिक्षा, व्यापार और संचार कुछ ही सेकंड में उपलब्ध हो जाता है।", "कंप्यूटर और इंटरनेट के मुख्य लाभ और उपयोग।"),
            ("chunk_fixed_p4", "Computers and the Internet are essential pillars of modern digital life. They enable rapid global communication, online education, e-commerce, and high-speed data processing.", "Benefits of computers and internet in English."),
            ("chunk_fixed_p5", "ताजमहल भारत के उत्तर प्रदेश राज्य के आगरा शहर में स्थित एक विश्व प्रसिद्ध सफेद संगमरमर का मकबरा है। इसका निर्माण मुगल सम्राट शाहजहाँ ने अपनी प्रिय पत्नी मुमताज़ महल की याद में करवाया था।", "ताजमहल का इतिहास और निर्माण।"),
            ("chunk_fixed_p6", "The Taj Mahal is a world-renowned white marble mausoleum located in Agra, Uttar Pradesh, India. It was commissioned by Mughal Emperor Shah Jahan in memory of his wife Mumtaz Mahal.", "Taj Mahal history in English."),
            ("chunk_fixed_p7", "भारतीय संविधान 26 जनवरी 1950 को लागू हुआ था। डॉ. भीमराव अंबेडकर को भारतीय संविधान का जनक माना जाता है। भारत दुनिया का सबसे बड़ा लिखित संविधान वाला लोकतांत्रिक देश है।", "भारतीय संविधान और डॉ. भीमराव अंबेडकर।"),
            ("chunk_fixed_p8", "The Constitution of India came into effect on 26 January 1950. Dr. B.R. Ambedkar is recognized as the chief architect and father of the Indian Constitution.", "Indian Constitution in English."),
            ("chunk_fixed_p9", "हमारे सौर मंडल का सबसे बड़ा ग्रह बृहस्पति (Jupiter) है। सौर मंडल में आठ मुख्य ग्रह हैं जिनमें बुध, शुक्र, पृथ्वी, मंगल, बृहस्पति, शनि, यूरेनस और नेपच्यून शामिल हैं।", "सौर मंडल और बृहस्पति ग्रह।"),
            ("chunk_fixed_p10", "Jupiter is the largest planet in our Solar System. The solar system comprises eight major planets orbiting the Sun, with Jupiter having the largest mass and volume.", "Solar system and Jupiter in English."),
            ("chunk_fixed_p11", "निगम (Corporation) एक कानूनी इकाई या कंपनी होती है जो अपने मालिकों और शेयरधारकों से अलग अस्तित्व रखती है। यह अनुबंध करने, संपत्ति रखने और मुकदमा दायर करने की क्षमता रखती है।", "कॉर्पोरेशन और कंपनी की परिभाषा।"),
            ("chunk_fixed_p12", "A corporation is an organization, usually a group of people or a company, authorized by the state to act as a single legal entity recognized in law for particular purposes.", "Corporation definition in English."),
            ("chunk_fixed_p13", "Artificial Intelligence (AI) aur Machine Learning (ML) mein main antar yeh hai ki AI ek broader concept hai jiska uddeshya smart machines banana hai, jabki ML AI ka ek subset hai jo algorithms ko data se seekhne mein madad karta hai.", "AI vs ML in Hinglish."),
            ("chunk_fixed_p14", "प्रकाश संश्लेषण (Photosynthesis) वह प्रक्रिया है जिसके द्वारा हरे पौधे सूर्य के प्रकाश, जल और कार्बन डाइऑक्साइड का उपयोग करके अपना भोजन (ग्लूकोज) बनाते हैं और ऑक्सीजन छोड़ते हैं।", "प्रकाश संश्लेषण की प्रक्रिया।"),
            ("chunk_fixed_p15", "Photosynthesis is the process used by plants and other organisms to convert light energy into chemical energy, synthesizing sugars and releasing oxygen from water and carbon dioxide.", "Photosynthesis overview in English."),
        ]

        self.chunk_list = []
        self.chunk_map = {}
        embeddings = []
        tokenized_corpus = []

        for cid, text, ptext in sample_passages:
            chunk = Chunk(
                chunk_id=cid,
                doc_id="doc_sample",
                source_passage_id=cid.split("_")[-1],
                text=text,
                parent_text=ptext,
                chunk_strategy="fixed_size",
                token_count=len(text.split()),
                character_count=len(text),
                byte_count=len(text.encode("utf-8")),
            )
            self.chunk_list.append(chunk)
            self.chunk_map[cid] = chunk
            emb = embed_query(text)
            embeddings.append(emb)
            tokenized_corpus.append(text.lower().split())

        dim = 384
        self.faiss_index = faiss.IndexHNSWFlat(dim, 16, faiss.METRIC_INNER_PRODUCT)
        self.faiss_index.hnsw.efConstruction = 100
        self.faiss_index.hnsw.efSearch = 64
        emb_matrix = np.vstack(embeddings).astype(np.float32)
        self.faiss_index.add(emb_matrix)

        self.bm25_index = BM25Okapi(tokenized_corpus)
        self.is_loaded = True
        print(f"[pipeline.retrieve] Initialized warm fallback index with {len(self.chunk_list)} bilingual chunks.", flush=True)


# Global warm registry
_REGISTRY = IndexRegistry()


def get_index_registry() -> IndexRegistry:
    """Access the global IndexRegistry instance."""
    return _REGISTRY


def warmup() -> None:
    """Explicitly ensure indices and warm embedding models are initialized."""
    _ = _REGISTRY
    from pipeline.embed import embed_query
    embed_query("warmup")


# ---------------------------------------------------------------------------
# Core Hybrid Retrieval function
# ---------------------------------------------------------------------------

def hybrid_retrieve(
    query_embedding: np.ndarray,
    query_text: str,
    top_k: int = 6,
    dense_candidates: int = 50,
    sparse_candidates: int = 50,
    rrf_k: int = 60,
    enable_rerank: bool = True,
    registry: Optional[IndexRegistry] = None,
) -> RetrievalResult:
    """Execute hybrid retrieval combining FAISS HNSW dense search and BM25 sparse search.

    Per HHGOA Task 2 architecture:
    1. Run FAISS and BM25 in parallel (ThreadPoolExecutor).
    2. Retrieve top-50 candidates from each.
    3. Fuse with RRF, deduplicate by parent_id.
    4. Rerank with cross-encoder on top-30 candidates.
    5. Small-to-big context expansion on final top_k.

    Parameters
    ----------
    query_embedding:
        1D or 2D numpy array containing the L2-normalized query vector.
    query_text:
        Raw query text string for lexical BM25 tokenization.
    top_k:
        Number of final chunks to return after reranking.
    dense_candidates:
        Top candidates fetched from dense index (default: 50).
    sparse_candidates:
        Top candidates fetched from sparse index (default: 50).
    rrf_k:
        RRF constant parameter (default: 60).
    enable_rerank:
        If True, apply cross-encoder reranker to top-30 RRF candidates.
    registry:
        Optional IndexRegistry override.
    """
    from concurrent.futures import ThreadPoolExecutor
    from pipeline.normalize import tokenize_for_bm25, build_transliteration_aliases

    t0 = time.perf_counter()
    reg = registry or _REGISTRY

    # Fallback if index is not loaded on disk yet
    if not reg.is_loaded or reg.faiss_index is None or reg.bm25_index is None or not reg.chunk_list:
        # Try reloading once in case index finished building
        if not reg.load_indices():
            latency = (time.perf_counter() - t0) * 1000.0
            return RetrievalResult(
                query=query_text,
                chunks=[],
                strategy_used=RetrievalStrategy.HYBRID,
                total_candidates_evaluated=0,
                latency_ms=latency,
            )

    total_chunks = len(reg.chunk_list)
    k_dense = min(dense_candidates, total_chunks)
    k_sparse = min(sparse_candidates, total_chunks)

    # ------------------------------------------------------------------
    # 1. Dense FAISS Search + 2. Sparse BM25 Search — run in PARALLEL
    # Per architecture doc: "Run FAISS and BM25 in parallel."
    # ------------------------------------------------------------------
    from concurrent.futures import ThreadPoolExecutor
    from pipeline.normalize import tokenize_for_bm25, build_transliteration_aliases

    dense_ranks: Dict[int, int] = {}
    dense_scores: Dict[int, float] = {}
    sparse_ranks: Dict[int, int] = {}
    sparse_scores: Dict[int, float] = {}

    if hasattr(reg.faiss_index, "hnsw"):
        reg.faiss_index.hnsw.efSearch = 64

    q_vec = np.ascontiguousarray(query_embedding, dtype=np.float32)
    if q_vec.ndim == 1:
        q_vec = q_vec.reshape(1, -1)

    def _dense_search():
        distances, indices = reg.faiss_index.search(q_vec, k_dense)
        d_ranks: Dict[int, int] = {}
        d_scores: Dict[int, float] = {}
        if len(indices) > 0:
            for rank_idx, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
                if idx >= 0 and idx < total_chunks:
                    d_ranks[int(idx)] = rank_idx
                    d_scores[int(idx)] = float(dist)
        return d_ranks, d_scores

    def _sparse_search():
        # NFC-normalized + Hindi-aware tokenization
        tokens = tokenize_for_bm25(query_text)
        # Add transliteration aliases for Hinglish queries
        tokens += build_transliteration_aliases(query_text)
        s_ranks: Dict[int, int] = {}
        s_scores: Dict[int, float] = {}
        if tokens:
            top_sparse = reg.fast_bm25_search(tokens, top_k=k_sparse)
            for rank_idx, (idx_int, score) in enumerate(top_sparse, start=1):
                if idx_int < total_chunks and score > 0.0:
                    s_ranks[idx_int] = rank_idx
                    s_scores[idx_int] = float(score)
        return s_ranks, s_scores

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_dense = pool.submit(_dense_search)
        fut_sparse = pool.submit(_sparse_search)
        dense_ranks, dense_scores = fut_dense.result()
        sparse_ranks, sparse_scores = fut_sparse.result()

    # ------------------------------------------------------------------
    # 3. Reciprocal Rank Fusion (RRF)
    # ------------------------------------------------------------------
    all_candidate_indices = set(dense_ranks.keys()).union(sparse_ranks.keys())
    rrf_scores: List[Tuple[int, float]] = []

    for idx in all_candidate_indices:
        score = 0.0
        if idx in dense_ranks:
            score += 1.0 / (rrf_k + dense_ranks[idx])
        if idx in sparse_ranks:
            score += 1.0 / (rrf_k + sparse_ranks[idx])
        rrf_scores.append((idx, score))

    rrf_scores.sort(key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------
    # 4. Deduplicate by parent_id (architecture doc requirement)
    # "Fuse the results, deduplicate by parent_id"
    # ------------------------------------------------------------------
    seen_parent_ids: set = set()
    deduped_rrf: List[Tuple[int, float]] = []
    for idx, score in rrf_scores:
        chunk = reg.chunk_list[idx]
        parent_id = getattr(chunk, "source_passage_id", None) or chunk.doc_id
        if parent_id not in seen_parent_ids:
            seen_parent_ids.add(parent_id)
            deduped_rrf.append((idx, score))

    # ------------------------------------------------------------------
    # 5. Cross-encoder reranking (top-30 → top-6)
    # Architecture doc: "Retrieve top 40–50, fuse, deduplicate, rerank 50–80."
    # "If too slow, rerank only top 30 or top 15."
    # ------------------------------------------------------------------
    reranked = deduped_rrf
    if enable_rerank and len(deduped_rrf) > 0:
        reranked = _rerank_candidates(query_text, deduped_rrf, reg, max_candidates=30)

    # ------------------------------------------------------------------
    # 6. Build ScoredChunks — compact evidence window (not full parent_text)
    # Architecture doc: "LLM should not receive five large parent passages.
    # Receive four to six compact evidence windows."
    # ------------------------------------------------------------------
    scored_chunks: List[ScoredChunk] = []

    for rank, (idx, score) in enumerate(reranked[:top_k], start=1):
        orig_chunk = reg.chunk_list[idx]

        # Use chunk text directly (not 512-token parent_text) per architecture doc.
        # parent_text is kept in the chunk for citation display only.
        final_text = orig_chunk.text

        chunk_copy = Chunk(
            chunk_id=orig_chunk.chunk_id,
            doc_id=orig_chunk.doc_id,
            text=final_text,
            metadata=orig_chunk.metadata.model_copy(deep=True),
            embedding=orig_chunk.embedding,
            source_passage_id=orig_chunk.source_passage_id,
            language=orig_chunk.language,
            chunk_strategy=orig_chunk.chunk_strategy,
            token_count=orig_chunk.token_count,
            parent_text=orig_chunk.parent_text,
        )

        # Keep original sentence text accessible for evidence window builder
        if orig_chunk.chunk_strategy == "small_to_big" and orig_chunk.parent_text:
            chunk_copy.metadata.extra["sentence_text"] = orig_chunk.text

        scored_chunks.append(
            ScoredChunk(
                chunk=chunk_copy,
                score=score,
                rank=rank,
                retrieval_strategy=RetrievalStrategy.HYBRID,
            )
        )

    latency = (time.perf_counter() - t0) * 1000.0

    return RetrievalResult(
        query=query_text,
        chunks=scored_chunks,
        strategy_used=RetrievalStrategy.HYBRID,
        total_candidates_evaluated=len(all_candidate_indices),
        latency_ms=latency,
    )


def _rerank_candidates(
    query_text: str,
    candidates: List[Tuple[int, float]],
    reg: "IndexRegistry",
    max_candidates: int = 30,
) -> List[Tuple[int, float]]:
    """Apply cross-encoder reranking to the top candidates.

    Per architecture doc: "Rerank the resulting 50–80 candidates. The reranker
    should receive the query and the candidate passage. If too slow, rerank only
    the top 30 candidates or use a lightweight score before applying cross-encoder
    to the top 15."

    Uses FlashRank (fast ONNX cross-encoder) if available; falls back to
    no-reranking if not installed so the pipeline degrades gracefully.
    """
    if not candidates:
        return candidates

    rerank_pool = candidates[:max_candidates]
    tail = candidates[max_candidates:]

    try:
        from flashrank import Ranker, RerankRequest  # type: ignore

        ranker = Ranker(model_name="ms-marco-MiniLM-L-12-H-384-v1", cache_dir=".cache/flashrank")
        passages = [
            {"id": str(idx), "text": reg.chunk_list[idx].text}
            for idx, _ in rerank_pool
            if idx < len(reg.chunk_list)
        ]
        if not passages:
            return candidates

        rerank_request = RerankRequest(query=query_text, passages=passages)
        rerank_result = ranker.rerank(rerank_request)

        # Rebuild list in reranked order
        id_to_orig_score = {str(idx): score for idx, score in rerank_pool}
        reranked_list: List[Tuple[int, float]] = []
        for item in rerank_result:
            item_id = str(item["id"])
            # Use reranker score as primary, blended with RRF for stability
            reranker_score = float(item.get("score", 0.0))
            orig_score = id_to_orig_score.get(item_id, 0.0)
            blended = 0.7 * reranker_score + 0.3 * orig_score
            reranked_list.append((int(item_id), blended))

        reranked_list.sort(key=lambda x: x[1], reverse=True)
        return reranked_list + tail

    except ImportError:
        # FlashRank not installed — fall back to RRF ordering
        return candidates
    except Exception:
        # Any other error — graceful degradation
        return candidates


# ---------------------------------------------------------------------------
# Class-based retriever interfaces
# ---------------------------------------------------------------------------

class BaseRetriever(ABC):
    """Abstract base class for chunk retrieval."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """Retrieve relevant chunks for a given query."""
        raise NotImplementedError


class FAISSDenseRetriever(BaseRetriever):
    """Dense vector retriever querying the FAISS index."""

    def __init__(self, registry: Optional[IndexRegistry] = None) -> None:
        self.registry = registry or _REGISTRY

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        from pipeline.embed import embed_query
        t0 = time.perf_counter()
        vec = embed_query(query)
        res = hybrid_retrieve(
            query_embedding=vec,
            query_text=query,
            top_k=top_k,
            dense_candidates=top_k,
            sparse_candidates=0,
            registry=self.registry,
        )
        res.strategy_used = RetrievalStrategy.DENSE
        res.latency_ms = (time.perf_counter() - t0) * 1000.0
        return res


class BM25SparseRetriever(BaseRetriever):
    """Sparse lexical retriever querying the BM25 index."""

    def __init__(self, registry: Optional[IndexRegistry] = None) -> None:
        self.registry = registry or _REGISTRY

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        t0 = time.perf_counter()
        dummy_vec = np.zeros(384, dtype=np.float32)
        res = hybrid_retrieve(
            query_embedding=dummy_vec,
            query_text=query,
            top_k=top_k,
            dense_candidates=0,
            sparse_candidates=top_k,
            registry=self.registry,
        )
        res.strategy_used = RetrievalStrategy.SPARSE
        res.latency_ms = (time.perf_counter() - t0) * 1000.0
        return res


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining FAISS dense and BM25 sparse search with RRF."""

    def __init__(self, registry: Optional[IndexRegistry] = None) -> None:
        self.registry = registry or _REGISTRY

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        from pipeline.embed import embed_query
        vec = embed_query(query)
        return hybrid_retrieve(
            query_embedding=vec,
            query_text=query,
            top_k=top_k,
            registry=self.registry,
        )
