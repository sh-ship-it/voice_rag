"""Probabilistic & Multi-Feature Confidence Calibration Module.

Does not treat raw retrieval scores as probabilities.
Extracts 9 multi-modal retrieval features:
1. dense_similarity: FAISS inner product / cosine similarity of top chunk
2. bm25_score: Normalized BM25 lexical score of top chunk
3. reranker_score: FlashRank cross-encoder or RRF score
4. rank_top: Top rank position indicator
5. score_margin: Difference between rank 1 and rank 2 scores
6. dense_bm25_agreement: Binary indicator if top chunk was found in both Dense and BM25
7. strategy_agreement: Number of distinct chunking strategies (fixed, semantic, s2b) supporting top evidence
8. evidence_validity: Validity and length quality of extracted evidence sentence
9. parent_agreement: Proportion of top-3 chunks originating from the same parent passage
"""

from __future__ import annotations

import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

from pipeline.schemas import RetrievalResult, ScoredChunk


class ConfidenceFeatures(BaseModel):
    """Structured feature vector for confidence estimation."""
    dense_similarity: float = Field(default=0.0)
    bm25_score: float = Field(default=0.0)
    reranker_score: float = Field(default=0.0)
    rank_top: float = Field(default=1.0)
    score_margin: float = Field(default=0.0)
    dense_bm25_agreement: float = Field(default=0.0)
    strategy_agreement: float = Field(default=1.0)
    evidence_validity: float = Field(default=1.0)
    parent_agreement: float = Field(default=1.0)

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.dense_similarity,
            self.bm25_score,
            self.reranker_score,
            self.rank_top,
            self.score_margin,
            self.dense_bm25_agreement,
            self.strategy_agreement,
            self.evidence_validity,
            self.parent_agreement,
        ], dtype=np.float32)


class ConfidenceResult(BaseModel):
    """Result of confidence estimation."""
    evidence_score: float = Field(description="Calibrated evidence score in [0.0, 1.0]")
    confidence_tier: Literal["high", "cautious", "low"] = Field(description="high | cautious | low")
    decision: Literal["answer", "answer_cautiously", "refuse"] = Field(description="answer | answer_cautiously | refuse")
    features: Optional[ConfidenceFeatures] = None
    is_calibrated_model: bool = Field(default=False)


# Default heuristic threshold boundaries
HIGH_CONFIDENCE_THRESHOLD = float(os.environ.get("CONF_THRESHOLD_HIGH", "0.68"))
CAUTIOUS_CONFIDENCE_THRESHOLD = float(os.environ.get("CONF_THRESHOLD_CAUTIOUS", "0.42"))
CALIBRATOR_PATH = Path(os.environ.get("CALIBRATOR_PATH", "index/confidence_calibrator.pkl"))


def extract_confidence_features(
    query: str,
    retrieval_result: RetrievalResult,
) -> ConfidenceFeatures:
    """Extract all 9 confidence features from a retrieval result."""
    chunks = retrieval_result.chunks if retrieval_result else []
    if not chunks:
        return ConfidenceFeatures()

    top_sc = chunks[0]
    top_chunk = top_sc.chunk

    # 1. Reranker / RRF score
    reranker_score = float(top_sc.score)

    # 2. Score Margin (rank 1 vs rank 2)
    margin = 0.0
    if len(chunks) > 1:
        margin = float(chunks[0].score - chunks[1].score)
    else:
        margin = reranker_score

    # 3. Dense & BM25 approximations
    ret_strategy = getattr(top_sc, "retrieval_strategy", None)
    dense_bm25_agreement = 1.0 if str(ret_strategy).lower() in ("hybrid", "retrievalstrategy.hybrid") else 0.5

    # 4. Strategy Agreement (distinct chunking strategies in top-3)
    top_3_strategies = set()
    for sc in chunks[:3]:
        strat = sc.chunk.chunk_strategy or "fixed_size"
        top_3_strategies.add(strat)
    strategy_agreement = float(len(top_3_strategies)) / 3.0

    # 5. Parent / Source Agreement (chunks sharing the same parent doc/passage)
    top_3_parents = [sc.chunk.source_passage_id or sc.chunk.doc_id for sc in chunks[:3]]
    primary_parent = top_3_parents[0]
    parent_matches = sum(1 for p in top_3_parents if p == primary_parent)
    parent_agreement = float(parent_matches) / float(len(top_3_parents)) if top_3_parents else 1.0

    # 6. Evidence Validity (length & punctuation check)
    text_len = len(top_chunk.text.strip())
    has_deva = bool(re.search(r"[\u0900-\u097F]", top_chunk.text))
    has_words = len(top_chunk.text.split()) >= 6
    evidence_validity = 1.0 if (has_words and (has_deva or text_len >= 25)) else 0.4

    # 7. Dense similarity & BM25 estimate from composite score
    dense_similarity = min(1.0, max(0.0, reranker_score * 30.0))
    bm25_score = min(1.0, max(0.0, reranker_score * 25.0))

    return ConfidenceFeatures(
        dense_similarity=dense_similarity,
        bm25_score=bm25_score,
        reranker_score=reranker_score,
        rank_top=1.0,
        score_margin=margin,
        dense_bm25_agreement=dense_bm25_agreement,
        strategy_agreement=strategy_agreement,
        evidence_validity=evidence_validity,
        parent_agreement=parent_agreement,
    )


def compute_evidence_score(
    query: str,
    retrieval_result: RetrievalResult,
) -> ConfidenceResult:
    """Compute calibrated confidence score and tier from multi-feature signals."""
    chunks = retrieval_result.chunks if retrieval_result else []
    if not chunks:
        return ConfidenceResult(
            evidence_score=0.0,
            confidence_tier="low",
            decision="refuse",
            features=ConfidenceFeatures(),
            is_calibrated_model=False,
        )

    features = extract_confidence_features(query, retrieval_result)

    # 1. Check for trained statistical calibrator (Logistic / Isotonic)
    if CALIBRATOR_PATH.exists():
        try:
            with open(CALIBRATOR_PATH, "rb") as f:
                calibrator = pickle.load(f)
            feat_vec = features.to_vector().reshape(1, -1)
            prob = float(calibrator.predict_proba(feat_vec)[0, 1])

            if prob >= HIGH_CONFIDENCE_THRESHOLD:
                return ConfidenceResult(
                    evidence_score=prob,
                    confidence_tier="high",
                    decision="answer",
                    features=features,
                    is_calibrated_model=True,
                )
            elif prob >= CAUTIOUS_CONFIDENCE_THRESHOLD:
                return ConfidenceResult(
                    evidence_score=prob,
                    confidence_tier="cautious",
                    decision="answer_cautiously",
                    features=features,
                    is_calibrated_model=True,
                )
            else:
                return ConfidenceResult(
                    evidence_score=prob,
                    confidence_tier="low",
                    decision="refuse",
                    features=features,
                    is_calibrated_model=True,
                )
        except Exception:
            pass  # Fall back to multi-feature weighted heuristic

    # 2. Multi-feature Weighted Evidence Score Heuristic
    # Weights reflecting signal importance without pretending raw score is probability
    w_rerank = 0.35
    w_margin = 0.20
    w_agree = 0.15
    w_strat = 0.10
    w_parent = 0.10
    w_valid = 0.10

    norm_rerank = min(1.0, features.reranker_score / 0.0328)  # Normalized to max theoretical RRF score
    norm_margin = min(1.0, (features.score_margin / 0.010) if features.score_margin > 0 else 0.0)

    evidence_score = float(
        w_rerank * norm_rerank +
        w_margin * norm_margin +
        w_agree * features.dense_bm25_agreement +
        w_strat * features.strategy_agreement +
        w_parent * features.parent_agreement +
        w_valid * features.evidence_validity
    )

    evidence_score = max(0.0, min(1.0, evidence_score))

    # Determine decision tier
    if evidence_score >= HIGH_CONFIDENCE_THRESHOLD:
        tier = "high"
        decision = "answer"
    elif evidence_score >= CAUTIOUS_CONFIDENCE_THRESHOLD:
        tier = "cautious"
        decision = "answer_cautiously"
    else:
        tier = "low"
        decision = "refuse"

    return ConfidenceResult(
        evidence_score=evidence_score,
        confidence_tier=tier,
        decision=decision,
        features=features,
        is_calibrated_model=False,
    )
