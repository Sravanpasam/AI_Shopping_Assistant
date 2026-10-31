"""
policy_search.py
----------------
Semantic search over the store's policy documents in docs/.

Each document is chunked by heading and paragraph (Markdown headings and blank
lines), the chunks are embedded with the same SentenceTransformer model the rest
of the app uses, and search_policies(query, k=3) returns the best-matching chunks
with their source file and similarity score.

This module is additive (System B). It does not touch System A or app.py.
"""

import os
import glob
import logging
from typing import List, Dict, Any, Optional

import numpy as np

from nlp_engine import get_embedding

logger = logging.getLogger(__name__)

DOCS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs"
)

# Minimum cosine similarity for a policy chunk to count as a relevant match.
# Chosen from `python policy_scores.py` (10 policy questions + 10 off-topic
# questions): policy questions scored 0.437–0.793, off-topic ones topped out at
# 0.360. 0.40 sits between the two.
MIN_POLICY_SIMILARITY: float = 0.40

# Module-level cache
_chunks: Optional[List[Dict[str, Any]]] = None
_chunk_matrix: Optional[np.ndarray] = None


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row so a dot product equals cosine similarity."""
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1e-10
    return matrix / norms


def _chunk_markdown(path: str, source: str) -> List[Dict[str, Any]]:
    """
    Split one Markdown file into chunks by heading and paragraph.

    A chunk is a run of non-blank lines (a paragraph or a bullet list) under the
    most recent heading. The heading (or document title) is prepended to the
    chunk text so the embedding has topical context.

    Returns a list of {"text", "source", "heading"} dicts.
    """
    chunks: List[Dict[str, Any]] = []
    title = ""
    heading = ""
    paragraph: List[str] = []

    def flush() -> None:
        nonlocal paragraph
        text = "\n".join(paragraph).strip()
        if text:
            label = heading or title
            chunks.append({
                "text": f"{label}\n{text}".strip() if label else text,
                "source": source,
                "heading": label,
            })
        paragraph = []

    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line.startswith("# "):
                flush()
                title = line[2:].strip()
                heading = ""
            elif line.startswith("## "):
                flush()
                heading = line[3:].strip()
            elif line.strip() == "":
                flush()
            else:
                paragraph.append(line)
    flush()
    return chunks


def _build_index() -> None:
    """Read docs/, build all chunks, and embed them once (cached)."""
    global _chunks, _chunk_matrix
    if _chunks is not None:
        return

    chunks: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(DOCS_DIR, "*.md"))):
        chunks.extend(_chunk_markdown(path, os.path.basename(path)))

    _chunks = chunks
    if chunks:
        vectors = np.vstack([get_embedding(c["text"]) for c in chunks])
        _chunk_matrix = _normalize(vectors)
    else:
        logger.warning("No policy documents found in %s", DOCS_DIR)
        _chunk_matrix = np.zeros((0, 0), dtype=np.float32)
    logger.info("Policy index built: %d chunks from docs/", len(chunks))


def get_chunks() -> List[Dict[str, Any]]:
    """Return the indexed policy chunks (building the index on first call)."""
    _build_index()
    return _chunks or []


def search_policies(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """
    Return the top-k policy chunks most similar to the query.

    Args:
        query: A user question (already translated to English).
        k: Maximum number of chunks to return.

    Returns:
        List of {"text", "source", "heading", "score"} dicts, highest score
        first. Chunks scoring below MIN_POLICY_SIMILARITY are dropped when that
        constant is set. Empty list if the query is blank or no docs exist.
    """
    if not query or not query.strip():
        return []

    _build_index()
    if not _chunks or _chunk_matrix is None or _chunk_matrix.shape[0] == 0:
        return []

    query_vec = _normalize(get_embedding(query)[None, :])[0]
    scores = _chunk_matrix @ query_vec

    results: List[Dict[str, Any]] = []
    for idx in np.argsort(-scores):
        score = float(scores[idx])
        if MIN_POLICY_SIMILARITY is not None and score < MIN_POLICY_SIMILARITY:
            continue
        chunk = _chunks[int(idx)]
        results.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "heading": chunk["heading"],
            "score": round(score, 4),
        })
        if len(results) >= k:
            break
    return results
