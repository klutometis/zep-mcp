"""Voyage embeddings and reranking over plain HTTP (no SDK).

The house standard (see ~/prg/artificial/analogies/bin/graphstore.py):
voyage-4-large, 1024 dimensions, input_type "document" for stored facts and
"query" for queries; rerank-2.5 for relevance. Both calls return the token
count Voyage bills, so callers can account for cost.
"""

from __future__ import annotations

import os
import time

import httpx
import numpy as np

EMBED_MODEL = "voyage-4-large"
RERANK_MODEL = "rerank-2.5"
DIMS = 1024
API = "https://api.voyageai.com/v1"


class Voyage:
    def __init__(self, api_key: str | None = None, timeout: float = 30.0, tries: int = 5):
        key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not key:
            raise RuntimeError("VOYAGE_API_KEY is not set")
        self.tries = tries
        # One keep-alive client: a fresh TLS handshake per call costs ~100 ms.
        self.http = httpx.Client(
            base_url=API,
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout,
        )

    def _post(self, path: str, body: dict) -> dict:
        for t in range(self.tries):
            try:
                r = self.http.post(path, json=body)
                if r.status_code in (429, 500, 502, 503, 504) and t < self.tries - 1:
                    time.sleep(2**t)
                    continue
                r.raise_for_status()
                return r.json()
            except (httpx.TransportError, httpx.TimeoutException):
                if t < self.tries - 1:
                    time.sleep(2**t)
                    continue
                raise
        raise RuntimeError("unreachable")

    def embed(self, texts: list[str], input_type: str) -> tuple[np.ndarray, int]:
        """Unit-normalised float32 vectors, one row per text, and tokens billed."""
        if not texts:
            return np.zeros((0, DIMS), dtype=np.float32), 0
        out: list[list[float]] = []
        tokens = 0
        for i in range(0, len(texts), 128):
            r = self._post(
                "/embeddings",
                {
                    "input": [t[:16000] for t in texts[i : i + 128]],
                    "model": EMBED_MODEL,
                    "input_type": input_type,
                    "output_dimension": DIMS,
                },
            )
            out += [d["embedding"] for d in sorted(r["data"], key=lambda d: d["index"])]
            tokens += int((r.get("usage") or {}).get("total_tokens", 0))
        v = np.asarray(out, dtype=np.float32)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9), tokens

    def rerank(self, query: str, docs: list[str]) -> tuple[list[float], int]:
        """rerank-2.5 relevance for every doc, in the docs' order, and tokens billed."""
        if not docs:
            return [], 0
        r = self._post(
            "/rerank",
            {"query": query[:8000], "documents": docs, "model": RERANK_MODEL},
        )
        scores = [0.0] * len(docs)
        for d in r["data"]:
            scores[d["index"]] = float(d["relevance_score"])
        return scores, int((r.get("usage") or {}).get("total_tokens", 0))
