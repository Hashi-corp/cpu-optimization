"""
KV-Cache and Prompt Cache implementations
=========================================
1. ManualKVCache – explicit past_key_values management
2. PrefixCache   – shared prompt prefix cache (CPU tensor store)
3. SemanticCache – embedding-similarity based response cache
"""

import hashlib
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


# ─── 1. Manual KV-Cache (wraps HF past_key_values) ──────────────────────────

class ManualKVCache:
    """
    Stores and reuses past_key_values across calls so the model avoids
    recomputing attention for tokens already seen.
    """

    def __init__(self, max_cache_size: int = 5):
        self._cache: Dict[str, Any]  = {}
        self._hits  = 0
        self._misses= 0
        self._max   = max_cache_size

    def _key(self, input_ids) -> str:
        import torch
        if isinstance(input_ids, torch.Tensor):
            return hashlib.md5(input_ids.numpy().tobytes()).hexdigest()
        return hashlib.md5(str(input_ids).encode()).hexdigest()

    def get(self, input_ids) -> Optional[Tuple]:
        k = self._key(input_ids)
        if k in self._cache:
            self._hits += 1
            return self._cache[k]["past_key_values"]
        self._misses += 1
        return None

    def put(self, input_ids, past_key_values):
        if len(self._cache) >= self._max:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        k = self._key(input_ids)
        self._cache[k] = {"past_key_values": past_key_values,
                          "timestamp": time.time()}

    @property
    def stats(self) -> Dict:
        total = self._hits + self._misses
        return {
            "hits":       self._hits,
            "misses":     self._misses,
            "hit_rate":   self._hits / max(total, 1),
            "cache_size": len(self._cache),
        }


# ─── 2. Prefix Cache (shared prompt prefix) ──────────────────────────────────

@dataclass
class PrefixEntry:
    prefix_ids:        Any          # torch.Tensor
    past_key_values:   Tuple
    prefix_len:        int
    created_at:        float        = field(default_factory=time.time)
    access_count:      int          = 0


class PrefixCache:
    """
    Cache the KV states for a fixed prompt prefix.  When the same prefix
    is reused, the model skips recomputing its attention layers.
    """

    def __init__(self, model, tokenizer, max_prefixes: int = 10):
        self.model      = model
        self.tokenizer  = tokenizer
        self._store: Dict[str, PrefixEntry] = {}
        self._max       = max_prefixes
        self._hits      = 0
        self._misses    = 0

    def _encode_prefix(self, text: str) -> Any:
        import torch
        return self.tokenizer(text, return_tensors="pt").input_ids

    def _prefix_key(self, text: str) -> str:
        return hashlib.md5(text.strip().encode()).hexdigest()

    def warm(self, prefix_text: str) -> "PrefixEntry":
        """Pre-compute and store the KV cache for a prefix."""
        import torch
        key       = self._prefix_key(prefix_text)
        input_ids = self._encode_prefix(prefix_text)
        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True)
        entry = PrefixEntry(
            prefix_ids=input_ids,
            past_key_values=outputs.past_key_values,
            prefix_len=input_ids.shape[1],
        )
        if len(self._store) >= self._max:
            lru = min(self._store, key=lambda k: self._store[k].access_count)
            del self._store[lru]
        self._store[key] = entry
        logger.info("PrefixCache: warmed prefix '%s…' (%d tokens)",
                    prefix_text[:40], entry.prefix_len)
        return entry

    def generate_with_prefix(
        self,
        prefix_text: str,
        suffix_text: str,
        max_new_tokens: int = 100,
    ) -> Tuple[str, Dict]:
        """Generate text, reusing prefix KV cache if available."""
        import torch

        key   = self._prefix_key(prefix_text)
        t0    = time.perf_counter()

        if key not in self._store:
            self._misses += 1
            self.warm(prefix_text)
        else:
            self._hits += 1

        entry = self._store[key]
        entry.access_count += 1

        suffix_ids = self.tokenizer(suffix_text, return_tensors="pt",
                                    add_special_tokens=False).input_ids

        with torch.no_grad():
            output_ids = self.model.generate(
                suffix_ids,
                past_key_values=entry.past_key_values,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                do_sample=False,
            )

        generated = self.tokenizer.decode(
            output_ids[0][suffix_ids.shape[1]:], skip_special_tokens=True
        )
        elapsed = (time.perf_counter() - t0) * 1000

        return generated, {
            "prefix_cache_hit": key in self._store,
            "latency_ms": elapsed,
            "prefix_len": entry.prefix_len,
        }

    @property
    def stats(self) -> Dict:
        total = self._hits + self._misses
        return {
            "hits":     self._hits,
            "misses":   self._misses,
            "hit_rate": self._hits / max(total, 1),
        }


# ─── 3. Semantic Cache (embedding similarity) ────────────────────────────────

@dataclass
class SemanticCacheEntry:
    query:     str
    response:  str
    embedding: np.ndarray
    latency_ms: float
    created_at: float = field(default_factory=time.time)
    hits:       int   = 0


class SemanticCache:
    """
    Cache LLM responses by semantic similarity of the input prompt.
    Uses a tiny sentence-transformer embedding + cosine similarity.
    """

    def __init__(
        self,
        encoder_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.92,
        max_entries: int = 500,
    ):
        from sentence_transformers import SentenceTransformer
        self.encoder   = SentenceTransformer(encoder_model)
        self.threshold = similarity_threshold
        self.max       = max_entries
        self._entries: List[SemanticCacheEntry] = []
        self._hits   = 0
        self._misses = 0

    def _embed(self, text: str) -> np.ndarray:
        return self.encoder.encode(text, normalize_embeddings=True)

    def lookup(self, query: str) -> Optional[SemanticCacheEntry]:
        if not self._entries:
            self._misses += 1
            return None
        q_emb  = self._embed(query)
        scores = np.array([e.embedding @ q_emb for e in self._entries])
        best_i = int(np.argmax(scores))
        if scores[best_i] >= self.threshold:
            self._hits += 1
            self._entries[best_i].hits += 1
            return self._entries[best_i]
        self._misses += 1
        return None

    def store(self, query: str, response: str, latency_ms: float):
        emb = self._embed(query)
        if len(self._entries) >= self.max:
            # evict least-used
            self._entries.sort(key=lambda e: e.hits)
            self._entries.pop(0)
        self._entries.append(
            SemanticCacheEntry(query=query, response=response,
                               embedding=emb, latency_ms=latency_ms)
        )

    @property
    def stats(self) -> Dict:
        total = self._hits + self._misses
        return {
            "hits":       self._hits,
            "misses":     self._misses,
            "hit_rate":   self._hits / max(total, 1),
            "cache_size": len(self._entries),
            "avg_saved_ms": (
                np.mean([e.latency_ms for e in self._entries])
                if self._entries else 0
            ),
        }
