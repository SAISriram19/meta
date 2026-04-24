"""Memory subsystem — cognitively-grounded, HippoRAG-inspired.

Layers:
    * Episodic store — timestamped events, embedding + cues + importance + decay.
    * Semantic store — consolidated rules (via REFLECT).
    * Associative graph — NetworkX DiGraph over episodic ids with typed edges.

Retrieval:
    * Extract query cues → seed the graph → Personalized PageRank → rank by PPR + embedding sim + ACT-R activation.
    * Retrieval bumps importance (activation spreading proxy).

Forgetting:
    * ACT-R style decay per step; retrieval resets decay clock.
    * Soft-delete via low activation threshold.

References:
    HippoRAG (arXiv:2405.14831), Generative Agents (Park et al.),
    A-MEM, GAM (arXiv:2604.12285), ACT-R (ACM HAI 2025).
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import networkx as nx

from env.models import (
    EpisodicMemory,
    RelationType,
    SemanticMemory,
)

if TYPE_CHECKING:
    import numpy as np


# --------------------------------------------------------------------------- #
# Embedding backend — lazy loaded so tests don't require the model.           #
# --------------------------------------------------------------------------- #


class Embedder:
    """Thin wrapper around sentence-transformers with a deterministic fallback.

    The fallback (hash-based pseudo-embedding) is for unit tests / offline dev.
    Real runs load all-MiniLM-L6-v2 (90MB, fast on CPU).

    The loaded model is cached at module level (`_MODEL_CACHE`) so every
    Embedder() instance shares the same underlying model — no repeated 90MB
    reloads across env.reset() calls.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_real: bool = True):
        self.model_name = model_name
        self.use_real = use_real
        self._dim = 384

    @property
    def _model(self):
        return _MODEL_CACHE.get(self.model_name)

    def _load(self):
        if self.model_name in _MODEL_CACHE:
            return
        if not self.use_real:
            return
        try:
            from sentence_transformers import SentenceTransformer

            _MODEL_CACHE[self.model_name] = SentenceTransformer(self.model_name)
        except ImportError:
            self.use_real = False

    def encode(self, text: str) -> list[float]:
        self._load()
        if self._model is not None:
            import numpy as np

            vec = self._model.encode(text, normalize_embeddings=True)
            return np.asarray(vec, dtype=float).tolist()
        return self._hash_embedding(text)

    def _hash_embedding(self, text: str) -> list[float]:
        """Deterministic pseudo-embedding for offline testing. Not semantic."""
        import hashlib

        h = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [(b / 255.0) * 2 - 1 for b in h][: self._dim]
        while len(raw) < self._dim:
            raw.extend(raw[: self._dim - len(raw)])
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]


# Shared across all Embedder instances — load once, reuse forever.
_MODEL_CACHE: dict[str, object] = {}


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot  # both normalized


# --------------------------------------------------------------------------- #
# Cue extraction — lightweight entity/keyword extractor.                      #
# --------------------------------------------------------------------------- #


_STOP = {
    "the", "a", "an", "and", "or", "but", "of", "in", "on", "at", "to", "for",
    "with", "by", "is", "are", "was", "were", "be", "been", "being", "this",
    "that", "these", "those", "i", "you", "he", "she", "it", "we", "they",
    "our", "their", "my", "your", "his", "her", "its", "not", "no", "yes",
}


def extract_cues(text: str, max_cues: int = 8) -> list[str]:
    """Cheap cue extractor: capitalised tokens + content nouns.

    This is a stand-in for proper OpenIE. Good enough for a training env where
    stakeholder_ids and key concepts tend to be explicit.
    """
    words = [w.strip(".,;:!?()[]\"'") for w in text.split()]
    cues: list[str] = []
    seen: set[str] = set()
    for w in words:
        if not w or w.lower() in _STOP or len(w) <= 2:
            continue
        key = w.lower()
        if key in seen:
            continue
        seen.add(key)
        # prefer capitalised or numeric tokens as entity-like cues
        if w[0].isupper() or any(c.isdigit() for c in w):
            cues.append(key)
        elif len(cues) < max_cues // 2:
            cues.append(key)
        if len(cues) >= max_cues:
            break
    return cues


# --------------------------------------------------------------------------- #
# ACT-R activation                                                            #
# --------------------------------------------------------------------------- #


@dataclass
class ActRRecord:
    """Per-memory ACT-R state. Activation = base_level + assoc + noise.

    Simplified ACT-R: base_level = log(sum t_i^-d) across retrievals.
    We keep per-retrieval timestamps and recompute on demand.
    """

    retrievals: list[int] = field(default_factory=list)  # step indices
    created_step: int = 0
    decay_rate: float = 0.5  # standard ACT-R d

    def activation(self, current_step: int) -> float:
        lags = [max(1, current_step - t) for t in self.retrievals]
        if not lags:
            lags = [max(1, current_step - self.created_step)]
        base = math.log(sum(lag ** -self.decay_rate for lag in lags))
        return base

    def bump(self, step: int):
        self.retrievals.append(step)


# --------------------------------------------------------------------------- #
# Memory store                                                                #
# --------------------------------------------------------------------------- #


class MemoryStore:
    """Unified episodic + semantic store with HippoRAG-style retrieval."""

    def __init__(self, embedder: Embedder | None = None, forget_threshold: float = -5.0):
        self.embedder = embedder or Embedder()
        self.episodic: dict[str, EpisodicMemory] = {}
        self.semantic: dict[str, SemanticMemory] = {}
        self.graph = nx.DiGraph()
        self.actr: dict[str, ActRRecord] = {}
        self.current_step = 0
        self.forget_threshold = forget_threshold

    # ------------------------------------------------------------------ #
    # Clock                                                              #
    # ------------------------------------------------------------------ #

    def tick(self, step: int):
        self.current_step = step

    # ------------------------------------------------------------------ #
    # Write                                                              #
    # ------------------------------------------------------------------ #

    def write_episode(
        self,
        step: int,
        content: str,
        cues: list[str] | None = None,
        importance: float = 0.5,
    ) -> EpisodicMemory:
        mid = f"ep_{uuid.uuid4().hex[:8]}"
        cues = cues if cues is not None else extract_cues(content)
        embedding = self.embedder.encode(content)
        mem = EpisodicMemory(
            memory_id=mid,
            step=step,
            content=content,
            cues=cues,
            importance=importance,
            embedding=embedding,
        )
        self.episodic[mid] = mem
        self.actr[mid] = ActRRecord(created_step=step)
        self._index_into_graph(mem)
        return mem

    def write_semantic(
        self,
        step: int,
        rule: str,
        derived_from: list[str],
    ) -> SemanticMemory:
        mid = f"sem_{uuid.uuid4().hex[:8]}"
        mem = SemanticMemory(
            memory_id=mid,
            rule=rule,
            derived_from=derived_from,
            created_step=step,
        )
        self.semantic[mid] = mem
        # graph link: semantic node ↔ its episodic origins
        self.graph.add_node(mid, kind="semantic")
        for ep_id in derived_from:
            if ep_id in self.episodic:
                self.graph.add_edge(mid, ep_id, rel="derived_from", weight=1.0)
                self.graph.add_edge(ep_id, mid, rel="consolidated_into", weight=0.5)
        return mem

    def _index_into_graph(self, mem: EpisodicMemory):
        self.graph.add_node(mem.memory_id, kind="episodic")
        # Cue pseudo-nodes let PPR bridge to memories sharing entities.
        for cue in mem.cues:
            cue_node = f"cue::{cue}"
            if not self.graph.has_node(cue_node):
                self.graph.add_node(cue_node, kind="cue")
            self.graph.add_edge(mem.memory_id, cue_node, rel="has_cue", weight=1.0)
            self.graph.add_edge(cue_node, mem.memory_id, rel="cue_of", weight=1.0)

    # ------------------------------------------------------------------ #
    # Link / forget                                                      #
    # ------------------------------------------------------------------ #

    def link(self, a: str, b: str, relation: RelationType) -> bool:
        if a not in self.graph or b not in self.graph:
            return False
        self.graph.add_edge(a, b, rel=relation.value, weight=1.0)
        # update the pydantic side too for episodic memories
        if a in self.episodic and b not in self.episodic[a].links:
            self.episodic[a].links.append(b)
        return True

    def forget(self, memory_id: str) -> bool:
        removed = False
        if memory_id in self.episodic:
            del self.episodic[memory_id]
            removed = True
        if memory_id in self.semantic:
            del self.semantic[memory_id]
            removed = True
        if memory_id in self.actr:
            del self.actr[memory_id]
        if self.graph.has_node(memory_id):
            self.graph.remove_node(memory_id)
        return removed

    def sweep_decayed(self) -> list[str]:
        """ACT-R: drop memories whose activation dropped below threshold."""
        dropped: list[str] = []
        for mid in list(self.episodic.keys()):
            rec = self.actr.get(mid)
            if rec is None:
                continue
            imp = self.episodic[mid].importance
            act = rec.activation(self.current_step) + imp  # importance protects
            if act < self.forget_threshold:
                self.forget(mid)
                dropped.append(mid)
        return dropped

    # ------------------------------------------------------------------ #
    # Retrieval (HippoRAG-lite)                                          #
    # ------------------------------------------------------------------ #

    def query(
        self,
        query_text: str,
        cues: list[str] | None = None,
        top_k: int = 5,
        alpha_ppr: float = 0.5,
        alpha_sim: float = 0.35,
        alpha_actr: float = 0.15,
    ) -> list[EpisodicMemory | SemanticMemory]:
        """Retrieve top-k memories blending PPR, embedding sim, and ACT-R activation."""
        if not self.episodic and not self.semantic:
            return []

        query_cues = cues if cues else extract_cues(query_text)
        query_embedding = self.embedder.encode(query_text)

        # --- Personalized PageRank seeded on cue nodes present in graph. ---
        seed_nodes = [f"cue::{c}" for c in query_cues if self.graph.has_node(f"cue::{c}")]
        ppr_scores: dict[str, float] = {}
        if seed_nodes:
            personalization = {n: 1.0 / len(seed_nodes) for n in seed_nodes}
            try:
                ppr = nx.pagerank(
                    self.graph,
                    alpha=0.85,
                    personalization=personalization,
                    max_iter=50,
                    tol=1e-4,
                )
                ppr_scores = {
                    n: s for n, s in ppr.items()
                    if n in self.episodic or n in self.semantic
                }
            except nx.PowerIterationFailedConvergence:
                ppr_scores = {}

        # --- Score each candidate memory. ---
        scored: list[tuple[float, EpisodicMemory | SemanticMemory]] = []
        all_memories: list[EpisodicMemory | SemanticMemory] = (
            list(self.episodic.values()) + list(self.semantic.values())
        )
        for mem in all_memories:
            mid = mem.memory_id
            # similarity
            if isinstance(mem, EpisodicMemory) and mem.embedding is not None:
                sim = cosine(query_embedding, mem.embedding)
            else:
                text = mem.rule if isinstance(mem, SemanticMemory) else mem.content
                mem_embed = self.embedder.encode(text)
                sim = cosine(query_embedding, mem_embed)
            # PPR
            ppr = ppr_scores.get(mid, 0.0)
            # ACT-R activation
            rec = self.actr.get(mid)
            act = rec.activation(self.current_step) if rec else 0.0
            # combined
            score = alpha_sim * sim + alpha_ppr * ppr * 10 + alpha_actr * act * 0.1
            # importance nudge
            importance = (
                mem.importance if isinstance(mem, EpisodicMemory) else 0.6
            )
            score += 0.1 * importance
            scored.append((score, mem))

        scored.sort(key=lambda t: t[0], reverse=True)
        top = [m for _, m in scored[:top_k]]
        # Retrieval bump — ACT-R spreading activation proxy.
        for m in top:
            rec = self.actr.get(m.memory_id)
            if rec is not None:
                rec.bump(self.current_step)
            # also bump importance a little — used memories matter
            if isinstance(m, EpisodicMemory):
                m.importance = min(1.0, m.importance + 0.02)
            elif isinstance(m, SemanticMemory):
                m.applications += 1
        return top

    # ------------------------------------------------------------------ #
    # Stats                                                              #
    # ------------------------------------------------------------------ #

    def stats(self) -> dict[str, int]:
        return {
            "episodic": len(self.episodic),
            "semantic": len(self.semantic),
            "graph_nodes": self.graph.number_of_nodes(),
            "graph_edges": self.graph.number_of_edges(),
        }
