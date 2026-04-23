"""Smoke tests for the memory subsystem — runs without heavy deps."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.memory import Embedder, MemoryStore, extract_cues  # noqa: E402
from env.models import RelationType  # noqa: E402


def test_cue_extraction():
    cues = extract_cues("Stakeholder Alice demanded a 40% budget cut by Friday.")
    assert "alice" in cues or "stakeholder" in cues
    assert any("40" in c for c in cues) or "friday" in cues


def test_write_and_query():
    store = MemoryStore(embedder=Embedder(use_real=False))
    store.tick(1)
    store.write_episode(
        step=1,
        content="Alice demanded aggressive 40% budget cut.",
        importance=0.8,
    )
    store.tick(5)
    store.write_episode(
        step=5,
        content="Bob praised the conservative spending plan.",
    )
    store.tick(10)
    store.write_episode(
        step=10,
        content="Alice now insists we must spend aggressively to win market share.",
    )
    store.tick(11)
    hits = store.query("Alice budget position", top_k=3)
    # Alice-related memories should surface; at least one present.
    alice_hits = [h for h in hits if "Alice" in getattr(h, "content", "") or "alice" in getattr(h, "cues", [])]
    assert alice_hits, f"expected Alice-related hits, got {[getattr(h, 'content', h) for h in hits]}"


def test_reflect_semantic():
    store = MemoryStore(embedder=Embedder(use_real=False))
    store.tick(1)
    e1 = store.write_episode(1, "Alice agreed to proposal A.")
    store.tick(5)
    e2 = store.write_episode(5, "Alice later rejected proposal A citing new info.")
    store.tick(6)
    sem = store.write_semantic(
        step=6,
        rule="Alice reverses within 5 steps when new info arrives.",
        derived_from=[e1.memory_id, e2.memory_id],
    )
    assert sem.memory_id.startswith("sem_")
    hits = store.query("Alice reliability")
    assert sem in hits or any(h.memory_id == sem.memory_id for h in hits)


def test_link_and_forget():
    store = MemoryStore(embedder=Embedder(use_real=False))
    store.tick(1)
    a = store.write_episode(1, "Contract milestone missed.")
    b = store.write_episode(2, "Invoice still sent regardless.")
    assert store.link(a.memory_id, b.memory_id, RelationType.CONTRADICTS)
    assert store.forget(a.memory_id)
    assert a.memory_id not in store.episodic


if __name__ == "__main__":
    test_cue_extraction()
    test_write_and_query()
    test_reflect_semantic()
    test_link_and_forget()
    print("memory smoke tests passed.")
