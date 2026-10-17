"""Repository-backed retrieval using the existing encode/cosine/rank logic.

No SQL, JSON imports, model construction or embedding/result cache. Inject the
existing SentenceTransformer instance at composition time.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from sklearn.metrics.pairwise import cosine_similarity

from memory_candidate import CandidateSnapshot, MemoryCandidate, _nonempty
from memory_repository import MemoryFact


# Mirror current JSON retrieval and V2 fact-relevance semantics. Compatibility
# tests execute those existing functions as the oracle without changing them.
MIN_RETRIEVAL_SCORE = 0.10
MIN_FACT_RELEVANCE_SCORE = 0.40


class FactReader(Protocol):
    def list_facts(self, sender_id: str) -> list[MemoryFact]: ...


class EmbeddingModel(Protocol):
    def encode(self, sentences: list[str]): ...


class SQLiteMemoryRetriever:
    def __init__(self, repository: FactReader, embedding_model: EmbeddingModel):
        self._repository = repository
        self._model = embedding_model

    def retrieve(
        self, sender_id: str, query: str, top_k: int = 2, *, high_recall: bool = False,
    ) -> CandidateSnapshot:
        """Standard: >=0.10, then top-k. High recall: top-k without filtering.

        Equal scores preserve Repository insertion order. A fresh fact snapshot
        is read exactly once per call. top_k=0 returns an empty snapshot; invalid
        inputs raise ValueError. This layer does not choose Alignment actions.
        """
        _nonempty(sender_id, "sender_id")
        _nonempty(query, "query")
        if type(top_k) is not int or top_k < 0:
            raise ValueError("top_k must be a non-negative integer.")
        if type(high_recall) is not bool:
            raise ValueError("high_recall must be a boolean.")
        facts = tuple(self._repository.list_facts(sender_id))
        if any(not isinstance(fact, MemoryFact) for fact in facts):
            raise ValueError("Repository must return MemoryFact snapshots.")
        # Validate identity and sender before encoding or exposing any content.
        original = CandidateSnapshot(sender_id, tuple(
            MemoryCandidate(fact.id, fact.sender_id, fact.content, fact.version, 0.0)
            for fact in facts
        ))
        if not facts or top_k == 0:
            return CandidateSnapshot(sender_id, ())
        memory_embeddings = self._model.encode(list(original.contents))
        query_embedding = self._model.encode([query])
        similarities = cosine_similarity(query_embedding, memory_embeddings)
        if similarities.shape != (1, len(facts)):
            raise ValueError("Embedding results do not match the fact/query snapshot.")
        ranked = sorted((
            MemoryCandidate(fact.id, fact.sender_id, fact.content, fact.version, float(score))
            for fact, score in zip(facts, similarities[0])
        ), key=lambda candidate: candidate.score, reverse=True)
        if not high_recall:
            ranked = [candidate for candidate in ranked if candidate.score >= MIN_RETRIEVAL_SCORE]
        return CandidateSnapshot(sender_id, tuple(ranked[:top_k]))

    def retrieve_fact_candidates(
        self, sender_id: str, atomic_fact: Mapping, top_k: int = 5,
    ) -> CandidateSnapshot:
        """Match V2's existing standard/high-recall selection and threshold order."""
        if not isinstance(atomic_fact, Mapping):
            raise ValueError("atomic_fact must be a mapping.")
        high_recall = (
            atomic_fact.get("kind") == "persistent_change"
            or atomic_fact.get("operation_hint") in {"update", "end"}
        )
        snapshot = self.retrieve(sender_id, atomic_fact.get("fact"), top_k, high_recall=high_recall)
        if high_recall:
            return snapshot
        # V2 first applies ordinary retrieval/top-k, then the fact threshold.
        return CandidateSnapshot(sender_id, tuple(
            candidate for candidate in snapshot.candidates if candidate.score >= MIN_FACT_RELEVANCE_SCORE
        ))
