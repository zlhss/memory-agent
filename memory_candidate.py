"""Immutable candidate snapshots; database identities never come from an LLM."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real


def _nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")


@dataclass(frozen=True, slots=True)
class MemoryCandidate:
    fact_id: str
    sender_id: str
    content: str
    version: int
    score: float

    def __post_init__(self):
        for name in ("fact_id", "sender_id", "content"):
            _nonempty(getattr(self, name), name)
        if type(self.version) is not int or self.version < 1:
            raise ValueError("Candidate version must be a positive integer.")
        if isinstance(self.score, bool) or not isinstance(self.score, Real) or not math.isfinite(self.score):
            raise ValueError("Candidate score must be a finite real number.")
        object.__setattr__(self, "score", float(self.score))


@dataclass(frozen=True, slots=True)
class CandidateSnapshot:
    """One sender's final ordered candidates for one LLM invocation.

    Freeze after all ranking/filtering. Send list(snapshot.contents) to the
    existing text-only alignment function, without reordering that list, and
    retain THIS snapshot until its target_index has been resolved. A later
    retrieval produces an independent snapshot, never a replacement for it.
    """

    sender_id: str
    candidates: tuple[MemoryCandidate, ...]

    def __post_init__(self):
        _nonempty(self.sender_id, "sender_id")
        # Defensive copy even when a caller supplies a mutable list/generator.
        candidates = tuple(self.candidates)
        for candidate in candidates:
            if not isinstance(candidate, MemoryCandidate):
                raise ValueError("Snapshot items must be MemoryCandidate objects.")
            if candidate.sender_id != self.sender_id:
                raise ValueError("Snapshot cannot contain another sender's facts.")
        if len({candidate.fact_id for candidate in candidates}) != len(candidates):
            raise ValueError("Snapshot contains repeated Fact IDs.")
        object.__setattr__(self, "candidates", candidates)

    @property
    def contents(self) -> tuple[str, ...]:
        """Only content crosses into the existing Prompt; retain duplicates."""
        return tuple(candidate.content for candidate in self.candidates)

    def resolve_target(self, sender_id: str, target_index: int) -> MemoryCandidate:
        _nonempty(sender_id, "sender_id")
        if sender_id != self.sender_id:
            raise ValueError("Target sender does not match the candidate snapshot.")
        # Match V2's exact-int rule: bool, floats, strings and negative indices
        # must never acquire Python indexing semantics or mutation authority.
        if type(target_index) is not int or not 0 <= target_index < len(self.candidates):
            raise ValueError("target_index must be an integer within the original snapshot.")
        return self.candidates[target_index]
