"""Storage contract for facts, independent of retrieval and LLM decisions.

IDs are opaque and stable for a fact's lifetime. Versions start at 1 and
increase on every successful update, including updates with identical content.
Sender IDs and content are validated but never normalized or deduplicated here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ContextManager, Protocol


@dataclass(frozen=True, slots=True)
class MemoryFact:
    id: str
    sender_id: str
    content: str
    version: int


class MutationStatus(str, Enum):
    SUCCESS = "success"
    VERSION_CONFLICT = "version_conflict"
    NOT_FOUND = "not_found"


@dataclass(frozen=True, slots=True)
class MutationResult:
    """Successful UPDATE returns the new fact; DELETE returns the removed fact.

    Failed mutations return no fact. An ID belonging to another sender is
    NOT_FOUND, regardless of its version; no other sender's state is exposed.
    """

    status: MutationStatus
    fact: MemoryFact | None = None


class MemoryRepository(Protocol):
    """All operations require an explicit sender namespace.

    list() preserves insertion order. add() always creates a fresh ID and does
    not perform semantic or exact-content deduplication. Invalid arguments raise
    ValueError; storage errors propagate rather than becoming NOT_FOUND.

    Mutations commit individually outside transaction(). An escaping exception
    rolls back a transaction; nested scopes use savepoints. Returned conflict /
    not-found statuses are ordinary results, so callers must raise an exception
    if a failed result should abort a batch. This is a storage transaction, not
    a transaction over the Agent's reply, feedback or Working Memory.
    """

    def list(self, sender_id: str) -> list[MemoryFact]: ...

    def list_facts(self, sender_id: str) -> list[MemoryFact]: ...

    def list_senders(self) -> list[str]: ...

    def ensure_sender(self, sender_id: str) -> None: ...

    def add(self, sender_id: str, content: str) -> MemoryFact: ...

    def update(
        self, sender_id: str, fact_id: str, content: str, *, expected_version: int,
    ) -> MutationResult: ...

    def delete(
        self, sender_id: str, fact_id: str, *, expected_version: int,
    ) -> MutationResult: ...

    def transaction(self) -> ContextManager["MemoryRepository"]: ...

    def close(self) -> None: ...
