"""Conditional storage operations bound to a retained candidate snapshot.

Call only after the existing Alignment/evidence/target-support gates authorize
the operation. This module does not replace those gates or interpret decisions.
It accepts an index, never an LLM-provided database ID or expected_version.
"""

from memory_candidate import CandidateSnapshot
from memory_repository import MemoryRepository, MutationResult


def update_from_snapshot(
    repository: MemoryRepository, snapshot: CandidateSnapshot, *,
    sender_id: str, target_index: int, content: str,
) -> MutationResult:
    target = snapshot.resolve_target(sender_id, target_index)
    return repository.update(
        sender_id, target.fact_id, content, expected_version=target.version,
    )


def delete_from_snapshot(
    repository: MemoryRepository, snapshot: CandidateSnapshot, *,
    sender_id: str, target_index: int,
) -> MutationResult:
    target = snapshot.resolve_target(sender_id, target_index)
    return repository.delete(sender_id, target.fact_id, expected_version=target.version)
