"""Shared sender-scoped conversation state and contact memory schema."""

from working_memory import WorkingMemory


# 当前会话的短期记忆
working_memories = {}


def get_working_memory(sender_id):
    if sender_id not in working_memories:
        working_memories[sender_id] = WorkingMemory(
            max_messages=6
        )

    return working_memories[sender_id]


def reset_working_memories():
    """Clear Working Memory for every sender."""

    working_memories.clear()


def is_valid_contact_memory(memory):
    """Return whether a value conforms to the contact memory schema."""

    return (
        isinstance(memory, str)
        and bool(memory)
        and memory.startswith("联系人")
    )
