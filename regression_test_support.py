import copy

import requests


TARGET_SENDER_ID = "regression_target"
SENTINEL_SENDER_ID = "regression_sentinel"

SENTINEL_MEMORIES = [
    "联系人住在哨兵城市",
    "联系人喜欢哨兵音乐"
]


def build_sender_state(target_memories):
    return {
        TARGET_SENDER_ID: copy.deepcopy(target_memories),
        SENTINEL_SENDER_ID: copy.deepcopy(SENTINEL_MEMORIES)
    }


def reset_case_state(
    memory_search,
    main,
    target_memories,
    working_messages=None
):
    memory_search.reset_memories(
        build_sender_state(target_memories)
    )
    main.reset_working_memories()

    working_memory = main.get_working_memory(
        TARGET_SENDER_ID
    )

    for message in working_messages or []:
        working_memory.add_message(
            message["role"],
            message["content"]
        )


def get_target_memories(memory_search):
    return memory_search.get_sender_memories(
        TARGET_SENDER_ID
    ).copy()


def get_sentinel_memories(memory_search):
    return memory_search.get_sender_memories(
        SENTINEL_SENDER_ID
    ).copy()


def sentinel_is_unchanged(memory_search):
    return (
        get_sentinel_memories(memory_search)
        == SENTINEL_MEMORIES
    )


def is_environment_error(error):
    current = error

    while current is not None:
        if isinstance(
            current,
            (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.HTTPError,
                ImportError,
                OSError,
                TimeoutError
            )
        ):
            return True

        current = current.__cause__ or current.__context__

    error_text = repr(error).lower()
    environment_markers = (
        "connection refused",
        "failed to establish a new connection",
        "maxretryerror",
        "huggingface.co",
        "sentence-transformers",
        "sentence_transformers",
        "ollama",
        "timed out",
        "timeout",
        "model not found"
    )

    return any(
        marker in error_text
        for marker in environment_markers
    )
