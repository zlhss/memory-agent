from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json


min_score = 0.10


# =========================================================
# Load / Save
# =========================================================

def load_memories():
    with open(
        "memories.json",
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def save_memories():
    with open(
        "memories.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            memories_by_sender,
            f,
            ensure_ascii=False,
            indent=2
        )


# =========================================================
# 初始化
# =========================================================

memories_by_sender = load_memories()

model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)


# =========================================================
# Sender Memory
# =========================================================

def get_sender_memories(sender_id):

    return memories_by_sender.get(
        sender_id,
        []
    )


def ensure_sender(sender_id):

    if sender_id not in memories_by_sender:
        memories_by_sender[sender_id] = []

    return memories_by_sender[sender_id]


# =========================================================
# CRUD
# =========================================================

def find_memory(
    sender_id,
    text
):

    sender_memories = get_sender_memories(
        sender_id
    )

    if text in sender_memories:
        return sender_memories.index(text)

    return None


def add_memory(
    sender_id,
    new_memory
):

    sender_memories = ensure_sender(
        sender_id
    )

    if find_memory(
        sender_id,
        new_memory
    ) is not None:

        print(
            f"Memory already exists: "
            f"{new_memory}"
        )

        return False

    sender_memories.append(
        new_memory
    )

    save_memories()

    return True


def update_memory(
    sender_id,
    old_memory,
    new_memory
):

    index = find_memory(
        sender_id,
        old_memory
    )

    if index is None:

        print(
            f"Memory not found: "
            f"{old_memory}"
        )

        return False

    sender_memories = get_sender_memories(
        sender_id
    )

    sender_memories[index] = (
        new_memory
    )

    save_memories()

    return True


def delete_memory(
    sender_id,
    old_memory
):

    index = find_memory(
        sender_id,
        old_memory
    )

    if index is None:

        print(
            f"Memory not found: "
            f"{old_memory}"
        )

        return False

    sender_memories = get_sender_memories(
        sender_id
    )

    sender_memories.pop(
        index
    )

    save_memories()

    return True


# =========================================================
# Retrieval
# =========================================================

def retrieve(
    sender_id,
    query,
    top_k=2
):

    sender_memories = get_sender_memories(
        sender_id
    )

    if not sender_memories:
        return []

    memory_embeddings = model.encode(
        sender_memories
    )

    query_embedding = model.encode(
        [query]
    )

    similarities = cosine_similarity(
        query_embedding,
        memory_embeddings
    )[0]

    pairs = list(
        zip(
            similarities,
            sender_memories
        )
    )

    ranked = sorted(
        pairs,
        key=lambda x: x[0],
        reverse=True
    )

    filtered = []

    for score, memory in ranked:

        if score >= min_score:

            filtered.append(
                (
                    score,
                    memory
                )
            )

    return filtered[:top_k]


def get_relevant_memories(
    sender_id,
    user_text,
    top_k=3
):

    results = retrieve(
        sender_id,
        user_text,
        top_k=top_k
    )

    return [
        memory
        for score, memory in results
    ]


# =========================================================
# Apply Action
# =========================================================

def apply_memory_action(
    sender_id,
    decision
):

    action = decision["action"]

    match action:

        case "ADD":

            return add_memory(
                sender_id,
                decision["memory"]
            )

        case "UPDATE":

            return update_memory(
                sender_id,
                decision["target"],
                decision["memory"]
            )

        case "DELETE":

            return delete_memory(
                sender_id,
                decision["target"]
            )

        case "IGNORE":

            return True

        case _:

            return False


# =========================================================
# 测试辅助
# =========================================================

def reset_memories(
    new_memories
):

    global memories_by_sender

    memories_by_sender = (
        new_memories.copy()
    )

    save_memories()
    
    
if __name__ == "__main__":
    print(retrieve(
        "xiaoming",
        "最近在学什么？",
        top_k=3
    ))

    print(retrieve(
        "teacher_wang",
        "喜欢喝什么？",
        top_k=3
    ))
