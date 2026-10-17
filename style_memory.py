import json
from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity

from memory_search import model


STYLE_MEMORY_FILE = (
    Path(__file__).resolve().parent
    / "style_samples.json"
)


def load_style_samples(path=STYLE_MEMORY_FILE):
    """Load and validate style samples from a JSON list."""

    path = Path(path)

    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as file:
        samples = json.load(file)

    if not isinstance(samples, list):
        raise ValueError(
            "Style Memory JSON must contain a list."
        )

    validated_samples = []

    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(
                f"Style sample {index} must be an object."
            )

        validated_sample = {}

        for field in ("sender_id", "incoming", "reply"):
            value = sample.get(field)

            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Style sample {index} has invalid {field}."
                )

            validated_sample[field] = value.strip()

        validated_samples.append(validated_sample)

    return validated_samples


def retrieve_style_samples(
    sender_id,
    current_message,
    top_k=3,
    samples=None
):
    """
    Retrieve semantically similar replies for one sender only.

    Similarity is calculated from the historical incoming message;
    Long-Term Memory is never included in this retrieval result.
    """

    if not isinstance(sender_id, str) or not sender_id.strip():
        raise ValueError("sender_id must be a non-empty string.")

    if (
        not isinstance(current_message, str)
        or not current_message.strip()
    ):
        raise ValueError(
            "current_message must be a non-empty string."
        )

    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    if samples is None:
        samples = load_style_samples()

    sender_samples = [
        sample
        for sample in samples
        if sample["sender_id"] == sender_id
    ]

    if not sender_samples:
        return []

    incoming_messages = [
        sample["incoming"]
        for sample in sender_samples
    ]

    sample_embeddings = model.encode(
        incoming_messages
    )
    query_embedding = model.encode(
        [current_message]
    )

    scores = cosine_similarity(
        query_embedding,
        sample_embeddings
    )[0]

    ranked = sorted(
        zip(scores, sender_samples),
        key=lambda item: item[0],
        reverse=True
    )

    return [
        {
            "incoming": sample["incoming"],
            "reply": sample["reply"],
            "score": float(score)
        }
        for score, sample in ranked[:top_k]
    ]
