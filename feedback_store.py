import json
from pathlib import Path


FEEDBACK_FILE = (
    Path(__file__).resolve().parent
    / "feedback.json"
)

VALID_FEEDBACK = {
    "accepted",
    "edited",
    "rejected"
}


def load_feedback(path=FEEDBACK_FILE):
    path = Path(path)

    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError(
            "feedback.json must contain a list."
        )

    return records


def record_feedback(
    sender_id,
    incoming,
    generated,
    feedback,
    final=None,
    path=FEEDBACK_FILE
):
    """Validate and append one human feedback record."""

    for field, value in (
        ("sender_id", sender_id),
        ("incoming", incoming),
        ("generated", generated)
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{field} must be a non-empty string."
            )

    if feedback not in VALID_FEEDBACK:
        raise ValueError(
            "feedback must be accepted, edited, or rejected."
        )

    if feedback == "accepted":
        final = generated.strip()
    elif feedback == "edited":
        if not isinstance(final, str) or not final.strip():
            raise ValueError(
                "Edited feedback requires a non-empty final reply."
            )
        final = final.strip()
    else:
        final = ""

    record = {
        "sender_id": sender_id.strip(),
        "incoming": incoming.strip(),
        "generated": generated.strip(),
        "final": final,
        "feedback": feedback
    }

    records = load_feedback(path)
    records.append(record)

    path = Path(path)
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            records,
            file,
            ensure_ascii=False,
            indent=2
        )

    return record
