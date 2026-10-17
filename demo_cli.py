"""Thin local CLI for the existing personalized response pipeline.

The wrapper intentionally contains no retrieval, memory, planning, or style
logic.  It only hides verbose internal diagnostics by default, presents the
returned artifacts in ``--debug`` mode, and forwards human feedback to the
existing public entry points in ``main``.
"""

import argparse
import io
import json
from contextlib import nullcontext, redirect_stdout

import main


EXIT_WORDS = {"exit", "quit"}
FEEDBACK_CHOICES = {
    "a": "accepted",
    "accept": "accepted",
    "e": "edited",
    "edit": "edited",
    "r": "rejected",
    "reject": "rejected",
}


def _print_json(label, value):
    print(f"\n{label}")
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _print_debug(sender_id, result):
    _print_json(
        "Working Memory (before this reply)",
        main.get_working_memory(sender_id).get_messages(),
    )
    _print_json("Retrieved Long-Term Memories", result["relevant_memories"])
    _print_json("Planner Result", result["plan"])
    print("\nDraft")
    print(result["draft"])
    _print_json("Retrieved Style Samples", result["style_samples"])


def _generate(sender_id, incoming, debug):
    diagnostics = io.StringIO()
    output_context = nullcontext() if debug else redirect_stdout(diagnostics)
    with output_context:
        return main.process_user_message(sender_id, incoming)


def _collect_feedback(sender_id, result):
    while True:
        choice = input("\nFeedback [accept/edit/reject]: ").strip().lower()
        feedback = FEEDBACK_CHOICES.get(choice)
        if feedback is None:
            print("Please enter accept, edit, or reject.")
            continue

        edited_reply = None
        if feedback == "edited":
            edited_reply = input("Final edited reply: ").strip()
            if not edited_reply:
                print("Edited reply cannot be empty.")
                continue

        return main.finalize_response(
            sender_id,
            result,
            feedback,
            edited_reply=edited_reply,
        )


def run_cli(debug=False):
    print("Personalized Conversational Agent (local demo)")
    print("Type exit as the sender ID or message to stop.")

    while True:
        try:
            sender_id = input("\nSender ID: ").strip()
            if sender_id.lower() in EXIT_WORDS:
                return 0
            if not sender_id:
                print("Sender ID cannot be empty.")
                continue

            incoming = input("Incoming message: ").strip()
            if incoming.lower() in EXIT_WORDS:
                return 0
            if not incoming:
                print("Incoming message cannot be empty.")
                continue

            print("Generating reply...")
            try:
                result = _generate(sender_id, incoming, debug)
            except Exception as error:
                print(f"Generation failed: {error}")
                continue

            if debug:
                _print_debug(sender_id, result)

            print("\nPersonalized Reply")
            print(result["personalized_reply"])

            try:
                record = _collect_feedback(sender_id, result)
            except (OSError, ValueError) as error:
                print(f"Feedback failed: {error}")
                continue

            print(f"Feedback saved: {record['feedback']}")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the local personalized conversational agent demo."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "show retrieval, Working Memory, planner, draft, style samples, "
            "and existing internal timing diagnostics"
        ),
    )
    return parser.parse_args()


def main_entry():
    args = parse_args()
    return run_cli(debug=args.debug)


if __name__ == "__main__":
    raise SystemExit(main_entry())
