"""Response orchestration regressions without third-party packages or models.

Load the real main.py with scoped service doubles. Existing integration tests
still import the real services; these doubles do not escape the loader.
"""

import importlib.util
import io
import sys
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, call, patch


def load_main_without_services():
    exports = {
        "memory_search": ("retrieve", "apply_memory_action"),
        "style_memory": ("retrieve_style_samples",),
        "ollama_client": (
            "get_ollama_timing_summary", "get_ollama_timings",
            "reset_ollama_timings", "resolve_contextual_query",
            "decide_memory_actions", "check_action_coverage",
            "check_add_evidence", "filter_relevant_memories",
            "check_memory_duplicate", "check_memory_consistency",
            "check_memory_worthiness", "plan_response",
            "generate_draft_reply", "rewrite_reply_style",
        ),
    }
    doubles = {}
    for name, functions in exports.items():
        module = types.ModuleType(name)
        for function in functions:
            setattr(module, function, Mock(name=function))
        doubles[name] = module

    spec = importlib.util.spec_from_file_location(
        "_response_pipeline_test_main",
        Path(__file__).with_name("main.py"),
    )
    main = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, doubles):
        spec.loader.exec_module(main)
    main.record_feedback = Mock(name="record_feedback")
    return main


class ResponsePipelineMockTest(unittest.TestCase):
    def setUp(self):
        self.main = load_main_without_services()
        self.main.reset_working_memories()
        self.addCleanup(self.main.reset_working_memories)
        self.sender_id = "pipeline_target"
        self.user_text = "Pause that course for now."
        self.memories = ["联系人正在学习摄影"]
        self.working_memory = self.main.get_working_memory(self.sender_id)
        self.working_memory.add_message("user", "The course is busy.")
        self.messages = self.working_memory.get_messages().copy()
        self.style_samples = [{"incoming": "Earlier message", "reply": "OK."}]
        self.plan = {"intent": "acknowledge", "points": ["pause"], "tone": "calm"}
        self.draft = "You can pause the course."
        self.reply = "OK, pause it for now."

        self.main.retrieve_style_samples.return_value = self.style_samples
        self.main.plan_response.return_value = self.plan
        self.main.generate_draft_reply.return_value = self.draft
        self.main.rewrite_reply_style.return_value = self.reply
        self.main.resolve_contextual_query.return_value = "resolved course query"
        self.main.retrieve.side_effect = [[(0.9, self.memories[0])], []]
        self.main.filter_relevant_memories.return_value = self.memories
        self.main.decide_memory_actions.return_value = [
            {"action": "DELETE", "target_index": 0, "memory": None}
        ]
        self.main.check_action_coverage.return_value = {"missing_actions": []}
        self.main.apply_memory_action.return_value = True
        self.main.get_ollama_timings.return_value = []
        self.main.get_ollama_timing_summary.return_value = {
            "total_llm_calls": 0,
            "stage_timings": [],
            "total_llm_elapsed_seconds": 0.0,
            "total_request_elapsed_seconds": 0.0,
        }

        self.sequence = Mock()
        for function in (
            "retrieve_style_samples", "plan_response", "generate_draft_reply",
            "rewrite_reply_style", "decide_memory_actions", "apply_memory_action",
        ):
            self.sequence.attach_mock(getattr(self.main, function), function)

    def _expected_reply_calls(self):
        return [
            call.retrieve_style_samples(self.sender_id, self.user_text, top_k=3),
            call.plan_response(
                self.sender_id, self.user_text, self.memories, self.messages,
            ),
            call.generate_draft_reply(
                self.sender_id, self.user_text, self.memories,
                self.messages, self.plan,
            ),
            call.rewrite_reply_style(self.draft, self.user_text, self.style_samples),
        ]

    def test_pipeline_preserves_call_order_arguments_and_artifacts(self):
        result = self.main.build_personalized_reply(
            self.sender_id, self.user_text, self.memories, self.messages,
        )

        self.assertEqual(self.sequence.mock_calls, self._expected_reply_calls())
        self.assertEqual(set(result), {"style_samples", "plan", "draft", "personalized_reply"})
        self.assertIs(result["style_samples"], self.style_samples)
        self.assertIs(result["plan"], self.plan)
        self.assertEqual(result["draft"], self.draft)
        self.assertEqual(result["personalized_reply"], self.reply)

    def test_empty_style_samples_still_reach_the_rewriter(self):
        self.style_samples = []
        self.main.retrieve_style_samples.return_value = self.style_samples
        result = self.main.build_personalized_reply(
            self.sender_id, self.user_text, self.memories, self.messages,
        )

        self.assertEqual(self.sequence.mock_calls, self._expected_reply_calls())
        self.assertIs(result["style_samples"], self.style_samples)
        self.assertEqual(result["personalized_reply"], self.reply)

    def test_process_preserves_fields_and_runs_memory_after_reply_once(self):
        with (
            patch.object(
                self.main, "build_personalized_reply",
                wraps=self.main.build_personalized_reply,
            ) as build_reply,
            redirect_stdout(io.StringIO()),
        ):
            result = self.main.process_user_message(self.sender_id, self.user_text)

        build_reply.assert_called_once_with(
            self.sender_id, self.user_text, self.memories, self.messages,
        )
        action = {
            "action": "DELETE", "target_index": 0, "memory": None,
            "target": self.memories[0],
        }
        self.assertEqual(self.sequence.mock_calls, self._expected_reply_calls() + [
            call.decide_memory_actions(
                self.sender_id, self.user_text, self.memories, self.messages,
            ),
            call.apply_memory_action(self.sender_id, action),
        ])
        self.assertEqual(result, {
            "sender_id": self.sender_id,
            "incoming": self.user_text,
            "contextual_query": "resolved course query",
            "candidate_memories": self.memories,
            "relevant_memories": self.memories,
            "style_samples": self.style_samples,
            "plan": self.plan,
            "draft": self.draft,
            "personalized_reply": self.reply,
            "processed_actions": [action],
            "action_results": [{"action": action, "success": True, "error": None}],
            "consistency_results": [],
            "consistency_errors": [],
            "timing_profile": {
                "outcome": "success", "calls": [],
                **self.main.get_ollama_timing_summary.return_value,
            },
        })
        self.main.reset_ollama_timings.assert_called_once_with()
        self.assertEqual(self.working_memory.get_messages(), self.messages)
        self.main.record_feedback.assert_not_called()

    def _assert_stage_failure(self, stage):
        error = RuntimeError("reply stage failed")
        getattr(self.main, stage).side_effect = error
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(RuntimeError) as caught:
            self.main.process_user_message(self.sender_id, self.user_text)

        self.assertIs(caught.exception, error)
        expected_calls = self._expected_reply_calls()
        failed_index = [item[0] for item in expected_calls].index(stage)
        self.assertEqual(self.sequence.mock_calls, expected_calls[:failed_index + 1])
        self.main.decide_memory_actions.assert_not_called()
        self.main.check_action_coverage.assert_not_called()
        self.main.apply_memory_action.assert_not_called()
        self.main.record_feedback.assert_not_called()
        self.assertEqual(self.working_memory.get_messages(), self.messages)
        self.main.reset_ollama_timings.assert_called_once_with()
        self.main.get_ollama_timing_summary.assert_called_once()
        self.assertIn('"outcome": "error"', output.getvalue())

    def test_style_retrieval_failure_propagates_before_memory_management(self):
        self._assert_stage_failure("retrieve_style_samples")

    def test_planner_failure_propagates_before_memory_management(self):
        self._assert_stage_failure("plan_response")

    def test_draft_failure_propagates_before_memory_management(self):
        self._assert_stage_failure("generate_draft_reply")

    def test_rewriter_failure_propagates_before_memory_management(self):
        self._assert_stage_failure("rewrite_reply_style")


if __name__ == "__main__":
    unittest.main()
