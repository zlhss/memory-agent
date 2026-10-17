"""Shared context and V2 interface checks with no models, HTTP, or file writes."""

import builtins
import copy
import importlib.util
import io
import json
import runpy
import sys
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import agent_context


class MemoryV2InterfaceMockTest(unittest.TestCase):
    def setUp(self):
        context_patch = patch.dict(agent_context.working_memories, {}, clear=True)
        context_patch.start()
        self.addCleanup(context_patch.stop)

        self.storage = types.ModuleType("memory_search")
        self.storage.retrieve = Mock(return_value=[])
        self.storage.get_sender_memories = Mock(return_value=[])
        self.storage.apply_memory_action = Mock(return_value=True)
        self.requests = types.ModuleType("requests")
        self.requests.post = Mock()
        self.requests.exceptions = types.SimpleNamespace(Timeout=TimeoutError)
        self._respond({"facts": []})
        style = types.ModuleType("style_memory")
        style.retrieve_style_samples = Mock(return_value=[])
        sklearn = types.ModuleType("sklearn")
        sklearn.__path__ = []
        metrics = types.ModuleType("sklearn.metrics")
        metrics.__path__ = []
        pairwise = types.ModuleType("sklearn.metrics.pairwise")
        pairwise.cosine_similarity = Mock()
        modules_patch = patch.dict(sys.modules, {
            "memory_search": self.storage,
            "requests": self.requests,
            "style_memory": style,
            "sklearn": sklearn,
            "sklearn.metrics": metrics,
            "sklearn.metrics.pairwise": pairwise,
        })
        modules_patch.start()
        self.addCleanup(modules_patch.stop)
        sys.modules.pop("main", None)
        self.ollama = self._load("ollama_client")
        self.v2 = self._load("memory_v2")
        self.v2_imported_main = "main" in sys.modules
        self.main = self._load("main")

    @staticmethod
    def _load(name):
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).with_name(f"{name}.py"),
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    def _respond(self, payload):
        response = Mock()
        response.json.return_value = {"response": json.dumps(payload)}
        self.requests.post.return_value = response

    def _seed_timing(self):
        self._respond({"ok": True})
        with redirect_stdout(io.StringIO()):
            self.ollama.call_ollama("existing prompt", stage="Existing Stage")
        self._respond({"facts": []})

    def test_main_reexports_the_single_context_registry_and_functions(self):
        self.assertIs(self.main.working_memories, agent_context.working_memories)
        self.assertIs(self.v2.agent_context, agent_context)
        for name in (
            "get_working_memory", "reset_working_memories", "is_valid_contact_memory",
        ):
            self.assertIs(getattr(self.main, name), getattr(agent_context, name))
        memory = self.main.get_working_memory("target")
        memory.add_message("user", "shared message")
        self.assertIs(self.v2.agent_context.get_working_memory("target"), memory)

    def test_reset_from_either_module_clears_the_same_registry_in_place(self):
        registry = self.main.working_memories
        previous = self.main.get_working_memory("target")
        for reset in (self.main.reset_working_memories, self.v2.agent_context.reset_working_memories):
            reset()
            self.assertIs(self.main.working_memories, registry)
            self.assertEqual(registry, {})
            current = self.v2.agent_context.get_working_memory("target")
            self.assertIs(current, self.main.get_working_memory("target"))
            self.assertIsNot(current, previous)
            self.assertEqual(current.get_messages(), [])
            previous = current

    def test_sender_isolation_and_six_message_capacity_are_preserved(self):
        target = self.main.get_working_memory("target")
        sentinel = self.v2.agent_context.get_working_memory("sentinel")
        sentinel.add_message("user", "sentinel message")
        for index in range(8):
            target.add_message("user", str(index))
        self.assertEqual(target.max_messages, 6)
        self.assertEqual(target.get_messages(), [
            {"role": "user", "content": str(index)} for index in range(2, 8)
        ])
        self.assertIsNot(target, sentinel)
        self.assertEqual(sentinel.get_messages(), [
            {"role": "user", "content": "sentinel message"}
        ])

    def test_contact_schema_keeps_the_original_prefix_rules(self):
        for value, expected in (
            (None, False), ([], False), (1, False), ("", False),
            (" 联系人喜欢摄影", False), ("用户喜欢摄影", False),
            ("联系人", True), ("联系人 ", True), ("联系人喜欢摄影", True),
        ):
            with self.subTest(value=value):
                self.assertEqual(self.main.is_valid_contact_memory(value), expected)
                self.assertEqual(self.v2.agent_context.is_valid_contact_memory(value), expected)

    def test_v2_import_does_not_require_or_load_main(self):
        self.assertFalse(self.v2_imported_main)
        self.assertNotIn("main", vars(self.v2))
        original_import = builtins.__import__

        def reject_main(name, *args, **kwargs):
            if name == "main":
                raise AssertionError("memory_v2 must not import main")
            return original_import(name, *args, **kwargs)

        sys.modules.pop("main", None)
        with patch("builtins.__import__", side_effect=reject_main):
            imported = self._load("memory_v2")
        self.assertIs(imported.agent_context, agent_context)
        self.assertNotIn("main", sys.modules)

    def test_both_import_orders_preserve_the_same_existing_context(self):
        memory = agent_context.get_working_memory("target")
        memory.add_message("user", "before imports")
        for order in (("main", "memory_v2"), ("memory_v2", "main")):
            with self.subTest(order=order):
                modules = {name: self._load(name) for name in order}
                self.assertIs(modules["main"].working_memories, agent_context.working_memories)
                self.assertIs(modules["memory_v2"].agent_context.get_working_memory("target"), memory)

    def test_main_script_and_imported_modules_share_one_registry(self):
        memory = self.main.get_working_memory("target")
        memory.add_message("user", "before CLI startup")
        with patch("builtins.input", return_value="exit") as cli_input:
            script = runpy.run_path(
                str(Path(__file__).with_name("main.py")), run_name="__main__",
            )
        cli_input.assert_called_once()
        self.assertIs(script["working_memories"], self.main.working_memories)
        self.assertIs(script["get_working_memory"]("target"), memory)
        self.assertIs(self.v2.agent_context.get_working_memory("target"), memory)

    def test_core_uses_the_explicit_snapshot_for_extraction_and_validation(self):
        global_memory = self.main.get_working_memory("target")
        global_memory.add_message("user", "GLOBAL_ONLY")
        snapshot = [{"role": "user", "content": "EXPLICIT_SNAPSHOT"}]
        original_snapshot = copy.deepcopy(snapshot)
        fact = {
            "source_span": "I like photography", "kind": "persistent",
            "operation_hint": "add", "fact": "联系人喜欢摄影", "constraints": [],
        }
        self._respond({"facts": [fact]})
        with (
            patch.object(agent_context, "get_working_memory", side_effect=AssertionError("global read")),
            patch.object(self.v2, "extract_atomic_facts", wraps=self.v2.extract_atomic_facts) as extract,
            patch.object(self.v2, "retrieve_fact_candidates", return_value={"candidates": []}),
            patch.object(self.v2, "align_fact_to_memory", return_value={
                "decision": "NEW", "target_index": None, "memory": fact["fact"],
            }),
            patch.object(self.v2, "check_add_evidence", return_value={"supported": True}) as evidence,
            patch.object(self.v2, "check_memory_worthiness", return_value={"should_save": True}) as worthiness,
            redirect_stdout(io.StringIO()),
        ):
            result = self.v2.run_memory_lifecycle_v2("target", "I like photography", snapshot)

        extract.assert_called_once_with("target", "I like photography", snapshot)
        self.assertIs(extract.call_args.args[2], snapshot)
        evidence.assert_called_once_with("I like photography", fact["fact"], snapshot)
        worthiness.assert_called_once_with(fact["fact"], "I like photography", snapshot)
        self.storage.apply_memory_action.assert_called_once_with(
            "target", {"action": "ADD", "memory": fact["fact"]},
        )
        prompt = self.requests.post.call_args.kwargs["json"]["prompt"]
        self.assertIn("EXPLICIT_SNAPSHOT", prompt)
        self.assertNotIn("GLOBAL_ONLY", prompt)
        self.assertIs(result["working_messages"], snapshot)
        self.assertEqual(snapshot, original_snapshot)
        self.assertEqual(global_memory.get_messages(), [{"role": "user", "content": "GLOBAL_ONLY"}])

    def test_core_appends_real_timing_records_without_reset_or_summary(self):
        self._seed_timing()
        previous_calls = self.ollama.get_ollama_timings()
        previous_prompts = self.ollama.get_ollama_profiled_prompts()
        output = io.StringIO()
        with (
            patch.object(self.v2, "reset_ollama_timings", wraps=self.v2.reset_ollama_timings) as reset,
            patch.object(self.v2, "get_ollama_timing_summary", wraps=self.v2.get_ollama_timing_summary) as summary,
            redirect_stdout(output),
        ):
            result = self.v2.run_memory_lifecycle_v2("target", "hello", [])
        reset.assert_not_called()
        summary.assert_not_called()
        calls = self.ollama.get_ollama_timings()
        prompts = self.ollama.get_ollama_profiled_prompts()
        self.assertEqual(calls[:1], previous_calls)
        self.assertEqual(prompts[:1], previous_prompts)
        self.assertEqual([item["stage"] for item in calls], ["Existing Stage", "Atomic Fact Extractor"])
        self.assertNotIn("timing_profile", result)
        self.assertNotIn("TIMING_SUMMARY:", output.getvalue())

    def test_core_propagates_failure_without_clearing_existing_timing(self):
        self._seed_timing()
        previous_calls = self.ollama.get_ollama_timings()
        previous_prompts = self.ollama.get_ollama_profiled_prompts()
        error = RuntimeError("atomic extraction failed")
        with patch.object(self.v2, "extract_atomic_facts", side_effect=error):
            with self.assertRaises(RuntimeError) as caught:
                self.v2.run_memory_lifecycle_v2("target", "hello", [])
        self.assertIs(caught.exception, error)
        self.assertEqual(self.ollama.get_ollama_timings(), previous_calls)
        self.assertEqual(self.ollama.get_ollama_profiled_prompts(), previous_prompts)
        self.storage.apply_memory_action.assert_not_called()

    def test_core_preserves_existing_snapshot_validation(self):
        with self.assertRaisesRegex(ValueError, "working_messages must be a list"):
            self.v2.run_memory_lifecycle_v2("target", "hello", None)
        self.requests.post.assert_not_called()
        self.storage.apply_memory_action.assert_not_called()

    def test_standalone_entry_keeps_snapshot_result_and_timing_contract(self):
        self._seed_timing()
        memory = self.main.get_working_memory("target")
        memory.add_message("user", "shared history")
        output = io.StringIO()
        with (
            patch.object(self.v2, "reset_ollama_timings", wraps=self.v2.reset_ollama_timings) as reset,
            redirect_stdout(output),
        ):
            result = self.v2.process_user_message_v2("target", "hello")
        reset.assert_called_once_with()
        self.assertEqual(result, {
            "pipeline_version": "v2", "sender_id": "target", "incoming": "hello",
            "working_messages": [{"role": "user", "content": "shared history"}],
            "atomic_facts": [], "fact_results": [], "coverage": {"uncovered_fact_indices": []},
            "processed_actions": [], "action_results": [], "consistency_results": [],
            "final_memories": [], "timing_profile": result["timing_profile"],
        })
        self.assertIsNot(result["working_messages"], memory.get_messages())
        self.assertEqual(result["timing_profile"]["outcome"], "success")
        self.assertEqual(result["timing_profile"]["total_llm_calls"], 1)
        self.assertEqual([item["stage"] for item in result["timing_profile"]["calls"]], ["Atomic Fact Extractor"])
        self.assertEqual(output.getvalue().count("MEMORY_V2_TIMING_SUMMARY:"), 1)
        self.storage.apply_memory_action.assert_not_called()

    def test_standalone_failure_still_propagates_and_emits_error_summary(self):
        error = RuntimeError("atomic extraction failed")
        output = io.StringIO()
        with patch.object(self.v2, "extract_atomic_facts", side_effect=error):
            with redirect_stdout(output), self.assertRaises(RuntimeError) as caught:
                self.v2.process_user_message_v2("target", "hello")
        self.assertIs(caught.exception, error)
        summary_line = next(line for line in output.getvalue().splitlines() if line.startswith("MEMORY_V2_TIMING_SUMMARY:"))
        self.assertEqual(json.loads(summary_line.split(":", 1)[1])["outcome"], "error")
        self.storage.apply_memory_action.assert_not_called()

    def test_standalone_validates_inputs_before_reading_global_context(self):
        for sender_id, user_text in (("", "hello"), ("target", "")):
            with self.subTest(sender_id=sender_id, user_text=user_text):
                with (
                    patch.object(agent_context, "get_working_memory") as get_memory,
                    redirect_stdout(io.StringIO()),
                    self.assertRaises(ValueError),
                ):
                    self.v2.process_user_message_v2(sender_id, user_text)
                get_memory.assert_not_called()
        self.assertEqual(agent_context.working_memories, {})


if __name__ == "__main__":
    unittest.main()
