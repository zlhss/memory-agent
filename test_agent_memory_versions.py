"""Agent integration with mocked inference and real CRUD in a temp directory."""

import copy
import functools
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import agent_context


PROJECT_ROOT = Path(__file__).resolve().parent


class AgentMemoryVersionsTest(unittest.TestCase):
    def setUp(self):
        context_patch = patch.dict(agent_context.working_memories, {}, clear=True)
        context_patch.start()
        self.addCleanup(context_patch.stop)
        temporary = tempfile.TemporaryDirectory(prefix="memory-agent-v3-")
        self.directory = Path(temporary.name).resolve()
        self.assertEqual(self.directory.parent, Path(tempfile.gettempdir()).resolve())
        self.addCleanup(temporary.cleanup)
        previous_directory = Path.cwd()
        os.chdir(self.directory)
        self.addCleanup(os.chdir, previous_directory)
        self.sender = "target"
        self.incoming = "我喜欢摄影"
        self.seed = {
            self.sender: ["联系人住在南京"],
            "sentinel": ["联系人喜欢哨兵音乐"],
        }
        self.memory_file = self.directory / "memories.json"
        self.feedback_file = self.directory / "feedback.json"
        self.memory_file.write_text(json.dumps(self.seed, ensure_ascii=False), encoding="utf-8")
        self.feedback_file.write_text("[]", encoding="utf-8")

        embedding = types.ModuleType("sentence_transformers")
        embedding.SentenceTransformer = Mock(return_value=types.SimpleNamespace(
            encode=Mock(side_effect=lambda messages: list(messages)),
        ))
        sklearn = types.ModuleType("sklearn")
        sklearn.__path__ = []
        metrics = types.ModuleType("sklearn.metrics")
        metrics.__path__ = []
        pairwise = types.ModuleType("sklearn.metrics.pairwise")
        pairwise.cosine_similarity = Mock(side_effect=lambda queries, documents: [
            [1.0] * len(documents) for _ in queries
        ])
        self.style_samples = [{"incoming": "previous", "reply": "OK.", "score": 0.9}]
        self.style = types.ModuleType("style_memory")
        self.style.retrieve_style_samples = Mock(return_value=self.style_samples)
        self.requests = types.ModuleType("requests")
        self.requests.exceptions = types.SimpleNamespace(Timeout=TimeoutError)
        self.requests.post = Mock(side_effect=self._fake_post)
        modules_patch = patch.dict(sys.modules, {
            "sentence_transformers": embedding,
            "sklearn": sklearn,
            "sklearn.metrics": metrics,
            "sklearn.metrics.pairwise": pairwise,
            "style_memory": self.style,
            "requests": self.requests,
        })
        modules_patch.start()
        self.addCleanup(modules_patch.stop)
        self.storage = self._load("memory_search")
        self.ollama = self._load("ollama_client")
        self.v2 = self._load("memory_v2")
        self.main = self._load("main")
        self.feedback = self._load("feedback_store")
        self.main.record_feedback = functools.partial(
            self.feedback.record_feedback, path=self.feedback_file,
        )
        self.relevant_indices = [0]
        self.facts = [self._fact(self.incoming, "联系人喜欢摄影")]
        self.alignment = "NEW"
        self.v1_actions = [{"action": "ADD", "target_index": None, "memory": "联系人喜欢摄影"}]
        self._seed_context()

    @staticmethod
    def _load(name):
        spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _fact(source, memory, kind="persistent", operation_hint="add"):
        return {
            "source_span": source, "kind": kind, "operation_hint": operation_hint,
            "fact": memory, "constraints": [],
        }

    def _seed_context(self):
        agent_context.reset_working_memories()
        memory = agent_context.get_working_memory(self.sender)
        memory.add_message("user", "ORIGINAL_CONTEXT")
        memory.add_message("assistant", "Earlier reply.")
        agent_context.get_working_memory("sentinel").add_message("user", "SENTINEL_CONTEXT")

    def _fake_post(self, url, **request):
        prompt = request["json"]["prompt"]
        responses = (
            ("上下文检索查询解析器", {"query": "联系人相关信息"}),
            ("长期记忆检索的相关性判断器", {"relevant_indices": self.relevant_indices}),
            ("Response Planner", {"intent": "acknowledge", "points": ["acknowledge preference"], "tone": "calm"}),
            ("Draft Generator", {"draft": "Got it, you like photography."}),
            ("Style Rewriter", {"reply": "OK, photography it is."}),
            ("原子事实提取器", {"facts": self.facts}),
            ("单事实对齐器", {
                "decision": self.alignment,
                "target_index": 0 if self.alignment in {"UPDATE", "DELETE", "DUPLICATE"} else None,
                "memory": "联系人占位文本" if self.alignment in {"NEW", "UPDATE"} else None,
            }),
            ("长期记忆管理器", {"actions": self.v1_actions}),
            ("Action Coverage Checker", {"missing_actions": []}),
            ("ADD Evidence Gate", {"supported": True}),
            ("长期记忆类型分类器", {"memory_type": "preference"}),
            ("长期记忆去重判断器", {"is_duplicate": False, "duplicate_index": None}),
            ("长期记忆一致性检查器", {"actions": []}),
        )
        for marker, payload in responses:
            if marker in prompt:
                response = Mock()
                response.json.return_value = {"response": json.dumps(payload)}
                return response
        raise AssertionError(f"Unexpected inference prompt: {prompt[:100]}")

    def _generate(self, version=None):
        output = io.StringIO()
        with redirect_stdout(output):
            if version is None:
                result = self.main.process_user_message(self.sender, self.incoming)
            else:
                result = self.main.process_user_message(
                    self.sender, self.incoming, memory_version=version,
                )
        return result, output.getvalue()

    def _disk_memories(self):
        return json.loads(self.memory_file.read_text(encoding="utf-8"))

    def test_default_and_explicit_v1_preserve_the_original_result_contract(self):
        with (
            patch.object(self.main, "run_memory_lifecycle_v1", wraps=self.main.run_memory_lifecycle_v1) as v1,
            patch.object(self.v2, "run_memory_lifecycle_v2") as v2,
        ):
            default, _ = self._generate()
            first_stages = [item["stage"] for item in default["timing_profile"]["calls"]]
            self.storage.reset_memories(copy.deepcopy(self.seed))
            self._seed_context()
            explicit, _ = self._generate("v1")
        self.assertEqual(v1.call_count, 2)
        v2.assert_not_called()
        self.assertEqual(
            {key: value for key, value in default.items() if key != "timing_profile"},
            {key: value for key, value in explicit.items() if key != "timing_profile"},
        )
        self.assertEqual(set(default), {
            "sender_id", "incoming", "contextual_query", "candidate_memories", "relevant_memories",
            "style_samples", "plan", "draft", "personalized_reply", "processed_actions",
            "action_results", "consistency_results", "consistency_errors", "timing_profile",
        })
        self.assertEqual(first_stages, [item["stage"] for item in explicit["timing_profile"]["calls"]])
        self.assertEqual(default["action_results"], [{
            "action": {"action": "ADD", "target_index": None, "memory": "联系人喜欢摄影"},
            "success": True, "error": None,
        }])
        self.assertEqual(self._disk_memories()[self.sender], self.seed[self.sender] + ["联系人喜欢摄影"])

    def test_explicit_v2_generates_a_reply_and_persists_memory_once(self):
        with (
            patch.object(self.main, "run_memory_lifecycle_v1") as v1,
            patch.object(self.v2, "run_memory_lifecycle_v2", wraps=self.v2.run_memory_lifecycle_v2) as v2,
            patch.object(self.storage, "apply_memory_action", wraps=self.storage.apply_memory_action) as apply,
        ):
            result, _ = self._generate("v2")
        v1.assert_not_called()
        v2.assert_called_once()
        apply.assert_called_once_with(self.sender, {"action": "ADD", "memory": "联系人喜欢摄影"})
        self.assertEqual(result["personalized_reply"], "OK, photography it is.")
        self.assertEqual(result["style_samples"], self.style_samples)
        self.assertEqual(result["memory_version"], "v2")
        self.assertEqual(result["memory_v2_details"]["action_results"], [{
            "action": {"action": "ADD", "memory": "联系人喜欢摄影"}, "success": True,
        }])
        self.assertEqual(self._disk_memories()[self.sender], self.seed[self.sender] + ["联系人喜欢摄影"])
        self.assertEqual(result["memory_v2_details"]["final_memories"], self._disk_memories()[self.sender])

    def test_invalid_versions_fail_before_context_inference_writes_or_timing_reset(self):
        initial_memories = self.memory_file.read_bytes()
        with (
            patch.object(self.main, "reset_ollama_timings") as reset,
            patch.object(self.main, "get_working_memory") as get_memory,
            patch.object(self.storage, "apply_memory_action") as apply,
        ):
            for version in ("v3", "V2", "", None, [], True):
                with self.subTest(version=version):
                    with self.assertRaisesRegex(ValueError, "memory_version must be"):
                        self.main.process_user_message(self.sender, self.incoming, memory_version=version)
        reset.assert_not_called()
        get_memory.assert_not_called()
        apply.assert_not_called()
        self.requests.post.assert_not_called()
        self.assertEqual(self.memory_file.read_bytes(), initial_memories)

    def test_each_reply_failure_prevents_both_lifecycles_for_both_versions(self):
        initial_memories = self.memory_file.read_bytes()
        for version in ("v1", "v2"):
            for stage in ("retrieve_style_samples", "plan_response", "generate_draft_reply", "rewrite_reply_style"):
                with self.subTest(version=version, stage=stage):
                    error = RuntimeError(f"{stage} failed")
                    with (
                        patch.object(self.main, stage, side_effect=error),
                        patch.object(self.main, "run_memory_lifecycle_v1") as v1,
                        patch.object(self.v2, "run_memory_lifecycle_v2") as v2,
                        patch.object(self.storage, "apply_memory_action") as apply,
                        self.assertRaises(RuntimeError) as caught,
                    ):
                        self._generate(version)
                    self.assertIs(caught.exception, error)
                    v1.assert_not_called()
                    v2.assert_not_called()
                    apply.assert_not_called()
                    self.assertEqual(self.memory_file.read_bytes(), initial_memories)

    def test_v2_failure_propagates_without_falling_back_to_v1(self):
        error = RuntimeError("atomic extraction failed")
        initial_memories = self.memory_file.read_bytes()
        output = io.StringIO()
        with (
            patch.object(self.v2, "extract_atomic_facts", side_effect=error),
            patch.object(self.main, "run_memory_lifecycle_v1") as v1,
            patch.object(self.v2, "run_memory_lifecycle_v2", wraps=self.v2.run_memory_lifecycle_v2) as v2,
            redirect_stdout(output),
            self.assertRaises(RuntimeError) as caught,
        ):
            self.main.process_user_message(self.sender, self.incoming, memory_version="v2")
        self.assertIs(caught.exception, error)
        v1.assert_not_called()
        v2.assert_called_once()
        self.assertEqual(self.memory_file.read_bytes(), initial_memories)
        self.assertEqual(self._summary(output.getvalue())["outcome"], "error")

    def test_v2_later_write_failure_exposes_error_and_preserves_prior_persisted_write(self):
        self.facts.append(self._fact("我喜欢登山", "联系人喜欢登山"))
        original_apply = self.storage.apply_memory_action
        error = OSError("second write failed")

        def fail_second(sender_id, action):
            if action["memory"] == "联系人喜欢登山":
                raise error
            return original_apply(sender_id, action)

        output = io.StringIO()
        with (
            patch.object(self.storage, "apply_memory_action", side_effect=fail_second) as apply,
            patch.object(self.main, "run_memory_lifecycle_v1") as v1,
            redirect_stdout(output),
            self.assertRaises(OSError) as caught,
        ):
            self.main.process_user_message(self.sender, self.incoming, memory_version="v2")
        self.assertIs(caught.exception, error)
        self.assertEqual(apply.call_count, 2)
        v1.assert_not_called()
        self.assertEqual(self._disk_memories()[self.sender], self.seed[self.sender] + ["联系人喜欢摄影"])
        self.assertEqual(self.storage.get_sender_memories(self.sender), self._disk_memories()[self.sender])
        self.assertEqual(self._disk_memories()["sentinel"], self.seed["sentinel"])
        self.assertEqual(self._summary(output.getvalue())["outcome"], "error")

    @staticmethod
    def _summary(output):
        line = next(line for line in output.splitlines() if line.startswith("OLLAMA_TIMING_SUMMARY:"))
        return json.loads(line.split(":", 1)[1])

    def test_v2_keeps_all_reply_and_memory_timing_with_only_one_outer_reset(self):
        with (
            patch.object(self.main, "reset_ollama_timings", wraps=self.main.reset_ollama_timings) as reset,
            patch.object(self.v2, "reset_ollama_timings") as inner_reset,
            patch.object(self.v2, "get_ollama_timing_summary") as inner_summary,
            patch.object(self.v2, "process_user_message_v2", side_effect=AssertionError("standalone entry")) as standalone,
        ):
            result, output = self._generate("v2")
        reset.assert_called_once_with()
        inner_reset.assert_not_called()
        inner_summary.assert_not_called()
        standalone.assert_not_called()
        expected_stages = [
            "Contextual Query Resolver", "Relevance Gate", "Planner", "Draft", "Style Rewriter",
            "Atomic Fact Extractor", "Fact Alignment", "ADD Evidence Gate", "Worthiness Gate",
        ]
        profile = result["timing_profile"]
        self.assertEqual([item["stage"] for item in profile["calls"]], expected_stages)
        self.assertEqual(profile["total_llm_calls"], len(expected_stages))
        self.assertEqual(output.count("TIMING_SUMMARY:"), 1)
        self.assertEqual(self._summary(output), profile)
        self.assertNotIn("timing_profile", result["memory_v2_details"])

    def test_v2_receives_the_same_stable_snapshot_as_reply_generation(self):
        original_messages = copy.deepcopy(agent_context.get_working_memory(self.sender).get_messages())
        original_plan = self.main.plan_response

        def plan_and_change_global(sender, incoming, memories, snapshot):
            memory = agent_context.get_working_memory(sender)
            memory.get_messages()[0]["content"] = "MUTATED_GLOBAL_CONTEXT"
            memory.add_message("user", "LATE_GLOBAL_CONTEXT")
            return original_plan(sender, incoming, memories, snapshot)

        with (
            patch.object(self.main, "plan_response", side_effect=plan_and_change_global) as plan,
            patch.object(self.main, "generate_draft_reply", wraps=self.main.generate_draft_reply) as draft,
            patch.object(self.v2, "run_memory_lifecycle_v2", wraps=self.v2.run_memory_lifecycle_v2) as core,
        ):
            result, _ = self._generate("v2")
        snapshot = core.call_args.args[2]
        self.assertIs(plan.call_args.args[3], snapshot)
        self.assertIs(draft.call_args.args[3], snapshot)
        self.assertEqual(snapshot, original_messages)
        self.assertIs(result["memory_v2_details"]["working_messages"], snapshot)
        atomic_prompt = next(
            item["prompt"] for item in self.ollama.get_ollama_profiled_prompts()
            if item["stage"] == "Atomic Fact Extractor"
        )
        self.assertIn("ORIGINAL_CONTEXT", atomic_prompt)
        self.assertNotIn("MUTATED_GLOBAL_CONTEXT", atomic_prompt)
        self.assertNotIn("LATE_GLOBAL_CONTEXT", atomic_prompt)

    def test_v2_indices_and_consistency_records_remain_in_the_versioned_contract(self):
        self.seed[self.sender].append("联系人喜欢音乐")
        self.storage.reset_memories(copy.deepcopy(self.seed))
        self.relevant_indices = [1]
        self.incoming = "我搬到苏州了"
        self.facts = [self._fact(self.incoming, "联系人住在苏州", "persistent_change", "update")]
        self.alignment = "UPDATE"
        result, _ = self._generate("v2")
        details = result["memory_v2_details"]
        self.assertEqual(result["relevant_memories"], ["联系人喜欢音乐"])
        self.assertEqual(details["processed_actions"], [{
            "action": "UPDATE", "target": "联系人住在南京", "memory": "联系人住在苏州",
        }])
        fact_result = details["fact_results"][0]
        self.assertEqual(fact_result["alignment"]["target_index"], 0)
        self.assertEqual(fact_result["retrieval"]["candidates"][0]["memory"], "联系人住在南京")
        self.assertEqual(details["consistency_results"][0]["actions"], [])
        self.assertEqual(details["consistency_results"][0]["stage"], "V2 Consistency")
        for key in ("processed_actions", "action_results", "consistency_results", "consistency_errors"):
            self.assertNotIn(key, result)
        self.assertEqual(self._disk_memories()[self.sender], ["联系人住在苏州", "联系人喜欢音乐"])

    def test_v2_false_crud_result_is_not_replaced_with_a_fabricated_error(self):
        with patch.object(self.storage, "apply_memory_action", return_value=False):
            result, _ = self._generate("v2")
        action_result = result["memory_v2_details"]["action_results"][0]
        self.assertFalse(action_result["success"])
        self.assertNotIn("error", action_result)
        self.assertEqual(self._disk_memories(), self.seed)

    def test_v2_memory_and_working_context_remain_sender_scoped(self):
        first, _ = self._generate("v2")
        self.assertEqual(self._disk_memories()["sentinel"], self.seed["sentinel"])
        with redirect_stdout(io.StringIO()):
            second = self.main.process_user_message("sentinel", self.incoming, memory_version="v2")
        self.assertEqual(self._disk_memories(), {
            sender: memories + ["联系人喜欢摄影"] for sender, memories in self.seed.items()
        })
        self.assertEqual(first["sender_id"], self.sender)
        self.assertEqual(second["sender_id"], "sentinel")
        self.assertEqual(second["memory_v2_details"]["working_messages"], [
            {"role": "user", "content": "SENTINEL_CONTEXT"}
        ])
        self.assertNotEqual(
            first["memory_v2_details"]["working_messages"],
            second["memory_v2_details"]["working_messages"],
        )

    def _check_feedback(self, feedback, input_values, final_reply):
        for version in ("v1", "v2"):
            with self.subTest(version=version):
                self.storage.reset_memories(copy.deepcopy(self.seed))
                self._seed_context()
                self.feedback_file.write_text("[]", encoding="utf-8")
                before = copy.deepcopy(agent_context.get_working_memory(self.sender).get_messages())
                result, _ = self._generate(version)
                self.assertEqual(agent_context.get_working_memory(self.sender).get_messages(), before)
                memory_after_generation = self.memory_file.read_bytes()
                inference_calls = self.requests.post.call_count
                cli = self._load("demo_cli")
                with patch("builtins.input", side_effect=input_values):
                    record = cli._collect_feedback(self.sender, result)
                self.assertEqual(record["feedback"], feedback)
                self.assertEqual(record["final"], final_reply)
                expected = before + [{"role": "user", "content": self.incoming}]
                if final_reply:
                    expected.append({"role": "assistant", "content": final_reply})
                self.assertEqual(agent_context.get_working_memory(self.sender).get_messages(), expected)
                self.assertEqual(json.loads(self.feedback_file.read_text(encoding="utf-8")), [record])
                self.assertEqual(self.memory_file.read_bytes(), memory_after_generation)
                self.assertEqual(self.requests.post.call_count, inference_calls)

    def test_accept_feedback_remains_compatible_with_both_versions(self):
        self._check_feedback("accepted", ["accept"], "OK, photography it is.")

    def test_edit_feedback_remains_compatible_with_both_versions(self):
        self._check_feedback("edited", ["edit", "  Revised reply.  "], "Revised reply.")

    def test_reject_feedback_remains_compatible_with_both_versions(self):
        self._check_feedback("rejected", ["reject"], "")

    def test_existing_cli_still_calls_the_default_v1_interface(self):
        cli = self._load("demo_cli")
        with (
            patch("builtins.input", side_effect=[self.sender, self.incoming, "accept", "exit"]),
            patch.object(self.main, "process_user_message", wraps=self.main.process_user_message) as process,
            patch.object(self.v2, "run_memory_lifecycle_v2") as v2,
            redirect_stdout(io.StringIO()),
        ):
            status = cli.run_cli()
        self.assertEqual(status, 0)
        process.assert_called_once_with(self.sender, self.incoming)
        v2.assert_not_called()
        self.assertEqual(json.loads(self.feedback_file.read_text(encoding="utf-8"))[0]["feedback"], "accepted")

    def test_v1_crud_errors_keep_the_existing_action_result_behavior(self):
        with patch.object(self.main, "apply_memory_action", side_effect=OSError("write failed")):
            result, _ = self._generate("v1")
        self.assertEqual(result["action_results"], [{
            "action": {"action": "ADD", "target_index": None, "memory": "联系人喜欢摄影"},
            "success": False, "error": "write failed",
        }])
        self.assertEqual(result["timing_profile"]["outcome"], "success")
        self.assertEqual(self._disk_memories(), self.seed)


if __name__ == "__main__":
    unittest.main()
