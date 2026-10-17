import copy
import unittest
from unittest.mock import patch

import main
import memory_search
import ollama_client


class ContextualMemoryFlowTest(unittest.TestCase):
    def setUp(self):
        self.original_memories = copy.deepcopy(
            memory_search.memories_by_sender
        )
        main.reset_working_memories()

    def tearDown(self):
        memory_search.memories_by_sender = self.original_memories
        main.reset_working_memories()

    @staticmethod
    def _fake_ollama(prompt, stage="Unspecified"):
        expected_stages = {
            "上下文检索查询解析器": "Contextual Query Resolver",
            "长期记忆检索的相关性判断器": "Relevance Gate",
            "Response Planner": "Planner",
            "Draft Generator": "Draft",
            "长期记忆管理器": "Memory Manager",
            "Action Coverage Checker": "Action Coverage Checker",
            "长期记忆一致性检查器": "Consistency",
        }
        for marker, expected_stage in expected_stages.items():
            if marker in prompt and stage != expected_stage:
                raise AssertionError(
                    f"Unexpected stage for {marker}: {stage}"
                )

        if "上下文检索查询解析器" in prompt:
            return {
                "query": "联系人准备暂停学习摄影"
            }

        if "长期记忆检索的相关性判断器" in prompt:
            indices = []
            for line in prompt.splitlines():
                if ". " not in line or "摄影" not in line:
                    continue
                prefix = line.split(". ", 1)[0]
                if prefix.isdigit():
                    indices.append(int(prefix))
            return {"relevant_indices": indices}

        if "Response Planner" in prompt:
            return {
                "intent": "确认联系人要暂停摄影学习",
                "points": ["回应暂停摄影学习的决定"],
                "tone": "平静"
            }

        if "Draft Generator" in prompt:
            return {"draft": "好，那摄影先停一阵。"}

        if "长期记忆管理器" in prompt:
            return {
                "actions": [
                    {
                        "action": "DELETE",
                        "target_index": 0,
                        "memory": None
                    }
                ]
            }

        if "Action Coverage Checker" in prompt:
            return {"missing_actions": []}

        if "长期记忆一致性检查器" in prompt:
            return {"actions": []}

        raise AssertionError(f"Unexpected prompt: {prompt[:80]}")

    def test_working_memory_reference_can_delete_retrieved_memory(self):
        memory_search.memories_by_sender = {
            "target": [
                "联系人正在学习摄影",
                "联系人住在厦门"
            ],
            "sentinel": ["联系人喜欢哨兵音乐"]
        }

        working_memory = main.get_working_memory("target")
        working_memory.add_message(
            "user",
            "最近摄影课作业有点多"
        )
        working_memory.add_message(
            "assistant",
            "你最近确实花了不少时间在摄影上。"
        )

        with (
            patch.object(
                ollama_client,
                "call_ollama",
                side_effect=self._fake_ollama
            ),
            patch.object(memory_search, "save_memories")
        ):
            result = main.process_user_message(
                "target",
                "这个我准备先停一阵"
            )

        self.assertEqual(
            result["contextual_query"],
            "联系人准备暂停学习摄影"
        )
        self.assertNotIn(
            "联系人正在学习摄影",
            memory_search.get_sender_memories("target")
        )
        self.assertEqual(
            memory_search.get_sender_memories("sentinel"),
            ["联系人喜欢哨兵音乐"]
        )
        self.assertEqual(
            result["processed_actions"][0]["target"],
            "联系人正在学习摄影"
        )

    def test_memory_manager_rejects_free_text_update_target(self):
        with patch.object(
            ollama_client,
            "call_ollama",
            return_value={
                "actions": [
                    {
                        "action": "UPDATE",
                        "target": "联系人住在南京",
                        "memory": "联系人住在苏州"
                    }
                ]
            }
        ):
            with self.assertRaises(ValueError):
                ollama_client.decide_memory_actions(
                    "target",
                    "我搬去苏州了",
                    ["联系人住在南京"],
                    []
                )


if __name__ == "__main__":
    unittest.main()
