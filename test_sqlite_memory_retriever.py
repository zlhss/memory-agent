"""Snapshot/target contracts, real SQLite, real cosine, and one real embedding test."""

import ast
import importlib.util
import math
import os
import subprocess
import sys
import tempfile
import time
import types
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from memory_candidate import CandidateSnapshot, MemoryCandidate
from memory_candidate_targets import delete_from_snapshot, update_from_snapshot
from memory_repository import MemoryFact, MutationStatus
from sqlite_memory_repository import SQLiteMemoryRepository
from sqlite_memory_retriever import SQLiteMemoryRetriever


ROOT = Path(__file__).resolve().parent


class VectorEncoder:
    """Deterministic vectors at the model seam; cosine/ranking remain real."""
    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    def encode(self, sentences):
        self.calls.append(tuple(sentences))
        return np.asarray([self.vectors[text] for text in sentences], dtype=float)


class SQLiteMemoryRetrieverTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="sqlite-retriever-test-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.repository = SQLiteMemoryRepository(self.directory / "facts.sqlite3")
        self.addCleanup(self.repository.close)
        self.encoder = VectorEncoder({
            "query": [1, 0], "high": [1, 0], "medium": [0.6, 0.8],
            "zero": [0, 1], "negative": [-1, 0], "same": [1, 0],
            "new": [1, 0], "updated": [0, 1],
        })
        self.retriever = SQLiteMemoryRetriever(self.repository, self.encoder)

    def _legacy_oracles(self):
        # Load unchanged JSON storage with only a scoped model constructor
        # double; its data comes from our temp JSON, never the real file.
        import json
        data = {sender: [fact.content for fact in self.repository.list(sender)] for sender in self.repository.list_senders()}
        (self.directory / "memories.json").write_text(json.dumps(data), encoding="utf-8")
        embedding = types.ModuleType("sentence_transformers")
        embedding.SentenceTransformer = Mock(return_value=self.encoder)
        spec = importlib.util.spec_from_file_location("_retriever_legacy_oracle", ROOT / "memory_search.py")
        legacy = importlib.util.module_from_spec(spec)
        previous = Path.cwd()
        try:
            os.chdir(self.directory)
            with patch.dict(sys.modules, {"sentence_transformers": embedding}):
                spec.loader.exec_module(legacy)
        finally:
            os.chdir(previous)
        # Execute the actual V2 retrieval function and literal threshold from
        # its AST without importing the rest of the model/LLM orchestration.
        tree = ast.parse((ROOT / "memory_v2.py").read_text(encoding="utf-8"))
        nodes = [node for node in tree.body if (
            isinstance(node, ast.FunctionDef) and node.name == "retrieve_fact_candidates"
        ) or (
            isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "V2_MIN_FACT_RELEVANCE_SCORE" for target in node.targets)
        )]
        namespace = {"memory_search": legacy, "cosine_similarity": cosine_similarity, "time": time}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / "memory_v2.py"), "exec"), namespace)
        return legacy.retrieve, namespace["retrieve_fact_candidates"]

    def test_list_facts_is_backward_compatible_ordered_and_sender_scoped(self):
        first = self.repository.add("a", "same")
        second = self.repository.add("a", "same")
        self.repository.add("b", "same")
        self.assertEqual(self.repository.list_facts("a"), [first, second])
        self.assertEqual(self.repository.list_facts("a"), self.repository.list("a"))
        self.assertEqual(self.repository.list_facts("missing"), [])

    def test_standard_retrieval_filters_and_preserves_metadata(self):
        medium = self.repository.add("a", "medium")
        high = self.repository.add("a", "high")
        self.repository.add("a", "zero")
        self.repository.add("a", "negative")
        snapshot = self.retriever.retrieve("a", "query", top_k=5)
        self.assertEqual(snapshot.contents, ("high", "medium"))
        self.assertEqual([item.fact_id for item in snapshot.candidates], [high.id, medium.id])
        self.assertEqual([item.version for item in snapshot.candidates], [1, 1])
        self.assertTrue(all(item.sender_id == "a" for item in snapshot.candidates))
        self.assertAlmostEqual(snapshot.candidates[0].score, 1.0)
        self.assertAlmostEqual(snapshot.candidates[1].score, 0.6)

    def test_high_recall_includes_zero_and_negative_scores(self):
        for content in ("negative", "medium", "zero", "high"):
            self.repository.add("a", content)
        snapshot = self.retriever.retrieve("a", "query", top_k=4, high_recall=True)
        self.assertEqual(snapshot.contents, ("high", "medium", "zero", "negative"))
        self.assertEqual(snapshot.candidates[-1].score, -1.0)
        self.assertIsInstance(snapshot, CandidateSnapshot)
        self.assertTrue(all(isinstance(item, MemoryCandidate) for item in snapshot.candidates))

    def test_top_k_is_applied_to_ranked_candidates(self):
        for content in ("zero", "medium", "negative", "high"):
            self.repository.add("a", content)
        for high_recall in (False, True):
            with self.subTest(high_recall=high_recall):
                self.assertEqual(self.retriever.retrieve("a", "query", top_k=1, high_recall=high_recall).contents, ("high",))

    def test_equal_scores_preserve_insertion_order_and_duplicate_text_ids(self):
        first = self.repository.add("a", "same")
        second = self.repository.add("a", "same")
        third = self.repository.add("a", "high")
        for high_recall in (False, True):
            snapshot = self.retriever.retrieve("a", "query", top_k=3, high_recall=high_recall)
            self.assertEqual(snapshot.contents, ("same", "same", "high"))
            self.assertEqual([item.fact_id for item in snapshot.candidates], [first.id, second.id, third.id])

    def test_standard_matches_unchanged_json_oracle_including_threshold_boundaries(self):
        for name, score in (("below", 0.099), ("boundary", 0.10), ("above", 0.101), ("fact_boundary", 0.40)):
            self.encoder.vectors[name] = [score, math.sqrt(1 - score * score)]
        for content in ("below", "boundary", "above", "fact_boundary", "same", "same", "negative"):
            self.repository.add("a", content)
        legacy, _ = self._legacy_oracles()
        for top_k in (0, 1, 3, 20):
            with self.subTest(top_k=top_k):
                expected = legacy("a", "query", top_k)
                snapshot = self.retriever.retrieve("a", "query", top_k)
                self.assertEqual([(item.score, item.content) for item in snapshot.candidates], [(float(score), text) for score, text in expected])

    def test_fact_standard_and_high_recall_match_unchanged_v2_oracle(self):
        for name, score in (("below_fact", 0.399), ("at_fact", 0.40), ("above_fact", 0.401)):
            self.encoder.vectors[name] = [score, math.sqrt(1 - score * score)]
        for content in ("negative", "below_fact", "at_fact", "above_fact", "same", "same", "zero"):
            self.repository.add("a", content)
        _, legacy_fact = self._legacy_oracles()
        for kind, hint in (("persistent", "add"), ("persistent_change", "update"), ("persistent", "update"), ("persistent", "end")):
            for top_k in (0, 1, 4, 20):
                with self.subTest(kind=kind, hint=hint, top_k=top_k):
                    fact = {"kind": kind, "operation_hint": hint, "fact": "query"}
                    expected = legacy_fact("a", fact, top_k)["candidates"]
                    snapshot = self.retriever.retrieve_fact_candidates("a", fact, top_k)
                    self.assertEqual([(item.score, item.content) for item in snapshot.candidates], [(item["score"], item["memory"]) for item in expected])

    def test_unknown_and_empty_senders_return_empty_without_embedding(self):
        self.repository.ensure_sender("empty")
        for sender in ("empty", "missing"):
            for high_recall in (False, True):
                snapshot = self.retriever.retrieve(sender, "query", high_recall=high_recall)
                self.assertEqual(snapshot.candidates, ())
                self.assertEqual(snapshot.sender_id, sender)
        self.assertEqual(self.encoder.calls, [])

    def test_zero_top_k_returns_empty_without_embedding(self):
        self.repository.add("a", "same")
        self.assertEqual(self.retriever.retrieve("a", "query", 0).contents, ())
        self.assertEqual(self.encoder.calls, [])

    def test_sender_isolation_applies_before_encoding(self):
        first = self.repository.add("a", "medium")
        self.repository.add("b", "high")
        snapshot = self.retriever.retrieve("a", "query", high_recall=True)
        self.assertEqual([item.fact_id for item in snapshot.candidates], [first.id])
        self.assertEqual(self.encoder.calls, [("medium",), ("query",)])

    def test_foreign_sender_from_broken_reader_is_rejected_before_encoding(self):
        reader = types.SimpleNamespace(list_facts=lambda sender: [MemoryFact("id", "other", "private", 1)])
        with self.assertRaisesRegex(ValueError, "another sender"):
            SQLiteMemoryRetriever(reader, self.encoder).retrieve("a", "query")
        self.assertEqual(self.encoder.calls, [])

    def test_reader_needs_only_list_facts_and_is_called_once_per_retrieval(self):
        fact = self.repository.add("a", "high")
        reader = types.SimpleNamespace(list_facts=Mock(return_value=[fact]))
        retriever = SQLiteMemoryRetriever(reader, self.encoder)
        self.assertEqual(retriever.retrieve("a", "query").contents, ("high",))
        reader.list_facts.assert_called_once_with("a")
        self.assertEqual(len(self.encoder.calls), 2)

    def test_add_and_update_are_visible_without_a_cache(self):
        self.assertEqual(self.retriever.retrieve("a", "query").contents, ())
        first = self.repository.add("a", "high")
        old = self.retriever.retrieve("a", "query")
        current = self.repository.update("a", first.id, "updated", expected_version=1).fact
        self.repository.add("a", "new")
        new = self.retriever.retrieve("a", "query", top_k=5, high_recall=True)
        self.assertEqual(new.contents, ("new", "updated"))
        selected = next(item for item in new.candidates if item.fact_id == first.id)
        self.assertEqual((selected.content, selected.version), (current.content, 2))
        self.assertEqual((old.contents, old.candidates[0].version), (("high",), 1))
        self.assertEqual(len(self.encoder.calls), 4)

    def test_reordering_on_later_retrieval_cannot_change_original_index_mapping(self):
        first = self.repository.add("a", "high")
        second = self.repository.add("a", "zero")
        original = self.retriever.retrieve("a", "query", high_recall=True)
        later = self.retriever.retrieve("a", "zero", high_recall=True)
        self.assertEqual([item.fact_id for item in original.candidates], [first.id, second.id])
        self.assertEqual([item.fact_id for item in later.candidates], [second.id, first.id])
        # No re-read/re-encode may occur while resolving or executing the old index.
        with patch.object(self.repository, "list_facts", side_effect=AssertionError("No re-query")), patch.object(self.encoder, "encode", side_effect=AssertionError("No re-encode")):
            result = update_from_snapshot(self.repository, original, sender_id="a", target_index=0, content="new")
        self.assertEqual(result.status, MutationStatus.SUCCESS)
        self.assertEqual(result.fact.id, first.id)
        self.assertEqual(self.repository.list("a")[1], second)

    def test_duplicate_text_target_index_updates_second_id_not_first(self):
        first = self.repository.add("a", "same")
        second = self.repository.add("a", "same")
        snapshot = self.retriever.retrieve("a", "query")
        result = update_from_snapshot(self.repository, snapshot, sender_id="a", target_index=1, content="new")
        self.assertEqual(result.status, MutationStatus.SUCCESS)
        self.assertEqual(result.fact.id, second.id)
        self.assertEqual(self.repository.list("a"), [first, result.fact])

    def test_duplicate_text_target_index_deletes_second_id_not_first(self):
        first = self.repository.add("a", "same")
        second = self.repository.add("a", "same")
        snapshot = self.retriever.retrieve("a", "query")
        result = delete_from_snapshot(self.repository, snapshot, sender_id="a", target_index=1)
        self.assertEqual(result.status, MutationStatus.SUCCESS)
        self.assertEqual(result.fact.id, second.id)
        self.assertEqual(self.repository.list("a"), [first])

    def test_invalid_indices_are_rejected_without_writes(self):
        fact = self.repository.add("a", "high")
        snapshot = self.retriever.retrieve("a", "query")
        for index in (None, True, False, -1, 1, 999, 0.0, "0", np.int64(0)):
            with self.subTest(index=index):
                for operation in (
                    lambda: update_from_snapshot(self.repository, snapshot, sender_id="a", target_index=index, content="new"),
                    lambda: delete_from_snapshot(self.repository, snapshot, sender_id="a", target_index=index),
                ):
                    with self.assertRaises(ValueError):
                        operation()
        self.assertEqual(self.repository.list("a"), [fact])
        with self.assertRaises(ValueError):
            CandidateSnapshot("a", ()).resolve_target("a", 0)

    def test_wrong_sender_cannot_use_original_snapshot(self):
        first = self.repository.add("a", "high")
        other = self.repository.add("b", "high")
        snapshot = self.retriever.retrieve("a", "query")
        with self.assertRaises(ValueError):
            update_from_snapshot(self.repository, snapshot, sender_id="b", target_index=0, content="new")
        with self.assertRaises(ValueError):
            delete_from_snapshot(self.repository, snapshot, sender_id="b", target_index=0)
        self.assertEqual(self.repository.list("a"), [first])
        self.assertEqual(self.repository.list("b"), [other])

    def test_stale_update_and_delete_report_version_conflict_without_retry(self):
        fact = self.repository.add("a", "high")
        original = self.retriever.retrieve("a", "query")
        with SQLiteMemoryRepository(self.directory / "facts.sqlite3") as second:
            current = second.update("a", fact.id, "new", expected_version=1).fact
        for result in (
            update_from_snapshot(self.repository, original, sender_id="a", target_index=0, content="stale"),
            delete_from_snapshot(self.repository, original, sender_id="a", target_index=0),
        ):
            self.assertEqual(result.status, MutationStatus.VERSION_CONFLICT)
            self.assertIsNone(result.fact)
        self.assertEqual(self.repository.list("a"), [current])

    def test_deleted_target_reports_not_found_instead_of_retargeting_same_content(self):
        first = self.repository.add("a", "same")
        second = self.repository.add("a", "same")
        original = self.retriever.retrieve("a", "query")
        self.repository.delete("a", first.id, expected_version=1)
        result = update_from_snapshot(self.repository, original, sender_id="a", target_index=0, content="wrong")
        self.assertEqual(result.status, MutationStatus.NOT_FOUND)
        self.assertEqual(self.repository.list("a"), [second])

    def test_update_during_embedding_keeps_old_snapshot_version_and_reports_conflict(self):
        fact = self.repository.add("a", "high")
        original_encode = self.encoder.encode
        updated = []

        def concurrent_update(sentences):
            if not updated:
                updated.append(self.repository.update("a", fact.id, "new", expected_version=1).fact)
            return original_encode(sentences)

        with patch.object(self.encoder, "encode", side_effect=concurrent_update):
            snapshot = self.retriever.retrieve("a", "query")
        self.assertEqual((snapshot.candidates[0].content, snapshot.candidates[0].version), ("high", 1))
        result = update_from_snapshot(self.repository, snapshot, sender_id="a", target_index=0, content="stale")
        self.assertEqual(result.status, MutationStatus.VERSION_CONFLICT)
        self.assertEqual(self.repository.list("a"), updated)

    def test_llm_database_id_and_version_cannot_be_supplied_to_mapping_helpers(self):
        first = self.repository.add("a", "same")
        second = self.repository.add("a", "same")
        snapshot = self.retriever.retrieve("a", "query")
        fabricated = {"target_index": 1, "fact_id": first.id, "expected_version": 999}
        with self.assertRaises(TypeError):
            update_from_snapshot(self.repository, snapshot, sender_id="a", content="new", **fabricated)
        result = update_from_snapshot(self.repository, snapshot, sender_id="a", target_index=fabricated["target_index"], content="new")
        self.assertEqual(result.fact.id, second.id)
        self.assertEqual(result.fact.version, 2)
        self.assertEqual(self.repository.list("a")[0], first)

    def test_invalid_retrieval_inputs_and_bad_embedding_shape_fail_explicitly(self):
        self.repository.add("a", "high")
        for sender, query, top_k, high_recall in (("", "query", 1, False), ("a", None, 1, False), ("a", "", 1, False), ("a", "query", True, False), ("a", "query", -1, False), ("a", "query", 1, "yes")):
            with self.subTest(sender=sender, query=query, top_k=top_k):
                with self.assertRaises(ValueError):
                    self.retriever.retrieve(sender, query, top_k, high_recall=high_recall)
        with self.assertRaises(ValueError):
            self.retriever.retrieve_fact_candidates("a", [])
        wrong = types.SimpleNamespace(encode=lambda sentences: np.array([[1.0, 0.0], [0.0, 1.0]]))
        with self.assertRaisesRegex(ValueError, "do not match"):
            SQLiteMemoryRetriever(self.repository, wrong).retrieve("a", "query")

    def test_storage_and_embedding_errors_propagate_without_empty_success(self):
        self.repository.add("a", "high")
        with patch.object(self.encoder, "encode", side_effect=RuntimeError("embedding failed")):
            with self.assertRaisesRegex(RuntimeError, "embedding failed"):
                self.retriever.retrieve("a", "query")
        self.repository.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.retriever.retrieve("a", "query")

    def test_real_existing_embedding_model_retrieves_added_and_updated_sqlite_facts(self):
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}):
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", local_files_only=True)
        retriever = SQLiteMemoryRetriever(self.repository, model)
        first = self.repository.add("a", "联系人住在南京")
        self.repository.add("a", "联系人喜欢摄影")
        self.repository.add("b", "联系人住在厦门")
        original = retriever.retrieve("a", "联系人住在南京", top_k=1)
        self.assertEqual(original.candidates[0].fact_id, first.id)
        self.assertGreater(original.candidates[0].score, 0.99)
        self.repository.update("a", first.id, "联系人住在杭州", expected_version=1)
        latest = retriever.retrieve("a", "联系人住在杭州", top_k=1)
        self.assertEqual((latest.candidates[0].fact_id, latest.candidates[0].version, latest.contents), (first.id, 2, ("联系人住在杭州",)))
        self.assertEqual(original.contents, ("联系人住在南京",))
        self.assertEqual(update_from_snapshot(self.repository, original, sender_id="a", target_index=0, content="stale").status, MutationStatus.VERSION_CONFLICT)


class CandidateSnapshotTest(unittest.TestCase):
    def test_candidate_and_snapshot_are_immutable_and_defensively_copy_input_order(self):
        first = MemoryCandidate("id1", "a", "same", 1, 0.9)
        second = MemoryCandidate("id2", "a", "same", 2, 0.8)
        mutable = [first, second]
        snapshot = CandidateSnapshot("a", mutable)
        prompt_list = list(snapshot.contents)
        mutable.reverse()
        mutable.clear()
        prompt_list.clear()
        self.assertEqual(snapshot.contents, ("same", "same"))
        self.assertEqual(snapshot.resolve_target("a", 1), second)
        for operation in (lambda: setattr(first, "version", 999), lambda: setattr(snapshot, "candidates", ())):
            with self.assertRaises(FrozenInstanceError):
                operation()

    def test_mixed_senders_and_duplicate_ids_are_rejected_but_duplicate_content_is_valid(self):
        first = MemoryCandidate("id1", "a", "same", 1, 1.0)
        with self.assertRaises(ValueError):
            CandidateSnapshot("b", (first,))
        with self.assertRaises(ValueError):
            CandidateSnapshot("a", (first, first))
        self.assertEqual(CandidateSnapshot("a", (first, MemoryCandidate("id2", "a", "same", 1, 1.0))).contents, ("same", "same"))

    def test_malformed_identity_version_and_score_are_rejected(self):
        invalid = [("", "a", "text", 1, 1.0), ("id", "", "text", 1, 1.0), ("id", "a", "", 1, 1.0),
                   ("id", "a", "text", True, 1.0), ("id", "a", "text", 0, 1.0),
                   ("id", "a", "text", 1, float("nan")), ("id", "a", "text", 1, float("inf")), ("id", "a", "text", 1, "0.5")]
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    MemoryCandidate(*values)

    def test_import_does_not_load_agent_json_or_construct_a_model(self):
        code = f"import sys; sys.path.insert(0,{str(ROOT)!r}); import memory_candidate, memory_candidate_targets, sqlite_memory_retriever; assert all(n not in sys.modules for n in ('main','memory_v2','memory_search','ollama_client','sentence_transformers'))"
        completed = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
