"""Aggregation regressions for the audio-agent-bench reference scoring rules."""

import copy
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from audio_arena.judging.llm_judge import JUDGE_SYSTEM_PROMPT, write_outputs
from benchmarks.grocery_bench.turns import turns as grocery_turns

review_path = Path(__file__).resolve().parents[1] / "scripts" / "build_experiment_review.py"
review_spec = importlib.util.spec_from_file_location("build_experiment_review", review_path)
review_module = importlib.util.module_from_spec(review_spec)
review_spec.loader.exec_module(review_module)


def judgment(tool_score=None, **scores):
    return {
        "scores": {
            "turn_taking": True,
            "tool_use_correct": tool_score,
            "instruction_following": True,
            "kb_grounding": True,
            "ambiguity_handling": None,
            "state_tracking": None,
            **scores,
        },
        "reasoning": "test judgment",
    }


def aggregate(goldens, judgments, *, judge="openai", calls=None):
    records = [
        {"turn": turn, "user_text": "user", "assistant_text": "assistant",
         "tool_calls": (calls or {}).get(turn, [])}
        for turn in sorted(judgments)
    ]
    with TemporaryDirectory() as directory:
        run_dir = Path(directory)
        write_outputs(run_dir, records, copy.deepcopy(judgments), "", "test-model",
                      expected_turns=goldens, judge_name=judge,
                      turn_taking_supported=False)
        summary = json.loads((run_dir / f"{judge}_summary.json").read_text())
        rows = [json.loads(line) for line in (run_dir / f"{judge}_judged.jsonl").read_text().splitlines()]
        analysis = (run_dir / f"{judge}_analysis.md").read_text()
    return summary, rows, analysis


class JudgeAggregationTests(unittest.TestCase):
    def test_no_action_labels_do_not_inflate_tool_score(self):
        goldens = [{"required_function_call": None}, {},
                   {"required_function_call": {"name": "lookup"}},
                   {"required_function_call": {"name": "lookup"}}]
        result, rows, _ = aggregate(goldens, {
            0: judgment(None), 1: judgment(True),
            2: judgment(True), 3: judgment(False),
        })
        self.assertEqual(result["passes"]["tool_use_correct"], 1)
        self.assertEqual(result["category_totals"]["tool_use_correct"], 2)
        self.assertIsNone(rows[1]["scores"]["tool_use_correct"])
        self.assertEqual(result["turns_scored"], 4)
        self.assertEqual(result["passes"]["instruction_following"], 4)
        self.assertEqual(result["passes"]["kb_grounding"], 4)

    def test_unexpected_calls_count_as_failures_in_main_denominator(self):
        goldens = [{}, {"required_function_call": {"name": "lookup"}}, {}]
        result, rows, _ = aggregate(
            goldens, {0: judgment(False), 1: judgment(True), 2: judgment(None)},
            calls={0: [{"name": "lookup"}, {"name": "lookup"}]},
        )
        self.assertEqual(result["passes"]["tool_use_correct"], 1)
        self.assertEqual(result["category_totals"]["tool_use_correct"], 2)
        self.assertFalse(rows[0]["scores"]["tool_use_correct"])
        self.assertIsNone(rows[2]["scores"]["tool_use_correct"])

    def test_sparse_ids_and_multi_call_turns_use_golden_turns_once(self):
        goldens = [{} for _ in range(75)]
        goldens[74] = {"required_function_call": [{"name": "a"}, {"name": "b"}]}
        result, rows, _ = aggregate(goldens, {74: judgment(True), 3: judgment(True)})
        self.assertEqual(result["category_totals"]["tool_use_correct"], 1)
        self.assertEqual(result["passes"]["tool_use_correct"], 1)
        self.assertEqual([row["turn"] for row in rows], [3, 74])
        self.assertIsNone(rows[0]["scores"]["tool_use_correct"])

    def test_extended_dimensions_keep_reference_non_null_denominators(self):
        goldens = [
            {"categories": ["long_range_memory", "ambiguous_entity"]},
            {"categories": ["basic_qa"]},
        ]
        result, _, _ = aggregate(goldens, {
            0: judgment(state_tracking=None, ambiguity_handling=False),
            1: judgment(state_tracking=True, ambiguity_handling=True),
        })
        # Category gating remains the judge's responsibility, as in the reference.
        self.assertEqual(result["category_totals"]["state_tracking"], 1)
        self.assertEqual(result["passes"]["state_tracking"], 1)
        self.assertEqual(result["category_totals"]["ambiguity_handling"], 2)
        self.assertEqual(result["passes"]["ambiguity_handling"], 1)

    def test_required_null_is_excluded_like_reference(self):
        result, rows, _ = aggregate(
            [{"required_function_call": {"name": "lookup"}}], {0: judgment(None)}
        )
        self.assertEqual(result["category_totals"]["tool_use_correct"], 0)
        self.assertIsNone(rows[0]["scores"]["tool_use_correct"])

    def test_non_required_true_is_normalized_even_if_call_is_recorded(self):
        result, rows, _ = aggregate(
            [{}], {0: judgment(True)}, calls={0: [{"name": "lookup"}]}
        )
        self.assertEqual(result["category_totals"]["tool_use_correct"], 0)
        self.assertIsNone(rows[0]["scores"]["tool_use_correct"])

    def test_missing_spec_keeps_reference_legacy_fallback(self):
        for goldens in (None, [], [{}]):
            with self.subTest(goldens=goldens):
                result, rows, _ = aggregate(goldens, {3: judgment(True), 4: judgment(False)})
                self.assertEqual(result["category_totals"]["tool_use_correct"], 2)
                self.assertEqual(result["passes"]["tool_use_correct"], 1)
                self.assertTrue(rows[0]["scores"]["tool_use_correct"])

    def test_grocery_denominator_includes_unexpected_call_failures(self):
        judgments = {turn: judgment(True) for turn in range(len(grocery_turns))}
        judgments[8] = judgment(False)
        judgments[10] = judgment(False)
        result, _, _ = aggregate(grocery_turns, judgments)
        self.assertEqual(result["passes"]["tool_use_correct"], 22)
        self.assertEqual(result["category_totals"]["tool_use_correct"], 24)
        self.assertEqual(result["turns_scored"], 30)

    def test_both_judge_outputs_and_review_use_normalized_scores(self):
        goldens = [{}, {"required_function_call": {"name": "lookup"}}, {}]
        for judge in ("claude", "openai"):
            with self.subTest(judge=judge):
                summary, rows, analysis = aggregate(
                    goldens, {0: judgment(False), 1: judgment(True), 2: judgment(True)},
                    judge=judge,
                )
                report = review_module.build_dimension_summary(pd.DataFrame([row["scores"] for row in rows]))
                tool = next(row for row in report if row["dimension"] == "tool_use_correct")
                self.assertEqual((tool["passed"], tool["applicable"], tool["failed"]), (1, 2, 1))
                self.assertEqual(tool["applicable"], summary["category_totals"]["tool_use_correct"])
                self.assertIn("**Tool Use Correct**: 1/2 (50.0%", analysis)

    def test_no_eligible_tool_scores_display_na(self):
        summary, rows, analysis = aggregate([{}], {0: judgment(True)})
        self.assertEqual(summary["category_totals"]["tool_use_correct"], 0)
        self.assertIn("**Tool Use Correct**: N/A", analysis)
        report = review_module.build_dimension_summary(pd.DataFrame([row["scores"] for row in rows]))
        tool = next(row for row in report if row["dimension"] == "tool_use_correct")
        self.assertIsNone(tool["pass_rate"])

    def test_prompt_uses_reference_tool_applicability_rules(self):
        self.assertIn("NULL if no function call was expected AND the assistant did not call any tool", JUDGE_SYSTEM_PROMPT)
        self.assertIn("FALSE if no function call was expected but the assistant still called a tool (counts as a tool-use failure in metrics)", JUDGE_SYSTEM_PROMPT)
        self.assertNotIn("TRUE if no function call was expected AND the assistant did not call any tool", JUDGE_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
