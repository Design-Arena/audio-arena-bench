import copy
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from audio_arena.judging.aggregation import aggregate_scores, score_applicability
from audio_arena.judging.llm_judge import write_outputs
from benchmarks.grocery_bench.turns import turns as grocery_turns

review_path = Path(__file__).resolve().parents[1] / "scripts" / "build_experiment_review.py"
review_spec = importlib.util.spec_from_file_location("build_experiment_review", review_path)
review_module = importlib.util.module_from_spec(review_spec)
review_spec.loader.exec_module(review_module)
SCORE_COLUMNS = review_module.SCORE_COLUMNS
build_dimension_summary = review_module.build_dimension_summary


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


def aggregate(goldens, judgments, records=(), turn_taking_supported=False):
    return aggregate_scores(
        records, judgments, goldens, turn_taking_supported=turn_taking_supported
    )[0]


class JudgeAggregationTests(unittest.TestCase):
    def test_no_action_labels_do_not_inflate_tool_score(self):
        goldens = [{"required_function_call": None}, {},
                   {"required_function_call": {"name": "lookup"}},
                   {"required_function_call": {"name": "lookup"}}]
        result = aggregate(goldens, {
            0: judgment(None), 1: judgment(True),
            2: judgment(True), 3: judgment(False),
        })
        self.assertEqual(result["passes"]["tool_use_correct"], 1)
        self.assertEqual(result["category_totals"]["tool_use_correct"], 2)
        self.assertEqual(result["category_totals"]["instruction_following"], 4)
        self.assertEqual(result["category_totals"]["kb_grounding"], 4)

    def test_sparse_ids_and_multi_call_turns_use_golden_turns_once(self):
        goldens = [{} for _ in range(75)]
        goldens[74] = {"required_function_call": [{"name": "a"}, {"name": "b"}]}
        result = aggregate(goldens, {74: judgment(True), 3: judgment(None)})
        self.assertEqual(result["category_totals"]["tool_use_correct"], 1)
        self.assertEqual(result["passes"]["tool_use_correct"], 1)
        self.assertEqual(result["category_totals"]["instruction_following"], 2)

    def test_categories_define_scope_even_with_inconsistent_judge_nulls(self):
        goldens = [
            {"categories": ["long_range_memory", "error_recovery", "ambiguous_entity"]},
            {"categories": ["basic_qa"]},
            {"category": "cross_turn_reasoning"},
        ]
        result = aggregate(goldens, {
            0: judgment(state_tracking=None, ambiguity_handling=False),
            1: judgment(state_tracking=True, ambiguity_handling=True),
            2: judgment(state_tracking=True),
        })
        self.assertEqual(result["category_totals"]["state_tracking"], 2)
        self.assertEqual(result["passes"]["state_tracking"], 1)
        self.assertEqual(result["missing_scores"]["state_tracking"], [0])
        self.assertEqual(result["category_totals"]["ambiguity_handling"], 1)
        self.assertEqual(result["passes"]["ambiguity_handling"], 0)

    def test_all_state_tags_and_only_ambiguity_tag_are_eligible(self):
        for tag in ("long_range_memory", "cross_turn_reasoning", "cancellation_flow",
                    "implicit_correction", "error_recovery"):
            with self.subTest(tag=tag):
                scope = score_applicability({"categories": [tag]}, turn_taking_supported=False)
                self.assertTrue(scope["state_tracking"])
                self.assertFalse(scope["ambiguity_handling"])
        scope = score_applicability({"categories": ["numerical_reasoning"]}, turn_taking_supported=False)
        self.assertFalse(scope["ambiguity_handling"])

    def test_unexpected_and_realigned_calls_remain_visible_outside_denominator(self):
        judgments = {0: judgment(False), 1: judgment(True)}
        original = copy.deepcopy(judgments)
        records = [{"turn": turn, "tool_calls": [{"name": "lookup"}]} for turn in judgments]
        result = aggregate([{}, {}], judgments, records)
        self.assertEqual(result["category_totals"]["tool_use_correct"], 0)
        self.assertEqual(result["passes"]["tool_use_correct"], 0)
        self.assertEqual(result["tool_use_diagnostics"], {
            "no_required_call_turns": 2,
            "turns_with_calls_without_required_call": [0, 1],
            "failed_turns_without_required_call": [0],
        })
        self.assertEqual(judgments, original)

    def test_required_null_is_missing_without_shrinking_denominator(self):
        result = aggregate([{"required_function_call": {"name": "lookup"}}], {0: judgment(None)})
        self.assertEqual(result["category_totals"]["tool_use_correct"], 1)
        self.assertEqual(result["missing_scores"]["tool_use_correct"], [0])

    def test_turn_taking_is_applicable_only_when_evaluated(self):
        for supported in (True, False):
            result = aggregate([{}], {0: judgment()}, turn_taking_supported=supported)
            self.assertEqual(result["category_totals"]["turn_taking"], int(supported))
            self.assertEqual(result["passes"]["turn_taking"], int(supported))

    def test_missing_or_out_of_range_goldens_fail_explicitly(self):
        for goldens, turn in ((None, 0), ([], 0), ([{}], -1)):
            with self.subTest(goldens=goldens, turn=turn), self.assertRaises(ValueError):
                aggregate(goldens, {turn: judgment()})

    def test_grocery_has_22_required_call_turns(self):
        judgments = {turn: judgment(True) for turn in range(len(grocery_turns))}
        result = aggregate(grocery_turns, judgments)
        self.assertEqual(result["passes"]["tool_use_correct"], 22)
        self.assertEqual(result["category_totals"]["tool_use_correct"], 22)
        self.assertEqual(result["category_totals"]["instruction_following"], 30)

    def test_both_judge_outputs_and_report_use_saved_scope(self):
        judgments = {0: judgment(True), 1: judgment(False), 2: judgment(True)}
        records = [{"turn": turn, "user_text": "user", "assistant_text": "assistant"}
                   for turn in judgments]
        goldens = [{}, {"required_function_call": {"name": "lookup"}}, {}]
        for judge in ("claude", "openai"):
            with self.subTest(judge=judge), TemporaryDirectory() as directory:
                run_dir = Path(directory)
                write_outputs(run_dir, records, judgments, "", "test-model",
                              expected_turns=goldens, judge_name=judge,
                              turn_taking_supported=False)
                summary = json.loads((run_dir / f"{judge}_summary.json").read_text())
                rows = [json.loads(line) for line in (run_dir / f"{judge}_judged.jsonl").read_text().splitlines()]
                self.assertTrue(rows[0]["scores"]["tool_use_correct"])
                self.assertFalse(rows[0]["score_applicability"]["tool_use_correct"])
                self.assertEqual(summary["category_totals"]["tool_use_correct"], 1)
                report = build_dimension_summary(pd.DataFrame([row["scores"] for row in rows]), summary)
                tool = next(row for row in report if row["dimension"] == "tool_use_correct")
                self.assertEqual((tool["passed"], tool["applicable"], tool["failed"]), (0, 1, 1))
                analysis = (run_dir / f"{judge}_analysis.md").read_text()
                self.assertIn("**Tool Use Correct**: 0/1 (0.0%)", analysis)
                self.assertIn("**Turn-Taking**: N/A (not evaluated)", analysis)

    def test_no_eligible_turns_are_na_in_analysis_and_review(self):
        with TemporaryDirectory() as directory:
            run_dir = Path(directory)
            write_outputs(run_dir, [], {}, "", "test", expected_turns=[])
            summary = json.loads((run_dir / "claude_summary.json").read_text())
            self.assertEqual(summary["category_totals"]["tool_use_correct"], 0)
            self.assertIn("**Tool Use Correct**: N/A", (run_dir / "claude_analysis.md").read_text())
            report = build_dimension_summary(pd.DataFrame(columns=SCORE_COLUMNS), summary)
            self.assertTrue(all(row["pass_rate"] is None for row in report))


if __name__ == "__main__":
    unittest.main()
