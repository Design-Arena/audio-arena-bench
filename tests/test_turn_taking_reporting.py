"""Unavailable timing evidence must not become a measured pass or failure."""

import copy
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd

from audio_arena.judging.llm_judge import prepare_turn_taking, write_outputs
from audio_arena.judging.turn_taking import TurnTakingAnalysis

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_experiment_review import build_review_payload, normalize_turn_taking_for_report
from compare_model_runs import build_run_record, create_markdown_summary

# Streamlit is optional; these tests exercise its pure dataframe preparation.
with patch.dict(sys.modules, {"streamlit": Mock()}):
    from review_streamlit_app import build_dimension_pass_rate_df


def records_and_judgments():
    records = [
        {"turn": turn, "user_text": "hello", "assistant_text": "hello",
         "latency_ms": 100, "ttfb_ms": 80}
        for turn in range(2)
    ]
    judgments = {
        turn: {"scores": {"turn_taking": bool(turn), "tool_use_correct": None,
                          "instruction_following": True, "kb_grounding": True,
                          "state_tracking": None, "ambiguity_handling": None},
               "reasoning": "test", "turn_taking_issues": ["stale issue"]}
        for turn in range(2)
    }
    return records, judgments


class TurnTakingReportingTests(unittest.TestCase):
    def test_cli_summaries_show_na_or_a_genuine_zero_for_both_judges(self):
        from click.testing import CliRunner
        from audio_arena.cli import cli

        for backend, module in [("claude", "llm_judge"), ("openai", "openai_judge")]:
            for supported in [False, True]:
                with self.subTest(backend=backend, supported=supported), TemporaryDirectory() as directory:
                    run_dir = Path(directory) / "conversation_bench" / "run"
                    run_dir.mkdir(parents=True)
                    records, judgments = records_and_judgments()
                    (run_dir / "transcript.jsonl").write_text("\n".join(map(json.dumps, records)))
                    result = {
                        "judgments": judgments, "summary": "test", "model_name": "test-model",
                        "turn_taking_supported": supported,
                        "turn_taking_analysis": {"per_turn": {str(turn): {"turn_taking": False} for turn in range(2)}} if supported else None,
                    }
                    with patch(f"audio_arena.judging.{module}.judge_with_{backend}", new=AsyncMock(return_value=result)):
                        output = CliRunner().invoke(cli, ["judge", str(run_dir), "--judge", backend])
                    self.assertEqual(output.exit_code, 0, output.output)
                    self.assertIn("Turn-taking: 0/2" if supported else "Turn-taking: N/A", output.output)

    def test_comparison_markdown_distinguishes_na_from_zero(self):
        args = SimpleNamespace(benchmark="test", runs_per_model=1, service=None,
                               rehydrate=True, disable_vad=True, parallel=1,
                               judge="openai", judge_model=None, skip_turn_taking=True)
        run_results = pd.DataFrame([{"model_name": "test", "run_index": 0, "run_dir": "test/run"}])
        metrics = pd.DataFrame([
            {"model_name": "test", "metric": "turn_taking", "mean_pass_rate": float("nan")},
            {"model_name": "test", "metric": "instruction_following", "mean_pass_rate": 0,
             "mean_pass_count": 0, "mean_applicable": 2,
             "mean_pass_rate_ci_lower": 0, "mean_pass_rate_ci_upper": 0.5},
        ])
        with TemporaryDirectory() as directory:
            output = Path(directory) / "summary.md"
            create_markdown_summary(args, run_results, metrics, pd.DataFrame(), output)
            report = output.read_text()
        self.assertIn("turn_taking: N/A (not evaluated)", report)
        self.assertIn("instruction_following: 0.00/2.00 (0.0%", report)
        self.assertNotIn("nan", report)

    def test_skipped_missing_and_failed_analysis_are_unavailable(self):
        with TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self.assertFalse(prepare_turn_taking(run_dir, True)[2])
            self.assertFalse(prepare_turn_taking(run_dir, False)[2])
            (run_dir / "conversation.wav").touch()
            for outcome in [TurnTakingAnalysis(directory, error="timing failed"),
                            TurnTakingAnalysis(directory), RuntimeError("analyzer crashed")]:
                with self.subTest(outcome=outcome):
                    kwargs = {"side_effect": outcome} if isinstance(outcome, Exception) else {"return_value": outcome}
                    with patch("audio_arena.judging.turn_taking.analyze_turn_taking", **kwargs):
                        data, _, supported, reason = prepare_turn_taking(run_dir, False)
                    self.assertFalse(data)
                    self.assertFalse(supported)
                    self.assertTrue(reason)

    def test_runtime_opt_out_is_respected_even_with_a_wav(self):
        with TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "conversation.wav").touch()
            (run_dir / "runtime.json").write_text(json.dumps({
                "turn_taking_supported": False, "turn_taking_skip_reason": "isolated audio",
            }))
            with patch("audio_arena.judging.turn_taking.analyze_turn_taking") as analyzer:
                _, _, supported, reason = prepare_turn_taking(run_dir, False)
                analyzer.assert_not_called()
            self.assertFalse(supported)
            self.assertEqual(reason, "isolated audio")

    def test_outputs_distinguish_unavailable_from_measured_zero_and_partial_coverage(self):
        cases = [
            (False, None, None, 0),
            (True, {"error": "timing failed", "per_turn": {}}, None, 0),
            (True, {"per_turn": {"0": {"turn_taking": False}, "1": {"turn_taking": False}}}, 0, 2),
            (True, {"per_turn": {"1": {"turn_taking": True}}}, 1, 1),
        ]
        for supported, analysis, passes, denominator in cases:
            with self.subTest(supported=supported, analysis=analysis), TemporaryDirectory() as directory:
                records, judgments = records_and_judgments()
                run_dir = Path(directory)
                write_outputs(run_dir, records, judgments, "", "test-model",
                              expected_turns=[{}, {}], turn_taking_supported=supported,
                              turn_taking_analysis=analysis)
                summary = json.loads((run_dir / "claude_summary.json").read_text())
                rows = [json.loads(line) for line in (run_dir / "claude_judged.jsonl").read_text().splitlines()]
                report = (run_dir / "claude_analysis.md").read_text()
                self.assertEqual(summary["passes"]["turn_taking"], passes)
                self.assertEqual(summary["category_totals"]["turn_taking"], denominator)
                self.assertEqual(summary["passes"]["instruction_following"], 2)
                self.assertEqual(summary["passes"]["kb_grounding"], 2)
                if not denominator:
                    self.assertIn("**Turn-Taking**: N/A", report)
                    self.assertFalse(summary["turn_taking_supported"])
                    self.assertEqual(summary["turn_taking_failures"], [])
                    self.assertTrue(all(row["scores"]["turn_taking"] is None for row in rows))
                else:
                    self.assertIn(f"**Turn-Taking**: {passes}/{denominator}", report)
                if denominator == 1:
                    self.assertIsNone(rows[0]["scores"]["turn_taking"])

    def test_legacy_reports_ignore_skipped_placeholder_scores_without_modifying_inputs(self):
        records, judgments = records_and_judgments()
        rows = [{**row, "scores": judgments[row["turn"]]["scores"]} for row in records]
        summary = {"passes": {"turn_taking": 0}, "turns_scored": 2,
                   "turn_taking_supported": False}
        original_rows = copy.deepcopy(rows)
        normalized, updated = normalize_turn_taking_for_report(rows, summary)
        self.assertEqual(rows, original_rows)
        self.assertEqual(summary["passes"]["turn_taking"], 0)
        self.assertIsNone(updated["passes"]["turn_taking"])
        self.assertTrue(all(row["scores"]["turn_taking"] is None for row in normalized))
        record = build_run_record(model_name="test", run_index=0, run_dir=Path("bench/run"),
                                  summary_payload=summary, runtime_payload={}, judged_rows=rows)
        self.assertIsNone(record["turn_taking_pass_rate"])
        self.assertEqual(record["turn_taking_applicable"], 0)
        self.assertEqual(record["pass_rows"], 2)
        summary["turn_taking_supported"] = True
        record = build_run_record(model_name="test", run_index=0, run_dir=Path("bench/run"),
                                  summary_payload=summary, runtime_payload={}, judged_rows=rows)
        self.assertEqual(record["turn_taking_pass_rate"], 0.0)
        self.assertEqual(record["turn_taking_applicable"], 2)

    def test_review_payload_preserves_availability_and_chart_keeps_na(self):
        with TemporaryDirectory() as directory:
            run_dir = Path(directory) / "conversation_bench" / "run"
            run_dir.mkdir(parents=True)
            records, judgments = records_and_judgments()
            rows = [{**row, "scores": judgments[row["turn"]]["scores"]} for row in records]
            (run_dir / "openai_judged.jsonl").write_text("\n".join(map(json.dumps, rows)))
            (run_dir / "transcript.jsonl").write_text("\n".join(map(json.dumps, records)))
            (run_dir / "runtime.json").write_text("{}")
            (run_dir / "openai_summary.json").write_text(json.dumps({
                "turn_taking_supported": False, "passes": {"turn_taking": 0},
            }))
            payload, _, _, _ = build_review_payload(run_dir)
        self.assertFalse(payload["summary"]["turn_taking_supported"])
        self.assertEqual(payload["summary"]["overall_pass_rows"], 2)
        chart = build_dimension_pass_rate_df(payload).set_index("dimension")
        self.assertTrue(pd.isna(chart.loc["turn_taking", "pass_rate"]))
        self.assertEqual(chart.loc["instruction_following", "pass_rate"], 1.0)
        chart = build_dimension_pass_rate_df({"dimension_summary": [
            {"dimension": "turn_taking", "pass_rate": 0.0},
        ]})
        self.assertEqual(chart.iloc[0]["pass_rate"], 0.0)


class JudgeTurnTakingEvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_both_backends_use_audio_evidence_instead_of_model_guesses(self):
        from audio_arena.judging.llm_judge import judge_with_claude
        from audio_arena.judging.openai_judge import judge_with_openai

        result_text = json.dumps({"final_judgments": [
            {"turn": 0, "turn_taking": True, "instruction_following": True, "kb_grounding": True},
            {"turn": 1, "turn_taking": False, "instruction_following": True, "kb_grounding": True},
        ]})
        async def claude_query(**kwargs):
            yield SimpleNamespace(content=result_text)
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=result_text))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=response))))
        with TemporaryDirectory() as directory:
            run_dir = Path(directory)
            records, _ = records_and_judgments()
            (run_dir / "transcript.jsonl").write_text("\n".join(map(json.dumps, records)))
            for backend, judge in [("llm_judge", judge_with_claude), ("openai_judge", judge_with_openai)]:
                for data, expected in [(None, [None, None]), ({0: {"turn_taking": False}}, [False, None])]:
                    with self.subTest(backend=backend, data=data), \
                         patch(f"audio_arena.judging.{backend}.prepare_turn_taking", return_value=(data, None, bool(data), None)), \
                         patch("openai.AsyncOpenAI", return_value=client), \
                         patch("audio_arena.judging.llm_judge.query", new=claude_query):
                        result = await judge(run_dir, expected_turns=[{}, {}])
                        self.assertEqual([result["judgments"][turn]["scores"]["turn_taking"] for turn in range(2)], expected)


if __name__ == "__main__":
    unittest.main()
