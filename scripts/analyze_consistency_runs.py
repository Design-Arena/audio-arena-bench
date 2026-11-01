import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from build_experiment_review import normalize_turn_taking_for_report

METRIC_COLUMNS = [
    "turn_taking",
    "tool_use_correct",
    "instruction_following",
    "kb_grounding",
    "ambiguity_handling",
    "state_tracking",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze judged benchmark comparison runs for pass/fail consistency, "
            "metric flips, and unstable turns."
        )
    )
    parser.add_argument(
        "comparison_dirs",
        nargs="+",
        help="One or more comparison output directories from scripts/compare_model_runs.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write combined analysis artifacts. Defaults to the single comparison dir, or results/consistency_<timestamp>/ for multiple inputs.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalize_metric_value(value) -> str:
    if value is True:
        return "pass"
    if value is False:
        return "fail"
    return "na"


def normalize_overall_status(scores: dict) -> str:
    metric_values = [value for value in scores.values() if value is not None]
    if not metric_values:
        return "incomplete"
    if any(value is False for value in metric_values):
        return "fail"
    return "pass"


def load_comparison_details(comparison_dir: Path) -> tuple[pd.DataFrame, dict]:
    run_results_path = comparison_dir / "run_results.csv"
    metadata_path = comparison_dir / "metadata.json"
    if not run_results_path.exists():
        raise FileNotFoundError(f"Missing run_results.csv in {comparison_dir}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json in {comparison_dir}")

    run_results_df = pd.read_csv(run_results_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    detail_rows: list[dict] = []

    for run_record in run_results_df.to_dict(orient="records"):
        run_dir = Path(run_record["run_dir"])
        judged_path = run_dir / "openai_judged.jsonl"
        if not judged_path.exists():
            raise FileNotFoundError(f"Missing judged rows for {run_dir}")
        judged_rows = load_jsonl(judged_path)
        summary_path = run_dir / "openai_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            judged_rows, _ = normalize_turn_taking_for_report(judged_rows, summary)
        for row in judged_rows:
            scores = row.get("scores", {})
            failing_metrics = [
                metric_name
                for metric_name, metric_value in scores.items()
                if metric_value is False
            ]
            detail_row = {
                "comparison_dir": str(comparison_dir),
                "benchmark_name": run_record["benchmark_name"],
                "model_name": run_record["model_name"],
                "run_index": int(run_record["run_index"]),
                "run_id": run_record["run_id"],
                "run_dir": str(run_dir),
                "turn": int(row["turn"]),
                "user_text": row.get("user_text"),
                "assistant_text": row.get("assistant_text"),
                "tool_calls_json": json.dumps(
                    row.get("tool_calls", []),
                    ensure_ascii=True,
                    sort_keys=True,
                ),
                "judge_reasoning": row.get("judge_reasoning"),
                "latency_ms": row.get("latency_ms"),
                "ttfb_ms": row.get("ttfb_ms"),
                "overall_status": normalize_overall_status(scores),
                "failed_metric_count": len(failing_metrics),
                "failing_metrics": ",".join(failing_metrics),
            }
            for metric_name in METRIC_COLUMNS:
                detail_row[metric_name] = normalize_metric_value(scores.get(metric_name))
            detail_rows.append(detail_row)

    details_df = pd.DataFrame(detail_rows)
    return details_df, metadata


def run_sanity_checks(details_df: pd.DataFrame) -> pd.DataFrame:
    duplicate_count = int(details_df.duplicated(["run_dir", "turn"]).sum())
    check_rows = [
        {"check": "rows", "value": int(len(details_df))},
        {
            "check": "unique_runs",
            "value": int(details_df["run_dir"].nunique()),
        },
        {
            "check": "unique_turns",
            "value": int(details_df[["benchmark_name", "turn"]].drop_duplicates().shape[0]),
        },
        {"check": "duplicate_run_turn_rows", "value": duplicate_count},
    ]

    for column_name in ["user_text", "assistant_text", "judge_reasoning", "latency_ms", "ttfb_ms"]:
        check_rows.append(
            {
                "check": f"missing_{column_name}",
                "value": int(details_df[column_name].isna().sum()),
            }
        )

    if duplicate_count:
        raise ValueError("Detected duplicate (run_dir, turn) rows in judged output.")

    return pd.DataFrame(check_rows)


def summarize_turns(details_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict] = []
    grouped = details_df.groupby(["benchmark_name", "turn"], sort=True)

    for (benchmark_name, turn), turn_df in grouped:
        ordered_df = turn_df.sort_values("run_index")
        overall_pattern = "|".join(
            f"{int(row.run_index)}:{row.overall_status}"
            for row in ordered_df.itertuples()
        )
        overall_statuses = set(ordered_df["overall_status"])
        metric_flip_names = []
        for metric_name in METRIC_COLUMNS:
            metric_values = [
                value
                for value in ordered_df[metric_name].tolist()
                if value != "na"
            ]
            if len(set(metric_values)) > 1:
                metric_flip_names.append(metric_name)

        summary_rows.append(
            {
                "benchmark_name": benchmark_name,
                "turn": turn,
                "runs": int(len(ordered_df)),
                "pass_runs": int((ordered_df["overall_status"] == "pass").sum()),
                "fail_runs": int((ordered_df["overall_status"] == "fail").sum()),
                "incomplete_runs": int((ordered_df["overall_status"] == "incomplete").sum()),
                "overall_status_pattern": overall_pattern,
                "overall_status_unique_count": int(len(overall_statuses)),
                "metric_flip_count": int(len(metric_flip_names)),
                "metric_flips": ",".join(metric_flip_names),
                "unstable_overall": len(overall_statuses) > 1,
                "unstable_metric": bool(metric_flip_names),
                "user_text": ordered_df["user_text"].iloc[0],
            }
        )

    return pd.DataFrame(summary_rows)


def summarize_metric_flips(details_df: pd.DataFrame) -> pd.DataFrame:
    flip_rows: list[dict] = []
    grouped = details_df.groupby(["benchmark_name", "turn"], sort=True)

    for (benchmark_name, turn), turn_df in grouped:
        ordered_df = turn_df.sort_values("run_index")
        for metric_name in METRIC_COLUMNS:
            values = ordered_df[metric_name].tolist()
            non_na_values = [value for value in values if value != "na"]
            if len(set(non_na_values)) <= 1:
                continue
            pattern = "|".join(
                f"{int(row.run_index)}:{getattr(row, metric_name)}"
                for row in ordered_df.itertuples()
            )
            counts = Counter(values)
            flip_rows.append(
                {
                    "benchmark_name": benchmark_name,
                    "turn": turn,
                    "metric": metric_name,
                    "pattern": pattern,
                    "pass_runs": int(counts["pass"]),
                    "fail_runs": int(counts["fail"]),
                    "na_runs": int(counts["na"]),
                    "user_text": ordered_df["user_text"].iloc[0],
                }
            )

    return pd.DataFrame(flip_rows)


def summarize_benchmarks(turn_summary_df: pd.DataFrame) -> pd.DataFrame:
    benchmark_rows: list[dict] = []
    for benchmark_name, benchmark_df in turn_summary_df.groupby("benchmark_name", sort=True):
        benchmark_rows.append(
            {
                "benchmark_name": benchmark_name,
                "turns": int(len(benchmark_df)),
                "unstable_overall_turns": int(benchmark_df["unstable_overall"].sum()),
                "unstable_metric_turns": int(benchmark_df["unstable_metric"].sum()),
                "max_pass_runs": int(benchmark_df["pass_runs"].max()),
                "min_pass_runs": int(benchmark_df["pass_runs"].min()),
            }
        )
    return pd.DataFrame(benchmark_rows)


def build_flip_details(details_df: pd.DataFrame, turn_summary_df: pd.DataFrame) -> pd.DataFrame:
    unstable_turns_df = turn_summary_df[
        turn_summary_df["unstable_overall"] | turn_summary_df["unstable_metric"]
    ][["benchmark_name", "turn"]]
    return (
        details_df.merge(unstable_turns_df, on=["benchmark_name", "turn"], how="inner")
        .sort_values(["benchmark_name", "turn", "run_index"])
        .reset_index(drop=True)
    )


def write_summary_markdown(
    *,
    metadata_by_dir: dict[str, dict],
    benchmark_summary_df: pd.DataFrame,
    turn_summary_df: pd.DataFrame,
    metric_flip_df: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        "# Consistency Summary",
        "",
        f"- generated_at: {datetime.now(UTC).isoformat(timespec='seconds').replace('+00:00', 'Z')}",
        f"- comparison_dirs: {len(metadata_by_dir)}",
        "",
        "## Benchmarks",
    ]

    for row in benchmark_summary_df.itertuples():
        lines.append(
            f"- {row.benchmark_name}: unstable_overall_turns={row.unstable_overall_turns}/{row.turns}, "
            f"unstable_metric_turns={row.unstable_metric_turns}/{row.turns}, "
            f"pass_run_range={row.min_pass_runs}-{row.max_pass_runs}"
        )

    lines.extend(["", "## Inputs"])
    for comparison_dir, metadata in metadata_by_dir.items():
        lines.append(
            f"- {comparison_dir}: benchmark={metadata.get('benchmark')}, "
            f"models={metadata.get('models')}, runs_per_model={metadata.get('runs_per_model')}"
        )

    lines.extend(["", "## Most Unstable Turns"])
    top_turns_df = turn_summary_df.sort_values(
        ["metric_flip_count", "overall_status_unique_count", "benchmark_name", "turn"],
        ascending=[False, False, True, True],
    ).head(20)
    for row in top_turns_df.itertuples():
        lines.append(
            f"- {row.benchmark_name} turn {row.turn}: overall={row.overall_status_pattern}; "
            f"metric_flips={row.metric_flips or 'none'}"
        )

    lines.extend(["", "## Metric Flips"])
    if metric_flip_df.empty:
        lines.append("- No metric flips were detected.")
    else:
        for row in metric_flip_df.sort_values(["benchmark_name", "turn", "metric"]).itertuples():
            lines.append(
                f"- {row.benchmark_name} turn {row.turn} {row.metric}: {row.pattern}"
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    comparison_dirs = [Path(path).expanduser().resolve() for path in args.comparison_dirs]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    elif len(comparison_dirs) == 1:
        output_dir = comparison_dirs[0]
    else:
        output_dir = (Path.cwd() / "results" / f"{timestamp}_consistency_analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    details_frames: list[pd.DataFrame] = []
    metadata_by_dir: dict[str, dict] = {}

    for comparison_dir in comparison_dirs:
        details_df, metadata = load_comparison_details(comparison_dir)
        details_frames.append(details_df)
        metadata_by_dir[str(comparison_dir)] = metadata

    combined_details_df = pd.concat(details_frames, ignore_index=True)
    sanity_df = run_sanity_checks(combined_details_df)
    turn_summary_df = summarize_turns(combined_details_df)
    metric_flip_df = summarize_metric_flips(combined_details_df)
    benchmark_summary_df = summarize_benchmarks(turn_summary_df)
    flip_details_df = build_flip_details(combined_details_df, turn_summary_df)

    combined_details_df.to_csv(output_dir / "consistency_run_details.csv", index=False, encoding="utf-8")
    turn_summary_df.to_csv(output_dir / "consistency_turn_summary.csv", index=False, encoding="utf-8")
    metric_flip_df.to_csv(output_dir / "consistency_metric_flips.csv", index=False, encoding="utf-8")
    flip_details_df.to_csv(output_dir / "consistency_flip_details.csv", index=False, encoding="utf-8")
    benchmark_summary_df.to_csv(output_dir / "consistency_benchmark_summary.csv", index=False, encoding="utf-8")
    sanity_df.to_csv(output_dir / "consistency_sanity_checks.csv", index=False, encoding="utf-8")

    write_summary_markdown(
        metadata_by_dir=metadata_by_dir,
        benchmark_summary_df=benchmark_summary_df,
        turn_summary_df=turn_summary_df,
        metric_flip_df=metric_flip_df,
        output_path=output_dir / "consistency_summary.md",
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "detail_rows_csv": str(output_dir / "consistency_run_details.csv"),
                "turn_summary_csv": str(output_dir / "consistency_turn_summary.csv"),
                "metric_flips_csv": str(output_dir / "consistency_metric_flips.csv"),
                "flip_details_csv": str(output_dir / "consistency_flip_details.csv"),
                "benchmark_summary_csv": str(output_dir / "consistency_benchmark_summary.csv"),
                "sanity_checks_csv": str(output_dir / "consistency_sanity_checks.csv"),
                "summary_md": str(output_dir / "consistency_summary.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
