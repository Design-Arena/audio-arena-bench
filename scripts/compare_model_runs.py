import argparse
import json
import math
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_experiment_review import SCORE_COLUMNS, normalize_turn_taking_for_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a multi-model benchmark comparison, judge each run, and write "
            "summary CSVs plus comparison plots."
        )
    )
    parser.add_argument(
        "--benchmark",
        default="conversation_bench",
        help="Benchmark name to run.",
    )
    parser.add_argument(
        "--models",
        required=True,
        help="Comma-separated list of model names to compare.",
    )
    parser.add_argument(
        "--runs-per-model",
        type=int,
        default=1,
        help="Number of fresh runs to execute per model.",
    )
    parser.add_argument(
        "--service",
        default=None,
        help="Optional shared service override (for example: openai-realtime).",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=16,
        help="Concurrency for rehydrated runs.",
    )
    parser.add_argument(
        "--judge",
        default="openai",
        help="Judge backend to use after each run.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Optional judge model override.",
    )
    parser.add_argument(
        "--rehydrate",
        action="store_true",
        default=True,
        help="Run in rehydrated mode.",
    )
    parser.add_argument(
        "--disable-vad",
        action="store_true",
        default=True,
        help="Disable server-side VAD for compatible realtime models.",
    )
    parser.add_argument(
        "--skip-turn-taking",
        action="store_true",
        default=True,
        help="Skip turn-taking grading in the judge.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional explicit output directory under results/.",
    )
    return parser.parse_args()


def infer_service(model_name: str, service_override: str | None) -> str | None:
    if service_override:
        return service_override
    lower_model_name = model_name.lower()
    if lower_model_name.startswith("gpt") and "realtime" in lower_model_name:
        return "openai-realtime"
    return None


def build_run_command(args: argparse.Namespace, model_name: str) -> list[str]:
    command = [
        "uv",
        "run",
        "audio-arena",
        "run",
        args.benchmark,
        "--model",
        model_name,
    ]
    service_name = infer_service(model_name, args.service)
    if service_name:
        command.extend(["--service", service_name])
    if args.rehydrate:
        command.append("--rehydrate")
        command.extend(["--parallel", str(args.parallel)])
    if args.disable_vad:
        command.append("--disable-vad")
    return command


def build_judge_command(
    args: argparse.Namespace,
    run_dir: Path,
) -> list[str]:
    command = [
        "uv",
        "run",
        "audio-arena",
        "judge",
        str(run_dir),
        "--judge",
        args.judge,
    ]
    if args.judge_model:
        command.extend(["--judge-model", args.judge_model])
    if args.skip_turn_taking:
        command.append("--skip-turn-taking")
    return command


def extract_run_dir(stdout_text: str) -> Path:
    for line in stdout_text.splitlines():
        if line.startswith("Output directory: "):
            return (Path.cwd() / line.split(": ", 1)[1].strip()).resolve()
    raise ValueError("Failed to parse run directory from benchmark output.")


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0

    proportion = successes / total
    denominator = 1 + (z**2 / total)
    center = proportion + (z**2 / (2 * total))
    margin = z * math.sqrt(
        (proportion * (1 - proportion) / total) + (z**2 / (4 * total**2))
    )
    lower = (center - margin) / denominator
    upper = (center + margin) / denominator
    return max(0.0, lower), min(1.0, upper)


def bootstrap_quantile_interval(
    values: pd.Series,
    *,
    quantile: float,
    iterations: int = 2000,
    seed: int = 0,
) -> tuple[float | None, float | None]:
    clean_values = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if clean_values.size == 0:
        return None, None

    if clean_values.size == 1:
        single_value = float(clean_values[0])
        return single_value, single_value

    rng = np.random.default_rng(seed)
    bootstrap_samples = rng.choice(
        clean_values,
        size=(iterations, clean_values.size),
        replace=True,
    )
    quantiles = np.quantile(bootstrap_samples, quantile, axis=1)
    lower = float(np.quantile(quantiles, 0.025))
    upper = float(np.quantile(quantiles, 0.975))
    return lower, upper


def compute_overall_status_counts(judged_rows: list[dict]) -> dict[str, int]:
    pass_rows = 0
    fail_rows = 0
    incomplete_rows = 0

    for row in judged_rows:
        scores = row.get("scores", {})
        completed_scores = [value for value in scores.values() if value is not None]
        failed_scores = [value for value in completed_scores if value is False]
        if not completed_scores:
            incomplete_rows += 1
        elif failed_scores:
            fail_rows += 1
        else:
            pass_rows += 1

    return {
        "pass_rows": pass_rows,
        "fail_rows": fail_rows,
        "incomplete_rows": incomplete_rows,
    }


def compute_latency_summary(judged_df: pd.DataFrame, column_name: str) -> dict[str, float | None]:
    numeric_series = pd.to_numeric(judged_df[column_name], errors="coerce").dropna()
    if numeric_series.empty:
        return {
            f"{column_name}_mean": None,
            f"{column_name}_p50": None,
            f"{column_name}_p90": None,
            f"{column_name}_p95": None,
            f"{column_name}_p99": None,
        }

    return {
        f"{column_name}_mean": round(float(numeric_series.mean()), 2),
        f"{column_name}_p50": round(float(numeric_series.quantile(0.50)), 2),
        f"{column_name}_p90": round(float(numeric_series.quantile(0.90)), 2),
        f"{column_name}_p95": round(float(numeric_series.quantile(0.95)), 2),
        f"{column_name}_p99": round(float(numeric_series.quantile(0.99)), 2),
    }


def build_run_record(
    *,
    model_name: str,
    run_index: int,
    run_dir: Path,
    summary_payload: dict,
    runtime_payload: dict,
    judged_rows: list[dict],
) -> dict:
    judged_rows, summary_payload = normalize_turn_taking_for_report(judged_rows, summary_payload)
    judged_df = pd.DataFrame(judged_rows)
    status_counts = compute_overall_status_counts(judged_rows)
    category_totals = summary_payload.get("category_totals", {})
    total_turns = int(summary_payload.get("turns_scored", len(judged_rows)))
    latency_p95_ci_lower, latency_p95_ci_upper = bootstrap_quantile_interval(
        judged_df["latency_ms"],
        quantile=0.95,
        seed=run_index,
    )

    record = {
        "model_name": model_name,
        "run_index": run_index,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "benchmark_name": run_dir.parent.name,
        "judge_name": summary_payload.get("judge_name"),
        "judge_model": summary_payload.get("judge_model"),
        "judge_version": summary_payload.get("judge_version"),
        "judged_at": summary_payload.get("judged_at"),
        "runtime_mode": runtime_payload.get("mode"),
        "runtime_parallel": runtime_payload.get("parallel"),
        "disable_vad": runtime_payload.get("disable_vad"),
        "turns_scored": total_turns,
        "turn_taking_supported": summary_payload.get("turn_taking_supported"),
        "turn_taking_skip_reason": summary_payload.get("turn_taking_skip_reason"),
        "pass_rows": status_counts["pass_rows"],
        "fail_rows": status_counts["fail_rows"],
        "incomplete_rows": status_counts["incomplete_rows"],
        "pass_row_rate": status_counts["pass_rows"] / total_turns if total_turns else None,
        "fail_row_rate": status_counts["fail_rows"] / total_turns if total_turns else None,
        "empty_response_count": int(
            sum(
                1
                for row in judged_rows
                if str(row.get("assistant_text", "")).startswith("[EMPTY_RESPONSE")
            )
        ),
        "model_ended_session_count": int(
            sum(
                1
                for row in judged_rows
                if str(row.get("assistant_text", "")).startswith("[MODEL_ENDED_SESSION]")
            )
        ),
    }

    for count_name in [
        "pass_rows",
        "fail_rows",
        "empty_response_count",
        "model_ended_session_count",
    ]:
        lower_rate, upper_rate = wilson_interval(int(record[count_name]), total_turns)
        record[f"{count_name}_ci_lower"] = lower_rate * total_turns
        record[f"{count_name}_ci_upper"] = upper_rate * total_turns

    pass_row_rate_lower, pass_row_rate_upper = wilson_interval(record["pass_rows"], total_turns)
    fail_row_rate_lower, fail_row_rate_upper = wilson_interval(record["fail_rows"], total_turns)
    record["pass_row_rate_ci_lower"] = pass_row_rate_lower
    record["pass_row_rate_ci_upper"] = pass_row_rate_upper
    record["fail_row_rate_ci_lower"] = fail_row_rate_lower
    record["fail_row_rate_ci_upper"] = fail_row_rate_upper

    for score_name in SCORE_COLUMNS:
        passes = summary_payload.get("passes", {}).get(score_name)
        denominator = category_totals.get(score_name, total_turns)
        record[f"{score_name}_passes"] = passes
        record[f"{score_name}_applicable"] = denominator
        record[f"{score_name}_pass_rate"] = (
            passes / denominator if passes is not None and denominator else None
        )
        if passes is not None and denominator:
            lower_rate, upper_rate = wilson_interval(int(passes), int(denominator))
            record[f"{score_name}_pass_rate_ci_lower"] = lower_rate
            record[f"{score_name}_pass_rate_ci_upper"] = upper_rate
        else:
            record[f"{score_name}_pass_rate_ci_lower"] = None
            record[f"{score_name}_pass_rate_ci_upper"] = None

    record.update(compute_latency_summary(judged_df, "latency_ms"))
    record.update(compute_latency_summary(judged_df, "ttfb_ms"))
    record["latency_ms_p95_ci_lower"] = latency_p95_ci_lower
    record["latency_ms_p95_ci_upper"] = latency_p95_ci_upper
    return record


def aggregate_metric_summary(run_results_df: pd.DataFrame) -> pd.DataFrame:
    metric_rows: list[dict] = []
    for model_name, model_df in run_results_df.groupby("model_name", sort=False):
        for score_name in SCORE_COLUMNS:
            rate_column = f"{score_name}_pass_rate"
            pass_column = f"{score_name}_passes"
            applicable_column = f"{score_name}_applicable"
            lower_column = f"{score_name}_pass_rate_ci_lower"
            upper_column = f"{score_name}_pass_rate_ci_upper"
            rates = pd.to_numeric(model_df[rate_column], errors="coerce")
            passes = pd.to_numeric(model_df[pass_column], errors="coerce")
            applicable_values = pd.to_numeric(model_df[applicable_column], errors="coerce")
            lower_bounds = pd.to_numeric(model_df[lower_column], errors="coerce")
            upper_bounds = pd.to_numeric(model_df[upper_column], errors="coerce")
            std_value = float(rates.std(ddof=0)) if len(rates.dropna()) else 0.0
            metric_rows.append(
                {
                    "model_name": model_name,
                    "metric": score_name,
                    "mean_pass_rate": float(rates.mean()),
                    "std_pass_rate": 0.0 if math.isnan(std_value) else std_value,
                    "mean_pass_rate_ci_lower": float(lower_bounds.mean()),
                    "mean_pass_rate_ci_upper": float(upper_bounds.mean()),
                    "mean_pass_count": float(passes.mean()),
                    "mean_applicable": float(applicable_values.mean()),
                    "runs": int(len(model_df)),
                }
            )
    return pd.DataFrame(metric_rows)


def aggregate_run_summary(run_results_df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        "pass_rows",
        "fail_rows",
        "incomplete_rows",
        "pass_row_rate",
        "pass_row_rate_ci_lower",
        "pass_row_rate_ci_upper",
        "fail_row_rate",
        "fail_row_rate_ci_lower",
        "fail_row_rate_ci_upper",
        "empty_response_count",
        "empty_response_count_ci_lower",
        "empty_response_count_ci_upper",
        "model_ended_session_count",
        "model_ended_session_count_ci_lower",
        "model_ended_session_count_ci_upper",
        "pass_rows_ci_lower",
        "pass_rows_ci_upper",
        "fail_rows_ci_lower",
        "fail_rows_ci_upper",
        "latency_ms_mean",
        "latency_ms_p50",
        "latency_ms_p90",
        "latency_ms_p95",
        "latency_ms_p95_ci_lower",
        "latency_ms_p95_ci_upper",
        "latency_ms_p99",
        "ttfb_ms_mean",
        "ttfb_ms_p50",
        "ttfb_ms_p90",
        "ttfb_ms_p95",
        "ttfb_ms_p99",
    ]
    summary_df = (
        run_results_df.groupby("model_name", sort=False)[numeric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary_df.columns = [
        "model_name"
        if index == 0
        else f"{column_name}_{stat_name}"
        for index, (column_name, stat_name) in enumerate(summary_df.columns)
    ]
    return summary_df


def create_metric_plot(metric_summary_df: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    metrics = SCORE_COLUMNS
    models = list(metric_summary_df["model_name"].drop_duplicates())
    x_positions = list(range(len(metrics)))
    bar_width = 0.8 / max(len(models), 1)
    palette = sns.color_palette("crest", n_colors=max(len(models), 1))

    figure, axis = plt.subplots(figsize=(15, 7))
    for model_index, model_name in enumerate(models):
        model_metric_df = (
            metric_summary_df[metric_summary_df["model_name"] == model_name]
            .set_index("metric")
            .reindex(metrics)
        )
        offsets = [
            position - (0.4 - bar_width / 2) + model_index * bar_width
            for position in x_positions
        ]
        lower_errors = (
            model_metric_df["mean_pass_rate"] - model_metric_df["mean_pass_rate_ci_lower"]
        ).fillna(0.0)
        upper_errors = (
            model_metric_df["mean_pass_rate_ci_upper"] - model_metric_df["mean_pass_rate"]
        ).fillna(0.0)
        axis.bar(
            offsets,
            model_metric_df["mean_pass_rate"],
            width=bar_width,
            yerr=np.vstack([lower_errors.to_numpy(), upper_errors.to_numpy()]),
            label=model_name,
            color=palette[model_index],
            capsize=5,
            edgecolor="black",
            linewidth=0.8,
        )
        for offset, rate in zip(offsets, model_metric_df["mean_pass_rate"]):
            if pd.isna(rate):
                axis.text(offset, 0.01, "N/A", rotation=90, ha="center", va="bottom", fontsize=8)

    axis.set_title("Pass Rate by Metric")
    axis.set_xlabel("")
    axis.set_ylabel("Pass rate")
    axis.set_ylim(0, 1.05)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(metrics, rotation=25, ha="right")
    axis.legend(title="Model")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def create_count_plot(
    summary_df: pd.DataFrame,
    *,
    mean_column: str,
    lower_column: str,
    upper_column: str,
    title: str,
    y_label: str,
    output_path: Path,
) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    palette = sns.color_palette("crest", n_colors=len(summary_df))
    figure, axis = plt.subplots(figsize=(11, 6))
    axis.bar(
        summary_df["model_name"],
        summary_df[mean_column],
        yerr=np.vstack(
            [
                (summary_df[mean_column] - summary_df[lower_column]).fillna(0.0).to_numpy(),
                (summary_df[upper_column] - summary_df[mean_column]).fillna(0.0).to_numpy(),
            ]
        ),
        capsize=5,
        color=palette,
        edgecolor="black",
        linewidth=0.8,
    )
    axis.set_title(title)
    axis.set_xlabel("")
    axis.set_ylabel(y_label)
    axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def create_markdown_summary(
    args: argparse.Namespace,
    run_results_df: pd.DataFrame,
    metric_summary_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        "# Model Comparison Summary",
        "",
        "## Setup",
        f"- benchmark: {args.benchmark}",
        f"- models: {', '.join(run_results_df['model_name'].drop_duplicates().tolist())}",
        f"- runs_per_model: {args.runs_per_model}",
        f"- service_override: {args.service or 'auto'}",
        f"- rehydrate: {args.rehydrate}",
        f"- disable_vad: {args.disable_vad}",
        f"- parallel: {args.parallel}",
        f"- judge: {args.judge}",
        f"- judge_model: {args.judge_model or 'default'}",
        f"- skip_turn_taking: {args.skip_turn_taking}",
        "",
        "## Run Paths",
    ]

    for _, row in run_results_df.iterrows():
        lines.append(
            f"- {row['model_name']} run {int(row['run_index'])}: {row['run_dir']}"
        )

    lines.extend(
        [
            "",
            "## Model Summary",
        ]
    )
    for _, row in summary_df.iterrows():
        lines.append(
            f"- {row['model_name']}: pass_rows={row['pass_rows_mean']:.2f}, "
            f"fail_rows={row['fail_rows_mean']:.2f} "
            f"[95% CI {row['fail_rows_ci_lower_mean']:.2f}, {row['fail_rows_ci_upper_mean']:.2f}], "
            f"empty_responses={row['empty_response_count_mean']:.2f}, "
            f"ended_session={row['model_ended_session_count_mean']:.2f}, "
            f"latency_p95={row['latency_ms_p95_mean']:.2f}ms "
            f"[95% CI {row['latency_ms_p95_ci_lower_mean']:.2f}, {row['latency_ms_p95_ci_upper_mean']:.2f}]"
        )

    lines.extend(
        [
            "",
            "## Metric Averages",
        ]
    )
    for _, row in metric_summary_df.iterrows():
        if pd.isna(row["mean_pass_rate"]):
            lines.append(f"- {row['model_name']} | {row['metric']}: N/A (not evaluated)")
            continue
        lines.append(
            f"- {row['model_name']} | {row['metric']}: "
            f"{row['mean_pass_count']:.2f}/{row['mean_applicable']:.2f} "
            f"({row['mean_pass_rate'] * 100:.1f}% "
            f"[95% CI {row['mean_pass_rate_ci_lower'] * 100:.1f}%, {row['mean_pass_rate_ci_upper'] * 100:.1f}%])"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    models = [model_name.strip() for model_name in args.models.split(",") if model_name.strip()]
    if not models:
        raise ValueError("At least one model is required.")

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    default_output_dir = (
        Path.cwd()
        / "results"
        / args.benchmark
        / f"{timestamp}_model_comparison"
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_output_dir.resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    run_records: list[dict] = []
    command_records: list[dict] = []

    for model_name in models:
        for run_index in range(1, args.runs_per_model + 1):
            print(
                f"[compare] benchmark={args.benchmark} model={model_name} run={run_index}/{args.runs_per_model}",
                flush=True,
            )
            run_command = build_run_command(args, model_name)
            run_process = subprocess.run(
                run_command,
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
            )
            if run_process.returncode != 0:
                raise RuntimeError(
                    f"Benchmark run failed for {model_name} run {run_index}.\n"
                    f"stdout tail:\n{run_process.stdout[-4000:]}\n"
                    f"stderr tail:\n{run_process.stderr[-4000:]}"
                )

            run_dir = extract_run_dir(run_process.stdout)
            command_records.append(
                {
                    "phase": "run",
                    "model_name": model_name,
                    "run_index": run_index,
                    "command": " ".join(run_command),
                    "run_dir": str(run_dir),
                }
            )

            print(
                f"[compare] judging model={model_name} run={run_index} run_dir={run_dir}",
                flush=True,
            )
            judge_command = build_judge_command(args, run_dir)
            judge_process = subprocess.run(
                judge_command,
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
            )
            if judge_process.returncode != 0:
                raise RuntimeError(
                    f"Judge failed for {model_name} run {run_index}.\n"
                    f"stdout tail:\n{judge_process.stdout[-4000:]}\n"
                    f"stderr tail:\n{judge_process.stderr[-4000:]}"
                )
            command_records.append(
                {
                    "phase": "judge",
                    "model_name": model_name,
                    "run_index": run_index,
                    "command": " ".join(judge_command),
                    "run_dir": str(run_dir),
                }
            )

            summary_payload = json.loads(
                (run_dir / "openai_summary.json").read_text(encoding="utf-8")
            )
            runtime_payload = json.loads(
                (run_dir / "runtime.json").read_text(encoding="utf-8")
            )
            judged_rows = load_jsonl(run_dir / "openai_judged.jsonl")
            run_records.append(
                build_run_record(
                    model_name=model_name,
                    run_index=run_index,
                    run_dir=run_dir,
                    summary_payload=summary_payload,
                    runtime_payload=runtime_payload,
                    judged_rows=judged_rows,
                )
            )

    run_results_df = pd.DataFrame(run_records)
    if run_results_df.empty:
        raise ValueError("No successful runs were recorded.")

    metric_summary_df = aggregate_metric_summary(run_results_df)
    summary_df = aggregate_run_summary(run_results_df)

    run_results_df.to_csv(output_dir / "run_results.csv", index=False, encoding="utf-8")
    metric_summary_df.to_csv(
        output_dir / "metric_summary.csv",
        index=False,
        encoding="utf-8",
    )
    summary_df.to_csv(output_dir / "model_summary.csv", index=False, encoding="utf-8")
    pd.DataFrame(command_records).to_csv(
        output_dir / "commands.csv",
        index=False,
        encoding="utf-8",
    )

    create_metric_plot(metric_summary_df, output_dir / "pass_rates_by_metric.png")
    create_count_plot(
        summary_df,
        mean_column="fail_rows_mean",
        lower_column="fail_rows_ci_lower_mean",
        upper_column="fail_rows_ci_upper_mean",
        title="Failed Turns by Model",
        y_label="Failed turns",
        output_path=output_dir / "failed_turns.png",
    )
    create_count_plot(
        summary_df,
        mean_column="latency_ms_p95_mean",
        lower_column="latency_ms_p95_ci_lower_mean",
        upper_column="latency_ms_p95_ci_upper_mean",
        title="Latency p95 by Model",
        y_label="Latency p95 (ms)",
        output_path=output_dir / "latency_p95.png",
    )
    create_count_plot(
        summary_df,
        mean_column="model_ended_session_count_mean",
        lower_column="model_ended_session_count_ci_lower_mean",
        upper_column="model_ended_session_count_ci_upper_mean",
        title="MODEL_ENDED_SESSION Count by Model",
        y_label="Turns",
        output_path=output_dir / "model_ended_session.png",
    )
    create_markdown_summary(
        args,
        run_results_df,
        metric_summary_df,
        summary_df,
        output_dir / "summary.md",
    )

    metadata = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "benchmark": args.benchmark,
        "models": models,
        "runs_per_model": args.runs_per_model,
        "service_override": args.service,
        "rehydrate": args.rehydrate,
        "disable_vad": args.disable_vad,
        "parallel": args.parallel,
        "judge": args.judge,
        "judge_model": args.judge_model,
        "skip_turn_taking": args.skip_turn_taking,
        "output_dir": str(output_dir),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"[compare] output_dir={output_dir}", flush=True)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "run_results_csv": str(output_dir / "run_results.csv"),
                "metric_summary_csv": str(output_dir / "metric_summary.csv"),
                "model_summary_csv": str(output_dir / "model_summary.csv"),
                "summary_md": str(output_dir / "summary.md"),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
