"""Aggregate judge labels over benchmark-defined scoring populations."""

AGGREGATION_VERSION = "golden-scope-v1"
STATE_CATEGORIES = frozenset({
    "long_range_memory", "cross_turn_reasoning", "cancellation_flow",
    "implicit_correction", "error_recovery",
})


def score_applicability(expected_turn: dict, *, turn_taking_supported: bool) -> dict:
    categories = expected_turn.get("categories") or [expected_turn.get("category")]
    return {
        "turn_taking": turn_taking_supported,
        "tool_use_correct": bool(expected_turn.get("required_function_call")),
        "instruction_following": True,
        "kb_grounding": True,
        "ambiguity_handling": "ambiguous_entity" in categories,
        "state_tracking": bool(STATE_CATEGORIES.intersection(categories)),
    }


def aggregate_scores(records, judgments, expected_turns, *, turn_taking_supported):
    """Use judged turn IDs to select goldens, including for partial/sparse runs.

    Raw labels stay unchanged. Missing applicable labels count in the denominator
    and are listed explicitly, so a judge's null cannot shrink the population.
    Calls on turns without a required call remain a separate diagnostic: these
    may include legitimate early/late calls accepted by the judge's realignment.
    """
    if expected_turns is None:
        raise ValueError("expected_turns is required for benchmark-scoped aggregation")

    applicability = {}
    for turn in judgments:
        if not 0 <= turn < len(expected_turns):
            raise ValueError(f"No golden annotation for judged turn {turn}")
        applicability[turn] = score_applicability(
            expected_turns[turn], turn_taking_supported=turn_taking_supported
        )

    dimensions = score_applicability({}, turn_taking_supported=turn_taking_supported)
    passes = dict.fromkeys(dimensions, 0)
    totals = dict.fromkeys(dimensions, 0)
    missing = {dimension: [] for dimension in dimensions}
    for turn, judgment in judgments.items():
        for dimension, applicable in applicability[turn].items():
            if not applicable:
                continue
            totals[dimension] += 1
            score = judgment["scores"].get(dimension)
            if score is True:
                passes[dimension] += 1
            elif score is not False:
                missing[dimension].append(turn)

    calls_by_turn = {record["turn"]: record.get("tool_calls", []) for record in records}
    no_required_call = sorted(
        turn for turn in judgments if not applicability[turn]["tool_use_correct"]
    )
    diagnostics = {
        "no_required_call_turns": len(no_required_call),
        "turns_with_calls_without_required_call": [
            turn for turn in no_required_call if calls_by_turn.get(turn)
        ],
        "failed_turns_without_required_call": [
            turn for turn in no_required_call
            if judgments[turn]["scores"].get("tool_use_correct") is False
        ],
    }
    return {
        "aggregation_version": AGGREGATION_VERSION,
        "passes": passes,
        "category_totals": totals,
        "missing_scores": missing,
        "tool_use_diagnostics": diagnostics,
    }, applicability


def format_metric(passes: int, total: int) -> str:
    return f"{passes}/{total} ({passes / total * 100:.1f}%)" if total else "N/A (no applicable turns)"
