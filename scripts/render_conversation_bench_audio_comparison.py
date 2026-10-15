import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PAPER = "#f7f4ee"
AXIS_BG = "#fbf9f5"
GRID = "#ddd7cc"
TEXT = "#2c2823"
EDGE = "#2d261f"

SERIES_COLORS = {
    "Synthetic TTS": "#dd7f00",
    "Real Audio person1": "#1c7f77",
    "Real Audio person2": "#3fc4b7",
}


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/System/Library/Fonts/Avenir Next.ttc",
                "/System/Library/Fonts/HelveticaNeue.ttc",
                "/System/Library/Fonts/Helvetica.ttc",
            ]
        )
    else:
        candidates.extend(
            [
                "/System/Library/Fonts/Avenir Next.ttc",
                "/System/Library/Fonts/Avenir.ttc",
                "/System/Library/Fonts/HelveticaNeue.ttc",
                "/System/Library/Fonts/Helvetica.ttc",
                "/Library/Fonts/Arial Unicode.ttf",
            ]
        )

    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def pct(num: int, den: int) -> float:
    return round((num / den) * 100, 1) if den else 0.0


def wilson_interval(num: int, den: int, z: float = 1.96) -> tuple[float, float]:
    if den == 0:
        return 0.0, 0.0
    phat = num / den
    denom = 1 + (z * z) / den
    center = (phat + (z * z) / (2 * den)) / denom
    margin = (
        z
        * math.sqrt((phat * (1 - phat) / den) + (z * z) / (4 * den * den))
        / denom
    )
    return center - margin, center + margin


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def draw_centered(draw: ImageDraw.ImageDraw, x: float, y: float, text: str, font, fill: str) -> None:
    width, height = text_size(draw, text, font)
    draw.text((x - width / 2, y - height / 2), text, font=font, fill=fill)


def load_rows() -> list[dict]:
    run_paths = {
        "Synthetic TTS": Path("runs/conversation_bench/20260318T144759_gpt-realtime-1.5_104fd465"),
        "Real Audio person1": Path("runs/conversation_bench/20260318T144759_gpt-realtime-1.5_36447e17"),
        "Real Audio person2": Path("runs/conversation_bench/20260318T144759_gpt-realtime-1.5_5a7438fd"),
    }
    metric_defs = [
        ("tool_use_correct", "Tool Use", "category_totals.tool_use_correct"),
        ("instruction_following", "Instruction", "turns_scored"),
        ("kb_grounding", "KB Grounding", "turns_scored"),
        ("ambiguity_handling", "Ambiguity", "category_totals.ambiguity_handling"),
        ("state_tracking", "State", "category_totals.state_tracking"),
    ]

    rows = []
    for label, run_dir in run_paths.items():
        summary = json.loads((run_dir / "openai_summary.json").read_text())
        runtime = json.loads((run_dir / "runtime.json").read_text())
        values = []
        for key, display, total_key in metric_defs:
            if total_key == "turns_scored":
                denominator = summary["turns_scored"]
            else:
                _, cat_key = total_key.split(".", 1)
                denominator = summary["category_totals"][cat_key]
            numerator = summary["passes"][key]
            lower, upper = wilson_interval(numerator, denominator)
            values.append(
                {
                    "key": key,
                    "label": display,
                    "numerator": numerator,
                    "denominator": denominator,
                    "percent": pct(numerator, denominator),
                    "lower_percent": lower * 100,
                    "upper_percent": upper * 100,
                }
            )
        rows.append(
            {
                "label": label,
                "audio_source": runtime["audio_source"],
                "failed_turns": runtime.get("failed_turns", []),
                "values": values,
            }
        )
    return rows


def render_chart(output_path: Path) -> None:
    rows = load_rows()
    metrics = [item["label"] for item in rows[0]["values"]]

    width = 2000
    height = 1360
    margin_left = 180
    margin_right = 90
    margin_top = 220
    margin_bottom = 250
    plot_left = margin_left
    plot_top = margin_top
    plot_right = width - margin_right
    plot_bottom = height - margin_bottom
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    title_font = load_font(54)
    subtitle_font = load_font(24)
    axis_font = load_font(24)
    tick_font = load_font(22)
    note_font = load_font(20)
    label_font = load_font(18)
    legend_font = load_font(22)
    small_font = load_font(17)

    image = Image.new("RGB", (width, height), PAPER)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        [(plot_left, plot_top), (plot_right, plot_bottom)],
        radius=26,
        fill=AXIS_BG,
        outline=GRID,
        width=2,
    )

    title = "conversation bench audio comparison"
    subtitle = "gpt-realtime-1.5, rehydrated, no vad, judged with gpt-5.2, skip-turn-taking"
    draw.text((plot_left, 55), title, font=title_font, fill=TEXT)
    draw.text((plot_left, 118), subtitle, font=subtitle_font, fill="#5e564d")

    legend_y = 165
    legend_x = plot_left
    for row in rows:
        color = SERIES_COLORS[row["label"]]
        draw.rounded_rectangle(
            [(legend_x, legend_y), (legend_x + 28, legend_y + 18)],
            radius=4,
            fill=color,
            outline=EDGE,
            width=2,
        )
        draw.text((legend_x + 40, legend_y - 6), row["label"], font=legend_font, fill=TEXT)
        legend_x += 320

    for pct_value in range(0, 101, 20):
        y = plot_bottom - (pct_value / 100.0) * plot_height
        draw.line([(plot_left, y), (plot_right, y)], fill=GRID, width=2)
        label = f"{pct_value}%"
        label_w, label_h = text_size(draw, label, tick_font)
        draw.text((plot_left - label_w - 18, y - label_h / 2), label, font=tick_font, fill=TEXT)

    draw.line([(plot_left, plot_top), (plot_left, plot_bottom)], fill=EDGE, width=3)
    draw.line([(plot_left, plot_bottom), (plot_right, plot_bottom)], fill=EDGE, width=3)

    group_count = len(metrics)
    group_width = plot_width / group_count
    bar_width = 68
    inner_gap = 18
    error_bar_cap = 10

    for idx, metric in enumerate(metrics):
        group_center = plot_left + group_width * idx + group_width / 2
        offsets = [-(bar_width + inner_gap), 0, bar_width + inner_gap]
        for row, offset in zip(rows, offsets):
            metric_row = row["values"][idx]
            value = metric_row["percent"]
            x0 = group_center + offset - bar_width / 2
            x1 = x0 + bar_width
            y1 = plot_bottom
            y0 = plot_bottom - (value / 100.0) * plot_height
            draw.rounded_rectangle(
                [(x0, y0), (x1, y1)],
                radius=10,
                fill=SERIES_COLORS[row["label"]],
                outline=EDGE,
                width=3,
            )

            err_low = plot_bottom - (metric_row["lower_percent"] / 100.0) * plot_height
            err_high = plot_bottom - (metric_row["upper_percent"] / 100.0) * plot_height
            center_x = (x0 + x1) / 2
            draw.line([(center_x, err_high), (center_x, err_low)], fill=EDGE, width=3)
            draw.line(
                [(center_x - error_bar_cap, err_high), (center_x + error_bar_cap, err_high)],
                fill=EDGE,
                width=3,
            )
            draw.line(
                [(center_x - error_bar_cap, err_low), (center_x + error_bar_cap, err_low)],
                fill=EDGE,
                width=3,
            )

            value_text = f"{value:.1f}%"
            draw_centered(draw, (x0 + x1) / 2, y0 - 18, value_text, label_font, TEXT)
            count_text = f"{metric_row['numerator']}/{metric_row['denominator']}"
            draw_centered(draw, (x0 + x1) / 2, y1 + 22, count_text, label_font, "#5e564d")

        label_y = plot_bottom + 62
        label_text = metric
        label_w, label_h = text_size(draw, label_text, axis_font)
        draw.text((group_center - label_w / 2, label_y), label_text, font=axis_font, fill=TEXT)

    y_label = "pass rate"
    temp = Image.new("RGBA", (220, 60), (0, 0, 0, 0))
    temp_draw = ImageDraw.Draw(temp)
    temp_draw.text((0, 0), y_label, font=axis_font, fill=TEXT)
    rotated = temp.rotate(90, expand=True)
    image.paste(rotated, (68, plot_top + plot_height // 2 - rotated.height // 2), rotated)

    notes = [
        "error bars show 95% Wilson intervals for each pass-rate proportion",
        "synthetic: failed turn 67 was a hung no-vad turn finalized as an explicit empty-response failure",
        "person1: failed turn 36 was a hung no-vad turn finalized as an explicit empty-response failure",
        "person2: no forced failed turns",
    ]
    notes_y = height - 126
    for note in notes:
        draw.text((plot_left, notes_y), note, font=small_font, fill="#5e564d")
        notes_y += 26

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")


if __name__ == "__main__":
    render_chart(Path("results/plots/conversation_bench_audio_comparison.png"))
