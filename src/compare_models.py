from __future__ import annotations

import csv
import json
from pathlib import Path

from model_utils import METRICS_DIR, ensure_output_dirs


SUMMARY_SUFFIX = "_summary.json"


def load_summaries() -> list[dict]:
    summaries = []
    for path in sorted(METRICS_DIR.glob(f"*{SUMMARY_SUFFIX}")):
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
            if "model" in data:
                summaries.append(data)
    return summaries


def save_csv(summaries: list[dict], output_path: Path) -> None:
    fieldnames = [
        "model",
        "accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "test_loss",
        "training_time_seconds",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: row.get(key) for key in fieldnames})


def save_markdown(summaries: list[dict], output_path: Path) -> None:
    lines = [
        "| Model | Accuracy | Precision | Recall | F1-score | Test Loss | Training Time (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        lines.append(
            "| {model} | {accuracy:.4f} | {precision_weighted:.4f} | {recall_weighted:.4f} | "
            "{f1_weighted:.4f} | {test_loss:.4f} | {training_time_seconds:.2f} |".format(
                **summary
            )
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_output_dirs()
    summaries = load_summaries()
    if not summaries:
        raise FileNotFoundError(
            "No model summaries found in results/metrics. Train models before comparing them."
        )

    save_csv(summaries, METRICS_DIR / "model_comparison.csv")
    save_markdown(summaries, METRICS_DIR / "model_comparison.md")
    print(f"Saved comparison for {len(summaries)} model(s)")


if __name__ == "__main__":
    main()
