# GenAI disclosure: Claude (Anthropic) generated this file. Brian Onieal reviewed it.
"""Collect evaluation files from every run into one table.

Run it with ``python -m gram_sdpo.report --runs runs``. It writes
``results.csv`` and ``results.md`` into the runs folder.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from gram_sdpo.logging_utils import configure_logging, get_logger

logger = get_logger(__name__)

BASE_COLUMNS = ["run", "checkpoint", "step", "seed", "gram_weight", "kl_weight", "gram_distance"]
BENCHMARK_METRICS = ["acc_at_k", "mean_response_tokens", "mean_epistemic_tokens"]


def collect_rows(runs_root: str | Path) -> list[dict[str, Any]]:
    """Read every ``<run>/eval/*.json`` file under ``runs_root``.

    Args:
        runs_root: Folder that holds the run folders.

    Returns:
        One flat row per evaluated checkpoint, sorted by run and step.
        Benchmark metrics become columns such as ``aime24_acc_at_k``.
    """
    rows = []
    for path in sorted(Path(runs_root).glob("*/eval/*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        row = {column: result.get(column) for column in BASE_COLUMNS}
        for benchmark, metrics in result.get("benchmarks", {}).items():
            for metric in BENCHMARK_METRICS:
                row[f"{benchmark}_{metric}"] = metrics.get(metric)
        rows.append(row)
    rows.sort(key=lambda r: (str(r["run"]), int(r["step"] or 0)))
    return rows


def to_markdown(rows: list[dict[str, Any]]) -> str:
    """Render rows as a Markdown table.

    Args:
        rows: Rows from ``collect_rows``.

    Returns:
        The table as text, or a one-line note when there are no rows.
    """
    if not rows:
        return "No evaluation files found.\n"
    columns = list(dict.fromkeys(key for row in rows for key in row))

    def cell(value: Any) -> str:
        """Format one value, with four decimals for floats."""
        if isinstance(value, float):
            return f"{value:.4f}"
        return "" if value is None else str(value)

    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines += ["| " + " | ".join(cell(row.get(c)) for c in columns) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def write_report(runs_root: str | Path) -> tuple[Path, Path]:
    """Write ``results.csv`` and ``results.md`` into ``runs_root``.

    Args:
        runs_root: Folder that holds the run folders.

    Returns:
        The paths of the two files written.
    """
    root = Path(runs_root)
    rows = collect_rows(root)
    columns = list(dict.fromkeys(key for row in rows for key in row)) or BASE_COLUMNS
    csv_path, md_path = root / "results.csv", root / "results.md"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    md_path.write_text(to_markdown(rows), encoding="utf-8")
    logger.info("Wrote %d rows to %s and %s", len(rows), csv_path, md_path)
    return csv_path, md_path


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point for the results table."""
    parser = argparse.ArgumentParser(description="Collect evaluation results into one table.")
    parser.add_argument("--runs", default="runs", help="Folder that holds the run folders.")
    args = parser.parse_args(argv)
    configure_logging()
    write_report(args.runs)


if __name__ == "__main__":
    main()
