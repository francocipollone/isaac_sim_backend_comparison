# Copyright 2026 Franco Cipollone
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Merge two or more benchmark_backend.py JSON reports into a Markdown comparison table.

Usage
-----
    python compare_results.py /tmp/benchmark_usd.json /tmp/benchmark_usdrt.json
    python compare_results.py /tmp/benchmark_*.json --metric mean_ms --format markdown
    python compare_results.py /tmp/benchmark_*.json --format csv > out.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from pathlib import Path


def load_report(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _entry_value(entry: dict, metric: str):
    """Return the metric value of an entry, or ``None`` if the entry is skipped
    or doesn't carry the metric (e.g. cross-backend ``mean_ms`` queried on a
    per-iteration column).
    """
    if entry.get("skipped"):
        return None
    if metric not in entry:
        return None
    val = entry[metric]
    if isinstance(val, float) and math.isnan(val):
        return None
    return val


def merge(reports: list[dict], metric: str) -> tuple[list[str], list[str], list[list[float | None]]]:
    """Return (operation names, backend names, value matrix[op][backend])."""
    backends = [r["backend"] for r in reports]
    # by_label[backend][label] -> metric value (or None if skipped)
    by_label: dict[str, dict[str, float | None]] = {
        b: {entry["label"]: _entry_value(entry, metric) for entry in reports[i]["results"]}
        for i, b in enumerate(backends)
    }
    # Union of labels, in the order they appear in the first report.
    labels: list[str] = []
    for report in reports:
        for entry in report["results"]:
            if entry["label"] not in labels:
                labels.append(entry["label"])
    matrix: list[list[float | None]] = []
    for label in labels:
        matrix.append([by_label[b].get(label) for b in backends])
    return labels, backends, matrix


def to_markdown(
    reports: list[dict],
    labels: list[str],
    backends: list[str],
    matrix: list[list[float | None]],
    metric: str,
) -> str:
    """Render a Markdown table with one row per operation and one column per backend.

    The first backend column is the reference; each subsequent column shows the
    relative speedup (``reference / current``) as a percentage. Skipped
    operations (backend doesn't support the method) show as ``n/a``.
    """
    ref_backend = backends[0]
    out: list[str] = []
    out.append(f"# Backend comparison — metric: `{metric}` (lower is better)")
    out.append("")
    out.append(f"- Reference backend: **`{ref_backend}`**")
    out.append(f"- Compared backends: {', '.join(f'`{b}`' for b in backends[1:])}")
    out.append(f"- Operations measured: {len(labels)}")
    iters = reports[0].get("iters")
    num_prims = reports[0].get("num_prims")
    if num_prims is not None:
        out.append(f"- Primitives per call: **{num_prims}**")
    if iters is not None:
        out.append(f"- Iterations per op (excl. warmup): **{iters}**")
    out.append("")

    header = ["operation", ref_backend] + [f"{b} (vs {ref_backend})" for b in backends[1:]]
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "|".join(["---"] * len(header)) + "|")
    for label, row in zip(labels, matrix):
        ref = row[0]
        cells = [f"`{label}`"]
        cells.append("n/a" if ref is None else f"{ref:.4f}")
        for v in row[1:]:
            if ref is None or v is None:
                cells.append("n/a")
            elif v == 0:
                cells.append(f"{v:.4f}  (ref is 0)")
            else:
                pct = (ref / v) * 100.0
                # speedup > 100% means "this backend is faster than the reference"
                cells.append(f"{v:.4f}  ({pct:+.1f}%)")
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def to_csv(labels: list[str], backends: list[str], matrix: list[list[float | None]], metric: str) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["operation", *backends])
    for label, row in zip(labels, matrix):
        w.writerow([label, *(("" if v is None else v) for v in row)])
    return buf.getvalue()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reports", nargs="+", type=Path, help="One or more benchmark_backend.py JSON outputs.")
    p.add_argument("--metric", default="mean_ms", help="Metric column to compare (default: mean_ms).")
    p.add_argument(
        "--format",
        choices=["markdown", "csv"],
        default="markdown",
        help="Output format (default: markdown).",
    )
    args = p.parse_args()

    if len(args.reports) < 1:
        print("error: at least one report is required", file=sys.stderr)
        return 2

    reports = [load_report(p) for p in args.reports]
    labels, backends, matrix = merge(reports, args.metric)

    if args.format == "csv":
        sys.stdout.write(to_csv(labels, backends, matrix, args.metric))
    else:
        sys.stdout.write(to_markdown(reports, labels, backends, matrix, args.metric))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
