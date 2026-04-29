#!/usr/bin/env python3
"""Post-process Ocean SpGEMM benchmark stats.json files into:
  - GFLOPS bar chart across matrices
  - Spy plot per matrix with detected segments overlaid (color = class)
  - Segmentation overhead breakdown (density / threshold / CC / classify) vs total
"""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

CLASS_COLORS = {
    "sparse_uniform": "#4c78a8",
    "dense":          "#e45756",
    "power_law":      "#f58518",
    "banded":         "#54a24b",
}
STAGE_KEYS = [
    ("time_density_ms",   "density"),
    ("time_threshold_ms", "threshold"),
    ("time_cc_ms",        "CC"),
    ("time_classify_ms",  "classify"),
]


def load_stats(results_dir: Path, matrix: str) -> dict:
    path = results_dir / f"{matrix}_stats.json"
    if not path.exists():
        print(f"  [warn] missing {path}", file=sys.stderr)
        return {}
    with open(path) as f:
        return json.load(f)


def plot_gflops(stats_by_matrix: dict, out: Path) -> None:
    matrices = list(stats_by_matrix.keys())
    gflops = [stats_by_matrix[m].get("gflops", 0.0) for m in matrices]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(matrices, gflops, color="#4c78a8")
    ax.set_ylabel("GFLOPS")
    ax.set_title("Ocean SpGEMM (block-segmented) — GFLOPS")
    for i, v in enumerate(gflops):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_overhead(stats_by_matrix: dict, out: Path) -> None:
    matrices = list(stats_by_matrix.keys())
    n = len(matrices)
    fig, (ax_abs, ax_pct) = plt.subplots(1, 2, figsize=(12, 4.5))

    bottoms_abs = np.zeros(n)
    totals = np.array([stats_by_matrix[m].get("total_time", 0.0) for m in matrices])
    for key, label in STAGE_KEYS:
        vals = np.array([
            stats_by_matrix[m].get("segmentation", {}).get(key, 0.0)
            for m in matrices
        ])
        ax_abs.bar(matrices, vals, bottom=bottoms_abs, label=label)
        bottoms_abs += vals

    ax_abs.bar(matrices, np.maximum(totals - bottoms_abs, 0), bottom=bottoms_abs,
               label="other SpGEMM", color="#cccccc")
    ax_abs.set_ylabel("time (ms / iter)")
    ax_abs.set_title("Stage breakdown (absolute)")
    ax_abs.legend(fontsize=8)

    seg_total = bottoms_abs
    pct = np.where(totals > 0, 100.0 * seg_total / totals, 0.0)
    ax_pct.bar(matrices, pct, color="#e45756")
    ax_pct.set_ylabel("segmentation overhead (% of total)")
    ax_pct.set_title("Segmentation overhead vs total")
    for i, v in enumerate(pct):
        ax_pct.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_spy(matrix: str, stats: dict, mtx_path: Path, out: Path) -> None:
    try:
        from scipy.io import mmread
    except ImportError:
        print("  [warn] scipy not available; skipping spy plots", file=sys.stderr)
        return
    if not mtx_path.exists():
        print(f"  [warn] missing {mtx_path}; skipping spy plot for {matrix}", file=sys.stderr)
        return

    print(f"  reading {mtx_path} ...")
    A = mmread(str(mtx_path)).tocoo()
    fig, ax = plt.subplots(figsize=(7, 7))
    # Subsample for very large matrices
    max_pts = 200_000
    if A.nnz > max_pts:
        idx = np.random.default_rng(0).choice(A.nnz, max_pts, replace=False)
        ax.scatter(A.col[idx], A.row[idx], s=0.2, c="black", marker=",")
    else:
        ax.scatter(A.col, A.row, s=0.2, c="black", marker=",")
    ax.set_xlim(0, A.shape[1])
    ax.set_ylim(A.shape[0], 0)
    ax.set_aspect("equal")
    ax.set_title(f"{matrix}  (segments overlaid)")

    seen = set()
    for seg in stats.get("segmentation", {}).get("segments", []):
        cls = seg.get("class", "sparse_uniform")
        color = CLASS_COLORS.get(cls, "#888888")
        rect = patches.Rectangle(
            (seg["col_left"], seg["row_start"]),
            max(seg["col_right"] - seg["col_left"], 1),
            max(seg["row_end"]  - seg["row_start"], 1),
            linewidth=1.0, edgecolor=color, facecolor=color, alpha=0.18,
            label=cls if cls not in seen else None,
        )
        seen.add(cls)
        ax.add_patch(rect)
    if seen:
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--matrices", required=True,
                    help="comma-separated, e.g. 144,road_usa,cant,belgium_osm")
    ap.add_argument("--mtx-dir", type=Path, default=None,
                    help="directory containing <name>/<name>.mtx for spy plots")
    ap.add_argument("--out", type=Path, default=Path("plots"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    matrices = [m.strip() for m in args.matrices.split(",") if m.strip()]

    stats_by_matrix = {}
    for m in matrices:
        s = load_stats(args.results_dir, m)
        if s:
            stats_by_matrix[m] = s

    if not stats_by_matrix:
        print("no stats files found", file=sys.stderr)
        return 1

    print("rendering gflops.png ...")
    plot_gflops(stats_by_matrix, args.out / "gflops.png")

    print("rendering overhead_breakdown.png ...")
    plot_overhead(stats_by_matrix, args.out / "overhead_breakdown.png")

    if args.mtx_dir is not None:
        for m, s in stats_by_matrix.items():
            mtx = args.mtx_dir / m / f"{m}.mtx"
            print(f"rendering spy_{m}.png ...")
            plot_spy(m, s, mtx, args.out / f"spy_{m}.png")

    print(f"done -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
