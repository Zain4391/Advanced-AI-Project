"""Figures for the report.

    python -m analysis.plots curve                          # Exp 2 learning curve(s)
    python -m analysis.plots curve --pattern "results/exp2_qlearning_curve_default_seed*.csv"

Curves from several run seeds are averaged (mean +/- std band). A* and random reference
lines come from results/exp1_*_default_val.csv when those files exist.
"""
from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FIG_DIR = Path("results/figures")


def _read(path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _ref(agent: str, key: str) -> float | None:
    p = Path(f"results/exp1_{agent}_default_val.csv")
    if not p.exists():
        return None
    return float(np.mean([float(r[key]) for r in _read(p)]))


def learning_curve(pattern: str, out: str | Path, title: str = "Independent Q-learners: team learning curve") -> Path:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no curve files match {pattern!r}")
    runs = [_read(f) for f in files]
    n = min(len(r) for r in runs)
    ep = np.array([float(r["episode"]) for r in runs[0][:n]])

    def stack(key):
        return np.array([[float(row[key]) for row in r[:n]] for r in runs])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, ref_key, label in (
        (axes[0], "eval_success", "success", "Team success rate (both escape)"),
        (axes[1], "eval_return", "return", "Team return"),
    ):
        y = stack(key)
        m, s = y.mean(0), y.std(0)
        ax.plot(ep, m, color="#3b82f6", lw=2, label=f"Q-learning (n={len(runs)} seeds)")
        ax.fill_between(ep, m - s, m + s, color="#3b82f6", alpha=0.2, lw=0)
        if key == "eval_success":
            p = stack("eval_partial").mean(0)
            ax.plot(ep, p, color="#f59e0b", lw=1.5, ls="--", label="one escapes (partial)")
        for agent, col in (("astar", "#16a34a"), ("random", "#9ca3af")):
            v = _ref(agent, ref_key)
            if v is not None:
                ax.axhline(v, color=col, ls=":", lw=1.5, label=f"{'A*' if agent == 'astar' else agent} (val)")
        ax.set_xlabel("training episodes")
        ax.set_title(label)
        ax.grid(alpha=0.3)
        if key == "eval_success":
            ax.set_ylim(-0.02, 1.02)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(title)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("curve")
    c.add_argument("--pattern", default="results/exp2_qlearning_curve_default_seed*.csv")
    c.add_argument("--out", default=str(FIG_DIR / "exp2_learning_curve.png"))
    c.add_argument("--title", default="Independent Q-learners: team learning curve")
    args = ap.parse_args()
    if args.cmd == "curve":
        print(learning_curve(args.pattern, args.out, args.title))


if __name__ == "__main__":
    main()
