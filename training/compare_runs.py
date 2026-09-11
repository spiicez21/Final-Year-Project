"""Compare training runs by their held-out loss, and plot the curves.

Reads `training_summary.json` from each adapter directory given (written by
train_adapter.py since 2026-09-10) and reports the numbers that matter for
choosing a configuration:

  * best eval loss, and the step it happened at
  * final eval loss -- what you get if you just keep the last checkpoint
  * the gap between final train and best eval loss (memorisation signal)
  * wall-clock time

Usage:
    .venv/Scripts/python.exe training/compare_runs.py \\
        training/adapters/modern_r16_a32_bartender_ho_legacy \\
        training/adapters/modern_r16_a32_e6_bartender_ho_new \\
        --plot paper/figures/loss_curves_bartender.png
"""
import argparse
import json
from pathlib import Path


def load(run_dir: Path) -> dict:
    path = run_dir / "training_summary.json"
    if not path.exists():
        raise SystemExit(f"{path} not found -- was this run made before summaries existed?")
    s = json.loads(path.read_text(encoding="utf-8"))
    s["_dir"] = run_dir
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--plot", type=Path, default=None)
    args = ap.parse_args()

    runs = [load(r) for r in args.runs]

    print("%-44s %6s %9s %9s %9s %8s %7s" % (
        "run", "steps", "best_eval", "@step", "final_ev", "train", "wall_s"))
    for s in runs:
        ev = s["eval_loss_curve"]
        best_step = min(ev, key=lambda p: p[1])[0] if ev else None
        final_eval = ev[-1][1] if ev else None
        print("%-44s %6d %9s %9s %9s %8s %7.1f" % (
            s["output_dir"][:44], s["global_steps"],
            "%.4f" % s["best_eval_loss"] if s["best_eval_loss"] is not None else "-",
            best_step if best_step is not None else "-",
            "%.4f" % final_eval if final_eval is not None else "-",
            "%.4f" % s["final_train_loss"] if s["final_train_loss"] is not None else "-",
            s["wall_seconds"]))

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=160)
        palette = ["#2a78d6", "#d95f02", "#1b9e77", "#7570b3"]
        for i, s in enumerate(runs):
            c = palette[i % len(palette)]
            label = s["output_dir"].split("_ho_")[-1] if "_ho_" in s["output_dir"] else s["output_dir"]
            if s["train_loss_curve"]:
                x, y = zip(*s["train_loss_curve"])
                ax.plot(x, y, color=c, alpha=0.35, linewidth=1.2, linestyle="--",
                        label=f"{label} train")
            if s["eval_loss_curve"]:
                x, y = zip(*s["eval_loss_curve"])
                ax.plot(x, y, color=c, linewidth=2.0, marker="o", markersize=3,
                        label=f"{label} held-out")
                bx, by = min(s["eval_loss_curve"], key=lambda p: p[1])
                ax.scatter([bx], [by], color=c, s=60, zorder=5, edgecolor="white", linewidth=1.2)
        ax.set_xlabel("optimiser step")
        ax.set_ylabel("loss (completion tokens only)")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot)
        print(f"\nplot -> {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
