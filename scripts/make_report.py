"""Build README tables and figures from benchmark CSVs.

    python scripts/make_report.py results/main/results.csv results/ablation/results.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path("docs")


def dice_chart(main: pd.DataFrame, run: str = "patchcore") -> None:
    df = main[main.run == run].set_index("category").sort_index()
    cols = {"dice_quantile": "quantile (label-free)", "dice_synthcal": "SynthCal (label-free, ours)",
            "dice_oracle": "oracle (uses test labels)"}
    ax = df[list(cols)].rename(columns=cols).plot.bar(figsize=(12, 4), width=0.8, rot=45,
                                                       color=["#9aa5b1", "#d9480f", "#364fc7"])
    ax.set_ylabel("mask Dice")
    ax.set_xlabel("")
    ax.set_title(f"Mask quality per MVTec AD category ({run})")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.set_ylim(0, 0.9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT / "dice_per_category.png", dpi=130)
    plt.close()


def main_tables(main: pd.DataFrame) -> str:
    agg = main.groupby("run").mean(numeric_only=True)
    rows = ["| Method | Image AUROC | Pixel AUROC | AUPRO | Dice: quantile | Dice: synth-abs | **Dice: SynthCal** "
            "| Dice: oracle | Image F1 (label-free) | Latency (T4) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for run, r in agg.iterrows():
        rows.append(f"| {run} | {r.image_auroc:.3f} | {r.pixel_auroc:.3f} | {r.aupro:.3f} | {r.dice_quantile:.3f} | "
                    f"{r.dice_synthabs:.3f} | **{r.dice_synthcal:.3f}** | {r.dice_oracle:.3f} | "
                    f"{r.image_f1_synthcal:.3f} | {r.latency_ms:.0f} ms |")
    per = main.pivot_table(index="category", columns="run",
                           values=["image_auroc", "aupro", "dice_quantile", "dice_synthcal", "dice_oracle"])
    lines = ["| Category | " + " | ".join(f"{m}: {k}" for k in ["img AUROC", "AUPRO", "Dice q", "**Dice SC**",
                                                               "Dice oracle"] for m in per.columns.levels[1]) + " |"]
    lines.append("|---" * (1 + 5 * len(per.columns.levels[1])) + "|")
    for cat, r in per.iterrows():
        vals = [r[(k, m)] for k in ["image_auroc", "aupro", "dice_quantile", "dice_synthcal", "dice_oracle"]
                for m in per.columns.levels[1]]
        lines.append(f"| {cat} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    wins = (main.dice_synthcal >= main.dice_oracle).groupby(main.run).sum()
    beats_q = (main.dice_synthcal > main.dice_quantile).groupby(main.run).sum()
    n = main.groupby("run").size()
    summary = "\n".join(f"- {run}: SynthCal beats the quantile baseline on {beats_q[run]}/{n[run]} categories and "
                        f"matches or beats the oracle on {wins[run]}/{n[run]}." for run in n.index)
    return "\n".join(rows) + "\n\n" + summary + "\n\n<details><summary>Per-category results</summary>\n\n" + \
        "\n".join(lines) + "\n\n</details>\n"


def ablation_table(abl: pd.DataFrame, main: pd.DataFrame) -> str:
    cats = sorted(abl.category.unique())
    base = main[(main.run == "patchcore") & main.category.isin(cats)].assign(run="baseline (σ=4, 10%, layer2+3)")
    df = pd.concat([base, abl[abl.run != "layers123"]])
    df = df[df.category.isin(cats)]
    agg = df.groupby("run", sort=False).mean(numeric_only=True)
    rows = ["| Variant | Image AUROC | AUPRO | Dice: quantile | **Dice: SynthCal** | Dice: oracle | Fit time "
            "| Latency |", "|---|---|---|---|---|---|---|---|"]
    for run, r in agg.iterrows():
        rows.append(f"| {run} | {r.image_auroc:.3f} | {r.aupro:.3f} | {r.dice_quantile:.3f} | "
                    f"**{r.dice_synthcal:.3f}** | {r.dice_oracle:.3f} | {r.fit_seconds:.0f} s | "
                    f"{r.latency_ms:.0f} ms |")
    return f"Averaged over {', '.join(cats)} (PatchCore).\n\n" + "\n".join(rows) + "\n"


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    main = pd.read_csv(sys.argv[1])
    dice_chart(main)
    text = "## Main results\n\n" + main_tables(main)
    if len(sys.argv) > 2:
        text += "\n## Ablations\n\n" + ablation_table(pd.read_csv(sys.argv[2]), main)
    (OUT / "results.md").write_text(text)
    print(text)
