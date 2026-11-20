"""
Draw the README chart from results/criteo_benchmark.json.

    python scripts/make_readme_chart.py
Writes docs/criteo_effects.svg.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e0"
REFERENCE = "#2a78d6"   # AIPW, the series the reader should follow
BIASED = "#a3a29c"      # raw difference, shown for contrast only


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/criteo_benchmark.json")
    ap.add_argument("--out", default="docs/criteo_effects.svg")
    args = ap.parse_args()
    sec = json.loads(Path(args.results).read_text())["sections"]
    eff, f8 = sec["full_data_effects"], sec["f8_groups"]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "path", "axes.edgecolor": GRID})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.9), facecolor=SURFACE,
                                   gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.35})

    # Left: relative lift, raw difference vs AIPW, per outcome
    outcomes = [("conversion", "Conversion"), ("visit", "Visit")]
    width, gap = 0.34, 0.03
    for i, (key, label) in enumerate(outcomes):
        e = eff[key]
        base = e["difference_in_means"]["control_rate"]
        for j, (name, color, est) in enumerate((
            ("Raw difference", BIASED, e["difference_in_means"]),
            ("AIPW (reference)", REFERENCE, e["aipw"]),
        )):
            x = i + (j - 0.5) * (width + gap)
            rel = est["estimate"] / base * 100
            lo, hi = est["ci"][0] / base * 100, est["ci"][1] / base * 100
            ax1.bar(x, rel, width, color=color, label=name if i == 0 else None, zorder=2)
            ax1.errorbar(x, rel, yerr=[[rel - lo], [hi - rel]], fmt="none",
                         ecolor=TEXT, elinewidth=1, capsize=3, zorder=3)
            ax1.text(x, hi + 1.5, f"+{rel:.1f}%", ha="center", va="bottom",
                     color=TEXT if j else MUTED, fontsize=9, fontweight="bold" if j else "normal")
    ax1.set_xticks(range(len(outcomes)), [lbl for _, lbl in outcomes])
    ax1.set_ylabel("Lift caused by ads (% of control rate)", color=MUTED)
    ax1.set_title("Raw comparison overstates the effect", loc="left", color=TEXT,
                  fontsize=11, fontweight="bold")
    ax1.legend(frameon=False, loc="upper right", fontsize=9)
    ax1.set_ylim(0, max(ax1.get_ylim()[1], 75))

    # Right: AIPW conversion effect per f8 group, relative to the average
    groups = f8["conversion"]
    total = sum(g["n"] for g in groups)
    labels = [f"f8 {g['value_from']:.3f}–{g['value_to']:.3f}\n{g['n'] / total:.0%} of users"
              for g in groups]
    vals = [g["aipw_lift_vs_overall"] for g in groups]
    bars = ax2.bar(range(len(groups)), vals, 0.5, color=REFERENCE, zorder=2)
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.2f}×", ha="center",
                 va="bottom", color=TEXT, fontsize=9, fontweight="bold")
    ax2.axhline(1, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=3)
    ax2.text(len(groups) - 1, 1.06, "average user", ha="center", va="bottom", color=MUTED, fontsize=8)
    ax2.set_xticks(range(len(groups)), labels)
    ax2.set_ylabel("Conversion effect vs average (×)", color=MUTED)
    ax2.set_title("A quarter of users drive the lift", loc="left", color=TEXT,
                  fontsize=11, fontweight="bold")
    ax2.set_ylim(0, max(vals) * 1.2)

    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
        ax.tick_params(colors=MUTED, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)

    fig.text(0.01, 0.01, f"Criteo Uplift v2.1, all {sec['data']['rows']:,} users. "
             "Bars show point estimates; whiskers are 95% CIs.", color=MUTED, fontsize=8)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.2)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor=SURFACE)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
