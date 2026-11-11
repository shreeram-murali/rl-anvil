"""
Slide-sized figures for the greediness study (16:10 beamer, 16 cm wide).

Every figure is drawn at the physical size it is shown at on the slide, so it is
included with \\includegraphics without scaling and its fonts stay readable. Data
and style come from plot_greedy.py (mean of per-seed moving averages, +-1 SE band).

    python plot_slides.py --out notes/figures/slides

writes
    return.pdf, entropy.pdf     classic control, one panel per environment
    p_<Env>.pdf                 test return for every value of p, one panel per mode
    wall_return.pdf, wall_entropy.pdf   Wall-50x50 at 500k steps
"""

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import plot_greedy as pg

warnings.filterwarnings("ignore")  # duplicate SUN p=0 configs (knob absent vs p=0) are equivalent

WIDTH = 5.85  # inches, about the text width of a 16:10 beamer frame
HEIGHT = 2.85
plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7.5,
    # white, not the off-white of plot_greedy.py, so figures don't show as boxes on the slide
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})
CLASSIC = ["MountainCar", "CartPole", "LunarLander"]


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"{name}.pdf")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print("wrote", path)


def best_greedy(R, min_seeds=3):
    greedy = [e for e in R.values() if e["method"] == "SUN + greedy" and len(e["data"]) >= min_seeds]
    return max(greedy, key=pg.final) if greedy else None


def mode_name(e):
    return {"step": "per-step", "sticky": "sticky", "step-sched": "sched.",
            "sticky-sched": "sticky sched."}[e["mode"]] + " " + e["label"]


def methods_row(runs, envs, stat, ylabel, out, name, smooth, shared=None):
    fig, axes = plt.subplots(1, len(envs), figsize=(WIDTH, HEIGHT), squeeze=False)
    axes = axes[0]
    for ax, env in zip(axes, envs):
        R = runs[env]
        picks = {e["method"]: e for e in R.values() if e["method"] != "SUN + greedy"}
        best = best_greedy(R)
        if best is not None:
            picks["SUN + greedy"] = best
        for method in reversed(list(pg.METHOD_COLORS)):
            if method in picks:
                pg.draw(ax, picks[method], stat, smooth, pg.METHOD_COLORS[method], lw=1.4)
        if stat == "test":
            pg.threshold(ax, env)
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi + 0.08 * (hi - lo))  # headroom so the "solved" label clears the title
        ax.set_title(env, color=pg.TEXT, fontweight="bold", pad=12 if stat == "test" else 6)
        if best is not None and stat == "test":
            ax.text(0.5, 1.01, f"SUN + greedy = {mode_name(best)}", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=6.5, color=pg.TEXT_2)
        ax.set_xlabel("steps (k)")
    axes[0].set_ylabel(ylabel)
    if shared is not None:
        for ax in axes:
            ax.set_ylim(*shared)
    handles = [Line2D([], [], color=c, lw=2) for c in pg.METHOD_COLORS.values()]
    fig.legend(handles, list(pg.METHOD_COLORS), loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.02), handlelength=1.6, columnspacing=0.9)
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=1.0)
    save(fig, out, name)


def p_row(runs, env, out, smooth):
    R = runs[env]
    short = {"step": "per-step $p$", "sticky": "sticky $p$", "step-sched": "scheduled\n(per-step)",
             "sticky-sched": "scheduled\n(sticky)"}
    modes = [(k, short[k]) for k, _ in pg.MODES if any(e["mode"] == k for e in R.values())]
    fig, axes = plt.subplots(1, len(modes), figsize=(WIDTH, HEIGHT), sharey=True, squeeze=False)
    axes = axes[0]
    refs = {e["method"]: e for e in R.values() if e["method"] in pg.REF_STYLE}
    for ax, (mode, title) in zip(axes, modes):
        for method, style in pg.REF_STYLE.items():
            if method in refs:
                pg.draw(ax, refs[method], "test", smooth, style["color"], lw=1.2,
                        ls=style["ls"], band=False, zorder=3)
        members = sorted((e for e in R.values() if e["mode"] == mode), key=lambda e: e["key"])
        idx = (np.full(len(members), len(pg.RAMP) - 1) if len(members) <= 1 else
               np.linspace(0, len(pg.RAMP) - 1, len(members)).round().astype(int))
        for e, k in zip(members, idx):
            pg.draw(ax, e, "test", smooth, pg.RAMP[k], lw=1.4, zorder=4)
        pg.threshold(ax, env)
        ax.legend([Line2D([], [], color=pg.RAMP[k], lw=2) for k in idx], [e["label"] for e in members],
                  loc="best", frameon=False, handlelength=1.2, fontsize=6.5, labelspacing=0.2)
        ax.set_title(title, color=pg.TEXT, fontweight="bold")
        ax.set_xlabel("steps (k)")
    axes[0].set_ylabel(f"{env}: test return")
    handles = [Line2D([], [], color=s["color"], ls=s["ls"], lw=1.4) for s in pg.REF_STYLE.values()]
    fig.legend(handles, list(pg.REF_STYLE), loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02), handlelength=2.4)
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=0.6)
    save(fig, out, f"p_{env}")


def wall(runs, out, smooth):
    R = runs["Wall-50x50"]
    w = WIDTH / 2 - 0.05
    # Left: test return of the greedy SUN settings; everything else is flat at 0
    fig, ax = plt.subplots(figsize=(w, 2.6))
    # sticky settings first (by p), then schedules; categorical slots 1-3
    members = sorted((e for e in R.values() if e["method"] == "SUN + greedy"),
                     key=lambda e: (e["mode"] != "sticky", str(e["key"])))
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    for e, c in zip(members, colors):
        pg.draw(ax, e, "test", smooth, c, lw=1.4, zorder=4)
    zero = next(e for e in R.values() if e["method"] == "SUN (p=0)")
    pg.draw(ax, zero, "test", smooth, "#8a8984", lw=1.4, ls=(0, (4, 2)), band=False, zorder=3)
    labels = [mode_name(e) for e in members] + ["DQN (both), DISCOVER,\nSUN $p=0$: all exactly 0"]
    handles = [Line2D([], [], color=c, lw=2) for c in colors[:len(members)]] + \
              [Line2D([], [], color="#8a8984", ls=(0, (4, 2)), lw=1.4)]
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(0, 0.93), frameon=False, fontsize=6.5,
              handlelength=1.6)
    ax.axhline(1.0, color=pg.TEXT_2, lw=0.8, ls=(0, (2, 2)))
    ax.annotate("goal (+1)", xy=(1, 1.0), xycoords=("axes fraction", "data"), xytext=(-2, -9),
                textcoords="offset points", ha="right", fontsize=6.5, color=pg.TEXT_2)
    ax.axhline(0.1, color=pg.TEXT_2, lw=0.6, ls=(0, (1, 2)))
    ax.annotate("small reward (+0.1)", xy=(0, 0.1), xycoords=("axes fraction", "data"), xytext=(3, 3),
                textcoords="offset points", ha="left", fontsize=6.5, color=pg.TEXT_2)
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("Wall-50x50: test return", color=pg.TEXT, fontweight="bold")
    ax.set_xlabel("steps (k)")
    ax.set_ylabel("test return")
    fig.tight_layout()
    save(fig, out, "wall_return")

    # Right: state entropy of every method
    fig, ax = plt.subplots(figsize=(w, 2.6))
    picks = {e["method"]: e for e in R.values() if e["method"] != "SUN + greedy"}
    best = best_greedy(R)
    if best is not None:
        picks["SUN + greedy"] = best
    for method in reversed(list(pg.METHOD_COLORS)):
        if method in picks:
            pg.draw(ax, picks[method], pg.ARGS_ENTROPY, smooth, pg.METHOD_COLORS[method], lw=1.4)
    names = [m if m != "SUN + greedy" else f"SUN + greedy ({mode_name(best)})" for m in pg.METHOD_COLORS]
    ax.legend([Line2D([], [], color=c, lw=2) for c in pg.METHOD_COLORS.values()], names,
              loc="lower right", frameon=False, fontsize=6.5, handlelength=1.6)
    ax.set_ylim(0.42, 0.99)  # room for the legend below the curves
    ax.set_title("Wall-50x50: state entropy", color=pg.TEXT, fontweight="bold")
    ax.set_xlabel("steps (k)")
    ax.set_ylabel("state entropy")
    fig.tight_layout()
    save(fig, out, "wall_entropy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classic", default="data_greedy_pilot")
    parser.add_argument("--wall", default="data_explore_hard")
    parser.add_argument("--out", default="notes/figures/slides")
    parser.add_argument("--smooth", type=int, default=5)
    args = parser.parse_args()
    pg.ARGS_ENTROPY = "s_h"

    runs = pg.load(args.classic, CLASSIC)
    methods_row(runs, CLASSIC, "test", "test return", args.out, "return", args.smooth)
    lo = min(np.nanmin(pg.curve(e, "s_h", args.smooth)[0]) for env in CLASSIC for e in runs[env].values())
    methods_row(runs, CLASSIC, "s_h", "state entropy", args.out, "entropy", args.smooth,
                shared=(np.floor(lo * 20) / 20, 0.95))
    for env in CLASSIC:
        p_row(runs, env, args.out, args.smooth)

    wall(pg.load(args.wall, ["Wall-50x50"], training_steps=500_000), args.out, args.smooth)
