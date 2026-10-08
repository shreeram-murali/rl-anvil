"""
Reward and exploration curves for the SUN greediness study (gcrl_taskexps branch).

Reads <folder>/<config_id>/{cfg.yaml, <seed>/data.npz} directly (no results.gzip
needed) and writes three figures to <folder>/plots/:

    greedy_methods.{png,pdf}
        Columns = environments. Rows = greedy test return, state entropy.
        Overlaid: eps-greedy DQN, DISCOVER, SUN (p=0), and SUN with the best
        greediness setting of that environment (highest final test return,
        named in the panel corner).

    greedy_p_test.{png,pdf}  and  greedy_p_entropy.{png,pdf}
        Rows = environments. Columns = how p is applied (per-step, sticky,
        scheduled per-step, scheduled sticky). Overlaid: every value of p in that
        mode, light -> dark as the agent gets greedier, plus eps-greedy DQN and
        SUN p=0 as gray references in every panel.

Y axes are shared wherever the quantity is the same: the return panels of one
environment share their scale, and every entropy panel shares one scale.
Curves are the mean over seeds of a per-seed moving average, with a band of
+-1 standard error; the dashed line on return panels is the "solved" threshold.

Example:

    python plot_greedy.py -f data_greedy_pilot
    python plot_greedy.py -f data_greedy_pilot --envs MountainCar LunarLander --smooth 9
"""

import argparse
import glob
import os
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D


# ------------------------------------------------------------------------------
# Style. Colors are the reference data-viz palette in its validated slot order;
# the ordinal ramp starts no lighter than step 250 so every curve clears 2:1 on
# white. Text never wears a series color.
# ------------------------------------------------------------------------------

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
METHOD_COLORS = {  # categorical slots 1-4, fixed order
    "SUN + greedy": "#2a78d6",
    "SUN (p=0)": "#eb6834",
    "DISCOVER": "#1baf7a",
    "eps-greedy DQN": "#eda100",
}
RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
        "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
REF_STYLE = {  # references in the p-sweep panels: neutral, told apart by dash
    "eps-greedy DQN": dict(color="#8a8984", ls=(0, (4, 2)), lw=1.4),
    "SUN (p=0)": dict(color="#3d3c39", ls=(0, (1, 1.5)), lw=1.6),
}

# Gymnasium's reward_threshold; Pendulum has none, -200 is our choice.
SOLVED = {"MountainCar": -110, "CartPole": 475, "Acrobot": -100,
          "LunarLander": 200, "Pendulum": -200}

MODES = [  # (key, column title)
    ("step", "per-step $p$"),
    ("sticky", "sticky $p$"),
    ("step-sched", "scheduled $p$ (per-step)"),
    ("sticky-sched", "scheduled $p$ (sticky)"),
]

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8.5,
    "axes.edgecolor": TEXT_2,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT_2,
    "ytick.color": TEXT_2,
    "text.color": TEXT,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
})


# ------------------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------------------

def env_name(cfg):
    return cfg["environment"]["id"].split("/")[-1].split("-v")[0]


def describe(cfg):
    """(method, mode, p, sort_key, label) for one configuration, or None to skip.

    method is one of the METHOD_COLORS families ("SUN + greedy" for any SUN with
    p > 0); mode/p/label describe the greediness knob for SUN + greedy runs.
    """
    actor = cfg["agent"]["actor"]
    critic = cfg["agent"]["critic"]
    aid = actor["id"]
    eps = actor.get("eps", {})

    if aid == "EpsilonGreedy":
        if eps.get("id") == "LinearDecay":
            return ("eps-greedy DQN", None, None, None, "eps-greedy DQN")
        return None  # random / greedy actors
    if aid == "DISCOVER":
        return ("DISCOVER", None, None, None, "DISCOVER") if not critic.get("explore_only", True) else None
    if aid != "SUN_Ratio" or critic.get("explore_only", True):
        return None  # pure explorers have no task policy to test

    if eps.get("end_value") != 0.1:
        return None  # other eps floors are a different knob; not plotted

    g = actor.get("greedy")
    mode = actor.get("greedy_mode", "step")
    if g is None or (g["id"] == "Constant" and g["value"] == 0):
        return ("SUN (p=0)", None, 0.0, None, "SUN (p=0)")
    if g["id"] == "Constant":
        p = g["value"]
        return ("SUN + greedy", mode, p, p, f"$p={p:g}$")
    if g["id"] == "LinearDecay":
        steps = cfg["experiment"]["training_steps"]
        full_at = (steps - g.get("constant_end_steps", 0)) // 1000
        p = g["end_value"]
        # greedier = higher final p, reached sooner
        key = (p, -full_at)
        return ("SUN + greedy", f"{mode}-sched", p, key, f"$p\\to{p:g}$ by {full_at}k")
    return None


def load(folder, envs):
    """runs[env][label] = dict(method, mode, p, key, steps, stats={name: (seeds, T)})."""
    runs = defaultdict(dict)
    for cfg_path in sorted(glob.glob(os.path.join(folder, "*", "cfg.yaml"))):
        cfg = yaml.safe_load(open(cfg_path))
        env = env_name(cfg)
        if envs and env not in envs:
            continue
        info = describe(cfg)
        if info is None:
            continue
        method, mode, p, key, label = info
        files = sorted(glob.glob(os.path.join(os.path.dirname(cfg_path), "*", "data.npz")))
        if not files:
            continue
        data = [np.load(f, allow_pickle=True) for f in files]
        n = min(len(d["steps"]) for d in data)
        entry = dict(method=method, mode=mode, p=p, key=key, label=label,
                     steps=data[0]["steps"][:n] / 1000,
                     data=data, n=n, cfg_dir=os.path.dirname(cfg_path))
        uid = (method, mode, label)
        if uid in runs[env]:
            old = runs[env][uid]
            warnings.warn(f"{env}: '{label}' matches two configurations "
                          f"({old['cfg_dir']}, {entry['cfg_dir']}); keeping the one with more seeds.")
            if len(old["data"]) >= len(data):
                continue
        runs[env][uid] = entry
    return runs


def curve(entry, stat, smooth):
    """Mean and standard error over seeds of the per-seed moving average."""
    if stat not in entry["data"][0]:
        return None
    Y = np.stack([d[stat][: entry["n"]].astype(float) for d in entry["data"]])
    if smooth > 1:
        k = np.ones(smooth)
        num = np.apply_along_axis(lambda y: np.convolve(np.nan_to_num(y), k, "same"), 1, Y)
        den = np.apply_along_axis(lambda y: np.convolve(~np.isnan(y), k, "same"), 1, Y)
        Y = num / np.maximum(den, 1)
    mean = np.nanmean(Y, 0)
    se = np.nanstd(Y, 0, ddof=1) / np.sqrt(len(Y)) if len(Y) > 1 else np.zeros_like(mean)
    return mean, se


def final(entry, stat="test", last=5):
    return np.mean([np.nanmean(d[stat][: entry["n"]][-last:]) for d in entry["data"]])


# ------------------------------------------------------------------------------
# Drawing
# ------------------------------------------------------------------------------

def draw(ax, entry, stat, smooth, color, lw=1.6, ls="-", band=True, zorder=2):
    c = curve(entry, stat, smooth)
    if c is None:
        return
    m, se = c
    x = entry["steps"]
    if band:
        ax.fill_between(x, m - se, m + se, color=color, alpha=0.15, lw=0, zorder=zorder - 1)
    ax.plot(x, m, color=color, lw=lw, ls=ls, zorder=zorder, solid_capstyle="round")


def threshold(ax, env):
    if env in SOLVED:
        ax.axhline(SOLVED[env], color=TEXT_2, lw=0.8, ls=(0, (2, 2)), zorder=1)
        ax.annotate("solved", xy=(0, SOLVED[env]), xycoords=("axes fraction", "data"),
                    xytext=(3, 2), textcoords="offset points", fontsize=7, color=TEXT_2)


def label_axes(ax, ylabel=None, xlabel=None):
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)


def save(fig, folder, name, formats):
    os.makedirs(os.path.join(folder, "plots"), exist_ok=True)
    for fmt in formats:
        path = os.path.join(folder, "plots", f"{name}.{fmt}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print("wrote", path)
    plt.close(fig)


def fig_methods(runs, envs, args):
    fig, axes = plt.subplots(2, len(envs), figsize=(3.3 * len(envs), 5.2),
                             squeeze=False, sharex="col")
    for j, env in enumerate(envs):
        R = runs[env]
        greedy = [e for e in R.values() if e["method"] == "SUN + greedy" and len(e["data"]) >= args.min_seeds]
        best = max(greedy, key=final) if greedy else None
        picks = {e["method"]: e for e in R.values() if e["method"] != "SUN + greedy"}
        if best is not None:
            picks["SUN + greedy"] = best
        for i, stat in enumerate(["test", args.entropy]):
            ax = axes[i, j]
            # draw back to front: baselines first, SUN last
            for method in reversed(list(METHOD_COLORS)):
                if method in picks:
                    draw(ax, picks[method], stat, args.smooth, METHOD_COLORS[method])
            if i == 0:
                threshold(ax, env)
                ax.set_title(env, color=TEXT, fontweight="bold", pad=16)
                if best is not None:
                    mode = {"step": "per-step", "sticky": "sticky",
                            "step-sched": "sched.", "sticky-sched": "sticky sched."}[best["mode"]]
                    ax.text(0.5, 1.02, f"SUN + greedy = {mode} {best['label']}",
                            transform=ax.transAxes, ha="center", va="bottom",
                            fontsize=7.5, color=TEXT_2)
            label_axes(ax, ylabel=("test return" if i == 0 else "state entropy") if j == 0 else None,
                       xlabel="steps (k)" if i == 1 else None)
    # entropy row: one shared scale
    lo = min(ax.get_ylim()[0] for ax in axes[1]); hi = max(ax.get_ylim()[1] for ax in axes[1])
    for ax in axes[1]:
        ax.set_ylim(lo, hi)
    handles = [Line2D([], [], color=c, lw=2) for c in METHOD_COLORS.values()]
    fig.legend(handles, list(METHOD_COLORS), loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, args.folder, "greedy_methods", args.formats)


def fig_p_sweep(runs, envs, stat, name, ylabel, args):
    modes = [(k, t) for k, t in MODES
             if any(e["mode"] == k for env in envs for e in runs[env].values())]
    fig, axes = plt.subplots(len(envs), len(modes), figsize=(3.0 * len(modes), 2.5 * len(envs) + 0.6),
                             squeeze=False, sharey="row" if stat == "test" else "all")
    for i, env in enumerate(envs):
        R = runs[env]
        refs = {e["method"]: e for e in R.values() if e["method"] in REF_STYLE}
        for j, (mode, title) in enumerate(modes):
            ax = axes[i, j]
            for method, style in REF_STYLE.items():
                if method in refs:
                    draw(ax, refs[method], stat, args.smooth, style["color"], lw=style["lw"],
                         ls=style["ls"], band=False, zorder=3)
            members = sorted((e for e in R.values() if e["mode"] == mode), key=lambda e: e["key"])
            # spread the members over the whole ramp (lightest -> darkest) so
            # even two values of p are easy to tell apart
            if len(members) <= 1:
                idx = np.full(len(members), len(RAMP) - 1)
            else:
                idx = np.linspace(0, len(RAMP) - 1, len(members)).round().astype(int)
            for e, k in zip(members, idx):
                draw(ax, e, stat, args.smooth, RAMP[k], zorder=4)
            if stat == "test":
                threshold(ax, env)
            if members:
                handles = [Line2D([], [], color=RAMP[k], lw=2) for k in idx]
                ax.legend(handles, [e["label"] for e in members], loc="best",
                          frameon=False, handlelength=1.4, fontsize=7, labelspacing=0.25)
            else:
                ax.text(0.5, 0.97, "not run", transform=ax.transAxes, ha="center",
                        va="top", color=TEXT_2, fontsize=8,
                        bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
            if i == 0:
                ax.set_title(title, color=TEXT, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{env}\n{ylabel}")
            if i == len(envs) - 1:
                ax.set_xlabel("steps (k)")
    handles = [Line2D([], [], **REF_STYLE[m]) for m in REF_STYLE]
    fig.legend(handles, list(REF_STYLE), loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.01), title="references (gray); lighter → darker blue = greedier",
               title_fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, args.folder, name, args.formats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-f", "--folder", required=True, help="Data directory, e.g. data_greedy_pilot.")
    parser.add_argument("--envs", nargs="+", default=["MountainCar", "CartPole", "LunarLander"],
                        help="Environment names (Gymnasium id without version), in column order.")
    parser.add_argument("--entropy", default="s_h", help="Entropy statistic: s_h (states) or sa_h.")
    parser.add_argument("--smooth", type=int, default=5, help="Moving-average window in checkpoints.")
    parser.add_argument("--min_seeds", type=int, default=3,
                        help="Seeds a SUN + greedy configuration needs to be picked as 'best'.")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    args = parser.parse_args()

    runs = load(args.folder, args.envs)
    envs = [e for e in args.envs if runs.get(e)]
    missing = sorted(set(args.envs) - set(envs))
    if missing:
        print("no data for:", missing)
    if not envs:
        raise SystemExit("Nothing to plot.")
    for env in envs:
        print(f"{env}: " + ", ".join(f"{e['label']}{'' if e['mode'] is None else ' [' + e['mode'] + ']'} (n={len(e['data'])})"
                                     for e in runs[env].values()))

    fig_methods(runs, envs, args)
    fig_p_sweep(runs, envs, "test", "greedy_p_test", "test return", args)
    fig_p_sweep(runs, envs, args.entropy, "greedy_p_entropy", "state entropy", args)
