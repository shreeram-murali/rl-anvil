"""
Overlay hand-picked configurations, for comparisons plot_greedy.py does not lay out:
the SUN score ablation and the task-learner stability tests.

Reads <folder>/<config_id>/{cfg.yaml, <seed>/data.npz} like plot_greedy.py and
reuses its style and curve helpers (mean of per-seed moving averages, +-1 SE band).

    python plot_compare.py -f data_greedy_pilot --figure ablation
        MountainCar: test return and state entropy of SUN's score variants
        (ratio V/n, novelty 1/n, reachability V) against AdaGoal, DISCOVER,
        fast-eps DQN and random. -> plots/compare_ablation.{png,pdf}

    python plot_compare.py -f data_learner --figure learner
        CartPole and TravelField-28x28: test return of SUN and of fast-eps DQN for
        each target-update rate (tau) x learning-rate schedule. Rows share the y axis.
        -> plots/compare_learner.{png,pdf}

    python plot_compare.py -f data_explore_hard --figure wall
        Wall-50x50: test return and state coverage of every method (SUN sticky p=0.002
        and both DQNs at 1M steps; plain SUN and DISCOVER with / without the knob at 800k)
        -> plots/compare_wall; and SUN sticky p=0.002 under the learner variants
        (500k / 1M, update every 2 steps, sequence 32 + lambda 0.99)
        -> plots/compare_wall_ablation.
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D

import plot_greedy as pg

# Reference palette, categorical slots 1-7 in validated order
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
NEUTRAL = dict(color="#8a8984", ls=(0, (4, 2)), lw=1.4)


def entries(folder):
    for cfg_path in sorted(glob.glob(os.path.join(folder, "*", "cfg.yaml"))):
        cfg = yaml.safe_load(open(cfg_path))
        files = sorted(glob.glob(os.path.join(os.path.dirname(cfg_path), "*", "data.npz")))
        if not files:
            continue
        data = [np.load(f, allow_pickle=True) for f in files]
        n = min(len(d["steps"]) for d in data)
        yield cfg, dict(data=data, n=n, steps=data[0]["steps"][:n] / 1000)


def actor_label(cfg):
    a, c = cfg["agent"]["actor"], cfg["agent"]["critic"]
    if a["id"] == "EpsilonGreedy":
        eps = a["eps"]
        if eps["id"] == "Constant":
            return "random" if eps["value"] == 1.0 else None
        return "eps-greedy DQN, fast ε" if eps.get("steps") == cfg["experiment"]["warm_up"] else "eps-greedy DQN"
    if c.get("explore_only", True):
        return None
    g = a.get("greedy")
    if a["id"] == "SUN_Ratio":
        return "SUN: V/n (ratio)" if g is None else "SUN greedy"
    return {"SUN_Novelty": "SUN: 1/n (novelty only)", "SUN_Reachability": "SUN: V (reachability only)",
            "AdaGoal": "AdaGoal", "DISCOVER": "DISCOVER"}.get(a["id"])


def panel(ax, series, stat, smooth):
    for label, (entry, style) in series.items():
        pg.draw(ax, entry, stat, smooth, style.get("color"), lw=style.get("lw", 1.6),
                ls=style.get("ls", "-"), band=style.get("band", True), zorder=style.get("zorder", 3))


def fig_ablation(args):
    order = ["SUN: V/n (ratio)", "SUN: 1/n (novelty only)", "SUN: V (reachability only)",
             "AdaGoal", "DISCOVER"]
    refs = {"eps-greedy DQN, fast ε": dict(color="#8a8984", ls=(0, (6, 1.5, 1, 1.5)), lw=1.4),
            "random": NEUTRAL}  # gray references, as in plot_greedy.py
    series = {}
    for cfg, e in entries(args.folder):
        if pg.env_name(cfg) != "MountainCar":
            continue
        lab = actor_label(cfg)
        if lab in order:
            series[lab] = (e, dict(color=SLOTS[order.index(lab)]))
        elif lab in refs:
            series[lab] = (e, dict(refs[lab], band=False, zorder=2))
    keys = [k for k in order + list(refs) if k in series]
    series = {k: series[k] for k in keys}
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    for ax, stat, ylabel in zip(axes, ["test", args.entropy], ["test return", "state entropy"]):
        panel(ax, series, stat, args.smooth)
        ax.set_xlabel("steps (k)")
        ax.set_ylabel(ylabel)
    pg.threshold(axes[0], "MountainCar")
    axes[0].set_title("MountainCar: test return", color=pg.TEXT, fontweight="bold")
    axes[1].set_title("state entropy", color=pg.TEXT, fontweight="bold")
    handles = [Line2D([], [], color=s.get("color"), ls=s.get("ls", "-"), lw=2) for _, s in series.values()]
    fig.legend(handles, keys, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.1),
               handlelength=2.6)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    pg.save(fig, args.folder, "compare_ablation", args.formats)


def fig_learner(args):
    combos = [(0.001, "Constant"), (0.0002, "Constant"), (0.001, "LinearDecay"), (0.0002, "LinearDecay")]
    names = {(0.001, "Constant"): "τ=0.001, constant lr (default)",
             (0.0002, "Constant"): "τ=0.0002, constant lr",
             (0.001, "LinearDecay"): "τ=0.001, lr decay 1→0.1",
             (0.0002, "LinearDecay"): "τ=0.0002, lr decay 1→0.1"}
    envs, methods = ["CartPole", "TravelField-28x28"], ["SUN greedy", "eps-greedy DQN, fast ε"]
    grid = {}
    for cfg, e in entries(args.folder):
        env, lab = pg.env_name(cfg), actor_label(cfg)
        c = cfg["agent"]["critic"]
        key = (c["tau"], c["lr"]["id"])
        if env in envs and lab in methods and key in combos:
            grid[(env, lab, key)] = (e, cfg)
    fig, axes = plt.subplots(len(envs), 2, figsize=(9.0, 3.0 * len(envs)), sharey="row", squeeze=False)
    for i, env in enumerate(envs):
        for j, lab in enumerate(methods):
            ax = axes[i, j]
            series = {k: (grid[(env, lab, k)][0], dict(color=SLOTS[combos.index(k)]))
                      for k in combos if (env, lab, k) in grid}
            panel(ax, series, "test", args.smooth)
            pg.threshold(ax, env)
            some = next((grid[(env, lab, k)][1] for k in combos if (env, lab, k) in grid), None)
            title = lab if lab != "SUN greedy" or some is None else \
                f"SUN, sticky $p={some['agent']['actor']['greedy']['value']:g}$"
            ax.set_title(f"{env}: {title}", color=pg.TEXT, fontweight="bold")
            if j == 0:
                ax.set_ylabel("test return")
            if i == len(envs) - 1:
                ax.set_xlabel("steps (k)")
    handles = [Line2D([], [], color=SLOTS[i], lw=2) for i in range(len(combos))]
    fig.legend(handles, [names[k] for k in combos], loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    pg.save(fig, args.folder, "compare_learner", args.formats)


def wall_panels(series, args, name, legend_cols):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3))
    for ax, stat, ylabel in zip(axes, ["test", "s_%"], ["test return", "states visited (%)"]):
        for entry, style in series.values():
            if stat == "s_%":  # fraction -> percent
                entry = dict(entry, data=[{"s_%": d["s_%"] * 100} for d in entry["data"]])
            pg.draw(ax, entry, stat, args.smooth, style["color"], lw=style.get("lw", 1.6),
                    ls=style.get("ls", "-"), band=style.get("band", True), zorder=style.get("zorder", 3))
        ax.set_xlabel("steps (k)")
        ax.set_ylabel(ylabel)
    ax = axes[0]
    ax.axhline(1.0, color=pg.TEXT_2, lw=0.8, ls=(0, (2, 2)), zorder=1)
    ax.annotate("goal (+1)", xy=(0, 1.0), xycoords=("axes fraction", "data"), xytext=(3, -10),
                textcoords="offset points", fontsize=8, color=pg.TEXT_2)
    ax.axhline(0.1, color=pg.TEXT_2, lw=0.6, ls=(0, (1, 2)), zorder=1)
    ax.annotate("small reward (+0.1)", xy=(0, 0.1), xycoords=("axes fraction", "data"), xytext=(3, 3),
                textcoords="offset points", fontsize=8, color=pg.TEXT_2)
    ax.set_ylim(-0.03, 1.08)
    axes[0].set_title("Wall-50x50: test return", color=pg.TEXT, fontweight="bold")
    axes[1].set_title("state coverage", color=pg.TEXT, fontweight="bold")
    handles = [Line2D([], [], color=s["color"], ls=s.get("ls", "-"), lw=2) for _, s in series.values()]
    fig.legend(handles, list(series), loc="lower center", ncol=legend_cols, frameon=False,
               bbox_to_anchor=(0.5, -0.1), handlelength=2.6)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    pg.save(fig, args.folder, name, args.formats)


def fig_wall(args):
    env = "Wall-50x50"
    pick = lambda R, method, label=None: next((e for e in R.values() if e["method"] == method
                                               and (label is None or e["label"] == label)), None)
    n = lambda e: f" ({len(e['data'])} seeds)"

    def discover(greedy):
        """DISCOVER at 800k, with (sticky p) or without the knob; plot_greedy.describe skips the former."""
        for cfg, e in entries(args.folder):
            a, x = cfg["agent"]["actor"], cfg["experiment"]
            if (pg.env_name(cfg) == env and a["id"] == "DISCOVER" and x["training_steps"] == 800_000
                    and not cfg["agent"]["critic"].get("explore_only", True)
                    and all(pg.get_key(cfg, k) == v for k, v in pg.BASE_LEARNER.items())
                    and (a.get("greedy") or {}).get("value") == greedy):
                return e

    # colour = method, line style = knob (solid: sticky p=0.002, dashed: none)
    long, mid = (pg.load(args.folder, [env], training_steps=t)[env] for t in (1_000_000, 800_000))
    sun, dsc, dashed = pg.METHOD_COLORS["SUN + greedy"], pg.METHOD_COLORS["DISCOVER"], (0, (4, 2))
    rows = [("SUN, sticky $p=0.002$, 1M", pick(long, "SUN + greedy", "$p=0.002$"), dict(color=sun, zorder=5)),
            ("SUN, $p=0$, 800k", pick(mid, "SUN (p=0)"), dict(color=sun, ls=dashed, zorder=4)),
            ("DISCOVER, sticky $p=0.002$, 800k", discover(0.002), dict(color=dsc)),
            ("DISCOVER, 800k", discover(None), dict(color=dsc, ls=dashed)),
            ("eps-greedy DQN, 1M", pick(long, "eps-greedy DQN"), dict(color=pg.METHOD_COLORS["eps-greedy DQN"])),
            ("eps-greedy DQN, fast ε, 1M", pick(long, "eps-greedy DQN, fast ε"),
             dict(color=pg.METHOD_COLORS["eps-greedy DQN, fast ε"]))]
    series = {lab + n(e): (e, st) for lab, e, st in rows if e is not None}
    wall_panels(series, args, "compare_wall", 3)

    def sun(steps, **where):
        R = pg.load(args.folder, [env], training_steps=steps, where=where).get(env, {})
        return pick(R, "SUN + greedy", "$p=0.002$")
    variants = [("1M steps", sun(1_000_000)),
                ("1M steps, update every 2 steps", sun(1_000_000, **{"experiment.update_frequency": 2})),
                ("500k steps", sun(500_000)),
                ("500k steps, sequence 32, $\\lambda=0.99$",
                 sun(500_000, **{"agent.critic.sequence_length": 32, "agent.critic.lmbda": 0.99}))]
    series = {lab + n(e): (e, dict(color=SLOTS[i])) for i, (lab, e) in enumerate(variants) if e is not None}
    wall_panels(series, args, "compare_wall_ablation", 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-f", "--folder", required=True)
    parser.add_argument("--figure", required=True, choices=["ablation", "learner", "wall"])
    parser.add_argument("--entropy", default="s_h")
    parser.add_argument("--smooth", type=int, default=5)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    args = parser.parse_args()
    {"ablation": fig_ablation, "learner": fig_learner, "wall": fig_wall}[args.figure](args)
