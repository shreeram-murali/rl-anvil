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

    python plot_compare.py -f . --figure entropy_rule
        Offline test of a "go greedy when exploration saturates" rule on plain SUN
        (p=0) runs: per seed, where the rule "growth of the signal over the last W
        checkpoints < r x its largest growth so far" fires, for the signal = state
        entropy and = state coverage, against the test return of plain SUN and of
        the best greedy SUN. Reads data_greedy_pilot and data_explore_hard under -f.
        -> plots/compare_entropy_rule.{png,pdf}

    python plot_compare.py -f . --figure adaptive
        Adaptive greediness (greedy_adaptive.yaml, wall_adaptive.yaml): test return and
        share of greedy steps of the per-episode bandit and of reverse sticky, against
        plain SUN and the best hand-tuned greedy SUN; third column: the bandit and the
        best greedy SUN against DISCOVER (with the knob too, on Wall) and both
        eps-greedy DQNs. Prints the
        final test return and solved seeds of each. Reads data_greedy_pilot and
        data_explore_hard under -f. -> plots/compare_adaptive.{png,pdf}
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


RULE_ENVS = [  # env, folder, plain SUN run length, greedy SUN run length (None = any)
    ("MountainCar", "data_greedy_pilot", None, None),
    ("CartPole", "data_greedy_pilot", None, None),
    ("LunarLander", "data_greedy_pilot", None, None),
    ("Wall-50x50", "data_explore_hard", 800_000, 1_000_000),
    ("TravelField-28x28", "data_explore_hard", None, None),
]
GOAL = {"Wall-50x50": 1.0}  # "solved" for envs without a pg.SOLVED threshold


def rule_fires(steps, signal, window, ratio):
    """First step where signal[t] - signal[t - window] < ratio x the largest such growth so far."""
    growth = np.full(len(signal), np.nan)
    growth[window:] = signal[window:] - signal[:-window]
    best = np.fmax.accumulate(np.nan_to_num(growth, nan=-np.inf))
    hits = [t for t in range(window + 1, len(signal)) if growth[t] < ratio * best[t]]
    return steps[hits[0]] if hits else None


def fig_entropy_rule(args):
    rules = [("s_h", "state entropy", args.entropy_rule, SLOTS[1]),
             ("s_%", "state coverage", args.coverage_rule, SLOTS[2])]
    fig, axes = plt.subplots(len(RULE_ENVS), 3, figsize=(11.0, 2.15 * len(RULE_ENVS)), squeeze=False)
    for row, (env, folder, t_plain, t_greedy) in enumerate(RULE_ENVS):
        folder = os.path.join(args.folder, folder)
        plain = next(e for e in pg.load(folder, [env], t_plain)[env].values() if e["method"] == "SUN (p=0)")
        greedy = [e for e in pg.load(folder, [env], t_greedy)[env].values()
                  if e["method"] == "SUN + greedy" and len(e["data"]) >= 3]
        best = max(greedy, key=pg.final) if greedy else None
        x, n = plain["steps"], plain["n"]
        fires = {}
        for col, (stat, name, (window, ratio), color) in enumerate(rules):
            ax = axes[row, col]
            fires[stat] = []
            for d in plain["data"]:
                y = d[stat][:n].astype(float) * (100 if stat == "s_%" else 1)
                ax.plot(x, y, color="#8a8984", lw=0.8, alpha=0.7)
                f = rule_fires(x, y, window, ratio)
                fires[stat].append(f)
                if f is not None:
                    ax.plot(f, y[np.searchsorted(x, f)], "o", ms=4.5, color=color, mec="white", mew=0.6, zorder=4)
            ax.set_ylabel(f"{name}" + (" (%)" if stat == "s_%" else ""))
            if row == 0:
                ax.set_title(f"{name} rule ($W={window}$, $r={ratio:g}$)", color=pg.TEXT,
                             fontweight="bold", fontsize=9)
        ax = axes[row, 2]
        for d in plain["data"]:
            ax.plot(x, d["test"][:n], color="#8a8984", lw=0.6, alpha=0.5)
        pg.draw(ax, plain, "test", args.smooth, "#52514e", lw=1.6, band=False, zorder=3)
        if best is not None:
            pg.draw(ax, best, "test", args.smooth, SLOTS[0], lw=1.6, zorder=4)
        if env in GOAL:
            ax.axhline(GOAL[env], color=pg.TEXT_2, lw=0.8, ls=(0, (2, 2)), zorder=1)
        else:
            pg.threshold(ax, env)
        lo, hi = ax.get_ylim()
        for k, (stat, _, _, color) in enumerate(rules):  # rug of firing steps under the curves
            xs = [f for f in fires[stat] if f is not None]
            ax.plot(xs, [lo + (0.04 + 0.07 * k) * (hi - lo)] * len(xs), "|", ms=9, mew=1.6, color=color)
        ax.set_ylim(lo, hi)
        ax.set_ylabel("test return")
        if row == 0:
            ax.set_title("test return (ticks: where each rule fired)", color=pg.TEXT, fontweight="bold",
                         fontsize=9)
        for a in axes[row]:
            a.set_xlim(0, x[-1] * 1.01)
        axes[row, 0].annotate(env, xy=(0, 0.5), xycoords="axes fraction", xytext=(-52, 0),
                              textcoords="offset points", rotation=90, ha="right", va="center",
                              fontsize=10, fontweight="bold", color=pg.TEXT)
        best_name = "none" if best is None else f"{best['mode']} {best['label']}"
        print(f"{env}: fires (k) entropy {[None if f is None else int(f) for f in fires['s_h']]}, "
              f"coverage {[None if f is None else int(f) for f in fires['s_%']]}; best greedy: {best_name}")
    for a in axes[-1]:
        a.set_xlabel("steps (k)")
    handles = [Line2D([], [], color="#8a8984", lw=1), Line2D([], [], color="#52514e", lw=2),
               Line2D([], [], color=SLOTS[0], lw=2),
               Line2D([], [], marker="o", ls="", color=SLOTS[1], mec="white"),
               Line2D([], [], marker="o", ls="", color=SLOTS[2], mec="white")]
    fig.legend(handles, ["plain SUN ($p=0$), each seed", "plain SUN, mean", "best greedy SUN, mean",
                         "entropy rule fired", "coverage rule fired"],
               loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    pg.save(fig, args.folder, "compare_entropy_rule", args.formats)


ADAPTIVE_ENVS = [  # env, folder, run length of the new runs and of plain SUN, of the best greedy SUN
    ("MountainCar", "data_greedy_pilot", None, None),
    ("CartPole", "data_greedy_pilot", None, None),
    ("LunarLander", "data_greedy_pilot", None, None),
    ("Wall-50x50", "data_explore_hard", 800_000, 1_000_000),
]


def adaptive_runs(folder, env, steps):
    """{label: entry} of the bandit / reverse-sticky SUN runs (plot_greedy.describe skips them)."""
    out = {}
    for cfg, e in entries(folder):
        a = cfg["agent"]["actor"]
        mode = a.get("greedy_mode")
        if (pg.env_name(cfg) != env or a["id"] != "SUN_Ratio" or mode not in ("bandit", "reverse_sticky")
                or (steps is not None and cfg["experiment"]["training_steps"] != steps)
                or any(pg.get_key(cfg, k) != v for k, v in pg.BASE_LEARNER.items())):
            continue
        if mode == "bandit":
            label = "bandit, explore first" if a.get("bandit_flat") == "min" else "bandit"
        else:
            label = f"reverse sticky $p={a['greedy']['value']:g}$"
        e["label"], e["mode"] = label, mode
        out[label] = e
    return dict(sorted(out.items(), key=lambda kv: (kv[1]["mode"] != "bandit", kv[0])))


def solved(entry, env, last=5):
    """Seeds whose mean test return over the last checkpoints reaches the threshold
    (Wall-50x50: 0.5, i.e. the policy reaches the +1 goal most of the time)."""
    goal = {"Wall-50x50": 0.5}.get(env, pg.SOLVED.get(env))
    if goal is None:
        return None
    return sum(np.nanmean(d["test"][: entry["n"]][-last:]) >= goal for d in entry["data"])


def discover_knob(folder, env, steps):
    """DISCOVER + sticky greediness knob (plot_greedy.describe skips it), or None."""
    for cfg, e in entries(folder):
        a = cfg["agent"]["actor"]
        if (pg.env_name(cfg) == env and a["id"] == "DISCOVER" and (a.get("greedy") or {}).get("value")
                and (steps is None or cfg["experiment"]["training_steps"] == steps)
                and not cfg["agent"]["critic"].get("explore_only", True)
                and all(pg.get_key(cfg, k) == v for k, v in pg.BASE_LEARNER.items())):
            return e


def fig_adaptive(args):
    reverse_colors = ["#9fd9c2", "#1baf7a", "#0b6b48"]  # light -> dark = longer greedy start
    base_colors = {"DISCOVER": SLOTS[6], "eps-greedy DQN": SLOTS[3], "eps-greedy DQN, fast ε": SLOTS[4]}
    titles = ["SUN variants: test return", "SUN variants: greedy steps (cumulative)", "vs baselines: test return"]
    fig, axes = plt.subplots(len(ADAPTIVE_ENVS), 3, figsize=(13.0, 2.4 * len(ADAPTIVE_ENVS)), squeeze=False)
    legend = {}
    for row, (env, folder, t_new, t_best) in enumerate(ADAPTIVE_ENVS):
        folder = os.path.join(args.folder, folder)
        R = pg.load(folder, [env], t_new).get(env, {})
        R_best = pg.load(folder, [env], t_best).get(env, {})
        greedy = [e for e in R_best.values() if e["method"] == "SUN + greedy" and len(e["data"]) >= 3]
        best = max(greedy, key=pg.final) if greedy else None
        sun = {}  # columns 1-2
        plain = next((e for e in R.values() if e["method"] == "SUN (p=0)"), None)
        if plain is not None:
            sun["plain SUN ($p=0$)"] = (plain, dict(color="#52514e", ls=(0, (4, 2)), band=False))
        if best is not None:
            sun["best hand-tuned greedy SUN"] = (best, dict(color=SLOTS[0], zorder=4))
        rev = 0
        for label, e in adaptive_runs(folder, env, t_new).items():
            if label == "bandit, explore first":
                sun[label] = (e, dict(color="#9c3a0c", lw=2.0, zorder=6))  # dark orange: bandit variant
            elif e["mode"] == "bandit":
                sun[label] = (e, dict(color=SLOTS[1], lw=2.0, zorder=5))
            else:
                sun[label] = (e, dict(color=reverse_colors[min(rev, 2)]))
                rev += 1
        base = {}  # column 3: the bandit and the best SUN against the other methods
        for k in ("bandit", "bandit, explore first", "best hand-tuned greedy SUN"):
            if k in sun:
                base[k] = sun[k]
        for method, color in base_colors.items():
            # DQNs on Wall-50x50 only exist at 1M steps
            e = next((e for e in R.values() if e["method"] == method), None) \
                or next((e for e in R_best.values() if e["method"] == method), None)
            if e is not None:
                base[method] = (e, dict(color=color, band=method == "DISCOVER", zorder=3))
        knob = discover_knob(folder, env, t_new)
        if knob is not None:
            base["DISCOVER + sticky $p=0.002$"] = (knob, dict(color=SLOTS[6], ls=(0, (4, 2)), band=False))
        for col, (series, stat) in enumerate([(sun, "test"), (sun, "greedy_steps_%"), (base, "test")]):
            ax = axes[row, col]
            for label, (e, st) in series.items():
                if stat == "greedy_steps_%":
                    if stat not in e["data"][0]:
                        continue
                    e = dict(e, data=[{stat: d[stat] * 100} for d in e["data"]])  # logged as a fraction
                pg.draw(ax, e, stat, args.smooth, st.get("color"), lw=st.get("lw", 1.6), ls=st.get("ls", "-"),
                        band=st.get("band", True), zorder=st.get("zorder", 3))
                legend.setdefault(label, st)
            ax.set_ylabel("greedy steps (%)" if stat == "greedy_steps_%" else "test return")
            if stat == "test":
                if env in GOAL:
                    ax.axhline(GOAL[env], color=pg.TEXT_2, lw=0.8, ls=(0, (2, 2)), zorder=1)
                else:
                    pg.threshold(ax, env)
            if row == 0:
                ax.set_title(titles[col], color=pg.TEXT, fontweight="bold", fontsize=9)
        lo = min(axes[row, c].get_ylim()[0] for c in (0, 2))
        hi = max(axes[row, c].get_ylim()[1] for c in (0, 2))
        for c in (0, 2):
            axes[row, c].set_ylim(lo, hi)
        axes[row, 0].annotate(env, xy=(0, 0.5), xycoords="axes fraction", xytext=(-52, 0),
                              textcoords="offset points", rotation=90, ha="right", va="center",
                              fontsize=10, fontweight="bold", color=pg.TEXT)
        print(f"\n{env}" + ("" if best is None else f"  (best hand-tuned: {best['mode']} {best['label']})"))
        for label, (e, _) in {**sun, **base}.items():
            finals = [np.nanmean(d["test"][: e["n"]][-5:]) for d in e["data"]]
            se = np.std(finals, ddof=1) / np.sqrt(len(finals)) if len(finals) > 1 else np.nan
            s = solved(e, env)
            gs = np.mean([d["greedy_steps_%"][e["n"] - 1] for d in e["data"]]) * 100 \
                if "greedy_steps_%" in e["data"][0] else np.nan
            extra = ""
            if e.get("mode") == "bandit" and "greedy_arm" in e["data"][0]:
                extra = f", final most-picked arm per seed {[float(d['greedy_arm'][e['n'] - 1]) for d in e['data']]}"
            print(f"  {label.replace('$', ''):32s} n={len(finals):2d}  final test {np.mean(finals):8.2f} "
                  f"+- {se:6.2f}  solved {'-' if s is None else f'{s}/{len(finals)}'}  "
                  f"greedy steps {'    -' if np.isnan(gs) else f'{gs:5.1f}'}%  @{int(e['steps'][-1])}k{extra}")
    for a in axes[-1]:
        a.set_xlabel("steps (k)")
    handles = [Line2D([], [], color=st.get("color"), ls=st.get("ls", "-"), lw=2) for st in legend.values()]
    fig.legend(handles, list(legend), loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01),
               handlelength=2.6)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    pg.save(fig, args.folder, "compare_adaptive", args.formats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-f", "--folder", required=True)
    parser.add_argument("--figure", required=True, choices=["ablation", "learner", "wall", "entropy_rule", "adaptive"])
    parser.add_argument("--entropy", default="s_h")
    parser.add_argument("--smooth", type=int, default=5)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--entropy_rule", nargs=2, type=float, default=[5, 0.1], metavar=("W", "R"),
                        help="entropy_rule: window (checkpoints) and growth ratio for the entropy signal")
    parser.add_argument("--coverage_rule", nargs=2, type=float, default=[10, 0.05], metavar=("W", "R"),
                        help="entropy_rule: window (checkpoints) and growth ratio for the coverage signal")
    args = parser.parse_args()
    args.entropy_rule[0], args.coverage_rule[0] = int(args.entropy_rule[0]), int(args.coverage_rule[0])
    {"ablation": fig_ablation, "learner": fig_learner, "wall": fig_wall, "entropy_rule": fig_entropy_rule,
     "adaptive": fig_adaptive}[args.figure](args)
