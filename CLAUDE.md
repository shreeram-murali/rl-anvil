# CLAUDE.md

Research code (PyTorch, Hydra) for **SUN: Reaching for Novelty in RL** (Aalto). The paper
is in [PAPER.md](PAPER.md); [README.md](README.md) is the user guide for running,
sweeping, and plotting. This file is what a coding session needs on top of those.

> Notes from an older, **JAX** sibling repo (`q_visit_reach`, `greedy_switch_prob`,
> `submitit_triton.py`, `plot_sweep.py`) do NOT apply here. None of those exist.

## Research goal (branch `gcrl_taskexps`)

The paper evaluates SUN as a *pure explorer* (coverage/entropy). This branch asks whether
SUN's exploration makes agents **better at the extrinsic task**:

1. Show SUN beats baselines on the task: eps-greedy Double DQN (`eps_greedy`), `random`,
   `discover`, `adagoal`.
2. Find the best SUN variant and simple modifications to it -- chiefly a knob for **how
   greedy** (task-exploiting) the behaviour policy is, tuned per environment -- to show
   that more exploration helps, and by how much.
3. Scale up: current envs (gridworlds, classic control, LunarLander) -> harder ones ->
   eventually Montezuma's Revenge.

Work one step at a time; the user decides when to move to the next.

## Environment setup

- Conda env **`sun_rl`** (Python 3.11), on both laptop and Triton.
  `gym_gridworlds` is installed editable from a clone of `sparisi/gym_gridworlds`
  (the PyPI package of that name is a different project; a non-editable install misses
  `grids/*.txt`).
- `pip install -e .` installs everything in `pyproject.toml` (`torch` is the one that was
  missing from `sun_rl`).
- `$SCRATCH` must be set: Hydra writes `outputs/`/`multirun/` there
  (`configs/hydra/default.yaml`). On Triton it is `$WRKDIR`.

## How the method is wired

- **Critics** ([src/critic.py](src/critic.py)): `QNetwork` = task Q (Double DQN, TD(λ)).
  `QVisitNetwork(QNetwork)` adds the successor value function (SVF) `q_visit(s, g, a)`,
  trained with HER-style relabeling ([src/goal_relabeling.py](src/goal_relabeling.py)).
  `critic(obs)` returns the task Q; `critic(obs, goal=g)` returns the SVF.
  `QEnsemble` (DISCOVER/AdaGoal) holds K `QVisitNetwork`s and updates ONE random member
  per update step (so each member's task Q also gets 1/K of the updates).
- **`agent.critic.explore_only`** (svf agent default **True**): the task Q is never
  trained. The paper's runs use this with `testing_episodes: 0`. Task experiments MUST set
  it False (only goal-conditioned agents declare the key; QNetwork always learns the task Q).
- **Actors** ([src/actor.py](src/actor.py)): `GoalConditioned.explore` does, per step:
  with prob `eps` (or during a random tail) a random action; otherwise (re)select a goal
  from `batch_size_goal` replay-buffer candidates by `_score`, then act greedily on
  the SVF towards it. Variants differ only in `_score`:
  `SUN_Ratio` = V/n (the paper's SUN), `SUN_Novelty` = 1/n, `SUN_Reachability` = V,
  `SUN_UCB` = V + sqrt(log t / n); `DISCOVER` = mu + beta*sigma and `AdaGoal` = sigma
  over the ensemble.
- **Greediness knob** (added on this branch): `agent.actor.greedy` (a `src.parameter`
  config; absent/None = original behaviour, no extra RNG draws) and
  `agent.actor.greedy_mode` (`step` | `sticky`). On a non-random step, with prob
  `greedy.value` the actor acts greedily on the task Q (same as the test policy) instead
  of following its goal. `sticky`: the first greedy step makes the rest of the episode
  greedy (value = per-step hazard). A `LinearDecay` schedule gives the "scheduled"
  variant. Order per step: eps/random-tail first, then greedy, then goal. Logged as
  `greedy`, `greedy_steps_%` (share of ALL steps, warm-up included).
  Configs: `algorithm=sun_greedy_{step,sticky,sched}` (all = `sun_ratio_adaptive` +
  `explore_only: False` + the knob). DISCOVER/AdaGoal inherit it via `++` overrides.
  Without the knob, the task Q only drives the greedy *test* policy.
- **Goal-selection timing** (`configs/algorithm/sun_ratio_*.yaml`):
  `_step` re-selects every policy step; `_episodic` keeps a goal until reached, then a
  random tail (`random_tail: True`); `_adaptive` re-selects only when V(s_now, g) drops
  below V(s_select, g) (`goal_value_check: True`).
- **Novelty `n`** comes from the replay memory ([src/replay_memory.py](src/replay_memory.py),
  [src/pseudocount.py](src/pseudocount.py)): `experiment.replay_memory.rho: null` = true
  binned counts (needs a counter from `tabular_count`), a float = pseudocount radius
  (works for any continuous obs). Goal "reached" uses the binned counter.
- **Training loop**: [src/experiment.py](src/experiment.py). `train` = return of the
  behaviour (exploration) policy; `test` = greedy w.r.t. task Q on fixed-seed episodes.
  Exploration metrics: `s_%`, `sa_%` (coverage), `s_h`, `sa_h` (normalised entropy).
  GCRL actors also log `goal_success_%`, `goal_selection_rate`, `avg_steps_to_goal`, ...

## Config and sweeps

- Hydra entry: `python main.py environment=<env> algorithm=<algo> results.data_dir=<dir> ...`
  Defaults in `configs/default.yaml`; every key documented in `configs/full_config.yaml`.
- Output: `<data_dir>/<config_id>/<seed>/data.npz`, `config_id` = hash of the config
  ([src/utils/id.py](src/utils/id.py)): MD5 of the whole resolved config minus
  `DROPPED_CFG_KEYS` (hydra, wandb, meta, results, seed). **Adding a new config key to a
  yaml, even with a "no-op" default, changes the config_id of every affected run**, so
  old results stop matching. Put new knobs only in new algorithm yamls / sweep
  overrides, or accept that old runs must be redone.
- Sweep files `configs/sweeps/*.yaml` (`global_sweep` x each `config_sweep` entry; see
  [src/utils/sweep.py](src/utils/sweep.py)). Existing: `gcrl.yaml` (paper, exploration
  only), `sun_pseudocount.yaml` (rho), and on this branch `task_baseline.yaml` (all
  methods x 8 envs on the task, 56 configs) and `greedy_pilot.yaml` (3 envs x
  {eps_greedy, discover, 3 greedy modes x values}, 33 configs).
- Validate a new sweep without running it by composing each config with
  `hydra.compose` (`initialize_config_dir(CONFIG_DIR)`) over the `load_sweep` product.

## Running

- Local: `submit_jobs_joblib.py --sweep=<s> --data_dir=<d> --seeds=0-4`.
- Triton: `submit_jobs_triton.py` (same args), a thin wrapper over `submit_jobs_slurm.py`
  (CSC) that swaps account/partition/env activation/`$SCRATCH`. Per-env timeouts and
  memory live in `TIMEOUTS_ENV`, `TIMEOUTS_ADJUST_ALGO`, `PER_SEED_MEM_MB` of
  `submit_jobs_slurm.py`; a new env needs entries there or submission refuses.
  Timeouts were sized for `explore_only: True`; learning the task Q adds an update per
  step, so task runs may need more time.
- Results: `process_data.py -f <d>` -> `<d>/results.gzip`; then `plot_results.py`
  (`--plot_config`, see `configs/plots/`) or `interactive_curves.py`.
- Debugging: `inspect_jobs.sh`, `detect_jobs_err.sh` (see README).

## Status and next steps

- 2026-10-05: greediness knob implemented and smoke-tested on `mini`; `greedy_pilot`
  (3 seeds) is the first Triton submission. Timeout factor 1.5 for `sun_greedy_*` in
  `TIMEOUTS_ADJUST_ALGO` is a guess -- check wall times in the logs.
- Pilot results (3 seeds, final test):
  - mountain_car: eps_greedy -200 (0/3 solve), DISCOVER -182 (1/3); SUN sticky 0.002
    -101 (3/3), sched->1.0 -102 (3/3), sticky 0.01 -108 (3/3); too greedy (step 0.6,
    sticky 0.05) -200. p=0 / 0.1 saved no data (likely timeout).
  - acrobot: test -500 for EVERY method incl. eps_greedy, although SUN's behaviour
    policy reaches the goal (train ~-200). The task Q-learner fails here, not exploration
    -> `dqn_acrobot.yaml` tunes tau/lrate on eps_greedy.
  - four_rooms_stuck_hard: eps_greedy solves (test 1.0); every goal-conditioned config
    saved no data on Triton, but runs fine locally -- check the Triton logs.
- Round 2: `greedy_round2.yaml` (22 configs), timeout factors raised to 2.0 / 1.6.
- Round 2 results (3 seeds): MountainCar SUN p=0 also solves (-101, 3/3) -> the win is
  SUN's exploration, the knob is neutral unless >~50% greedy. CartPole: DISCOVER 394 >
  SUN sticky 0.002 226 > eps_greedy 95; all SUN seeds hit 500 early but don't hold it.
  LunarLander: eps_greedy 203 > SUN sched->1.0 171 > sticky 0.01 153 >> SUN p=0 -47;
  sticky 0.01 leads early (89 vs -63 at 100k) then plateaus. Pendulum: all tie.
  Tuned DQN on Acrobot still -500 (skipped for now).
- Round 3 (2026-10-06): `greedy_confirm.yaml` (15 configs, seeds 3-9 -> 10 seeds total)
  and `greedy_round3.yaml` (15 configs, seeds 0-4): LunarLander variants (earlier ramp
  to p=1, more sticky greed, `sun_greedy_sticky_sched` = sticky hazard growing over
  training, eps floor 0.05, step p=1.0 diagnostic, DISCOVER rerun) + CartPole p grid.
- FourRooms "failures" (2026-10-08) were TIMEOUTS, not crashes: Triton batch-hsw runs
  SUN/DISCOVER at ~8.5 steps/s/seed (laptop ~77), seeds reached ~165k/200k. submitit
  reports a timeout as FAILED (exit 137, "timed-out and not checkpointable"); check
  the last progress line of the .out log before assuming a crash. Gridworld timeouts
  are raised in submit_jobs_triton.py. Local full run: solves (test 1.0 by 90k),
  peak RSS ~550 MB (Triton ~740 MB).
- Round 3 results (10 seeds main configs): MountainCar SUN 10/10 solved vs eps 0/10,
  DISCOVER 0/10. LunarLander: SUN sched p->1 by 150k 208 = eps_greedy 198, but SUN
  p=1 (never goal-directed) also 206 -> no exploration gain there; eps floor 0.05
  hurts SUN. CartPole: DISCOVER 383 > SUN 292 > eps 94; SUN reaches 500 but forgets.
  Confound: eps_greedy decays eps over the whole run, SUN after warm-up ->
  `eps_greedy_fast.yaml` / `dqn_fast_eps.yaml` is the fair control.
- Next: analyse the pilot (`test` vs greedy value per env and mode), widen to all envs /
  more seeds for the promising modes, then `task_baseline.yaml` for the other methods,
  then other simple modifications (eps floor, score variants, rho), then harder envs.

## Working with this user

- Cluster: give the commands, don't run them (no SSH access). Always say which machine
  a command is for (laptop vs Triton login node).
- Verify against code/logs; say when something is a guess.
- Keep changes scoped to the request; no unrequested refactors. The CSC scripts and
  shared code are the collaborator's (Simone Parisi) -- keep edits there minimal.
