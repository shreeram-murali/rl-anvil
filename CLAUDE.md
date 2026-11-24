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
- 2026-10-09 results: fast-eps DQN also fails MountainCar (0/10) -> SUN's win there is
  real. CartPole fast-eps 235 ~ SUN 243-292; LunarLander all ~190-210 -> SUN's earlier
  edge there was the eps schedule. Gridworlds (FourRooms-Hard, ThreeRooms-Wall,
  TravelField-12x12): fast-eps DQN solves all -> too easy. Triton with
  constraint milan|csl: SUN ~13-14 steps/s on gridworlds (vs ~8.5 on hsw).
- Weekend batch 2026-10-09: `explore_hard.yaml` (42 configs, 5 seeds) on river_swim,
  straight, four_rooms_stuck, four_rooms_loop, travel_field, wall_large -- every
  method everywhere, to find more envs where fast-eps DQN fails. Fixed env ids
  (RiverSwim-1x6, Straight-1x20). taxi is unusable for SUN/DISCOVER (no binned counter
  -> crash on goal reached). Timeouts/memory for these envs live in
  submit_jobs_triton.py, from local probes (replay memory is allocated in full at
  start-up, so first-checkpoint RSS of a full-length run = steady state).
  CORRECTION: macOS RSS badly underestimates that (zero pages are compressed); the
  wall_large / travel_field DISCOVER jobs were OOM-killed at start-up. Gridworld obs
  are stored one-hot, so buffer = steps x n_states values (wall_large 7.2 GB/seed for
  SUN/DISCOVER). Size PER_SEED_MEM_MB from array shapes, not laptop RSS.
- 2026-10-12 results (explore_hard): small gridworlds solved by all (SUN somewhat
  earlier/more consistent); TravelField-28x28 (distance-shaped reward) favours DQN,
  SUN finds high returns early but doesn't hold them (as on CartPole); Wall-50x50: both
  DQNs 0/5 at 1M steps -> second DQN failure, but SUN/DISCOVER did not finish.
- Batch 2026-10-23: `wall_500k.yaml` (7 configs, every method at 500k steps,
  --seeds_per_chunk=1), `mc_ablation.yaml` (5 configs, 10 seeds: random, AdaGoal and
  SUN's score ablation -- ratio / novelty / reachability, all learning the task Q),
  `learner_stability.yaml` (16 configs: tau x lr-decay x {SUN, fast-eps DQN} on
  CartPole and TravelField@500k). SUN_UCB == SUN_Ratio in practice (counts clipped at
  1e-3 -> both pick the most reachable untried (s, a)); left out.
  plot_greedy.py does not yet show the ablation actors, AdaGoal, random, or the
  tau / lr variants -- extend it when the results come in.
- 2026-10-26 results: Wall-50x50 @500k: DQN (both, also @1M), DISCOVER and SUN p=0 all
  0; greedy SUN learns the small +0.1 reward in 14/15 seeds and the +1 goal in 5/15
  (sticky 0.002: 3/5, 2 hold it) -> second DQN-failure task, and here the knob is
  essential (plain SUN covers 98% but never collects reward). MountainCar ablation (10
  seeds): novelty-only 1/n = ratio V/n (both 10/10), reachability-only V fails (0/10,
  coverage below random), AdaGoal -175 (0/10). Learner stability: slower tau / lr
  decay don't help SUN (CartPole default 315 best); they do help fast-eps DQN (394).
  Figures: plot_greedy.py --training_steps, new plot_compare.py (ablation, learner).
- Wall-50x50 layout: start bottom-left (49,0); goal +1 at (49,49) in a pocket closed by
  a vertical wall (col 39, rows 9-49) and a horizontal wall (row 9, cols 4-39), so
  the path goes up, across the top, down: ~130 steps. Small +0.1 at (0,49), ~98 steps,
  next to that path; both terminate. 500-step episodes. Obs are one-hot (2500).
  Profile: ~80% of a SUN step is the SVF update (obs+goal one-hot = 5000 inputs),
  ~12% the task Q update; goal selection negligible. Triton SUN ~4.8 steps/s.
- Batch 2026-10-29 (Wall-50x50, all in data_explore_hard, --seeds_per_chunk=1):
  `wall_long.yaml` (A: 1M steps, update_frequency=2, sticky 0.001/0.002/0.004 +
  fast-eps DQN; --timeout_min 2700), `wall_credit.yaml` (B: task-Q sequence_length
  32, lmbda 0.99, sticky 0.002 + fast-eps DQN), `wall_500k_more.yaml` (C: seeds 5-9
  of sticky 0.002 @500k, same config_id dcf24627), `wall_1m.yaml` (D: full 1M, sticky
  0.002, config_id 98601d6c; --timeout_min 4240). submit_jobs_triton.py gained
  --timeout_min (one limit for every job of a submission).
- 2026-11-02 results (Wall-50x50, sticky 0.002): 1M steps solves it, 5/5 seeds learn
  and hold the goal (0.88 +- 0.05; first goal 400-560k); 500k (10 seeds) 0.60, 8/10
  reach it, all just before the end. The goal appears ~50k steps after SUN's coverage
  hits 99% (the pocket is the last region covered) -> coverage is the bottleneck, and
  it tracks gradient UPDATES, not env steps (SVF must learn to navigate):
  update_frequency 2 delays coverage/goal to ~600-790k / 780-990k, so no saving per
  unit of compute. seq 32 + lambda 0.99 no help (0.56). p 0.001/0.002/0.004 ~ same.
  DQN 0 in every variant (coverage 76-86%). Triton hours: A 35, B 41, C 29-34, D 60.
  Notes: main.tex Wall section rewritten (tab:wall-topline, tab:wall-ablation);
  figures from `plot_compare.py --figure wall`. plot_greedy.py now skips runs with
  non-default update_frequency / sequence_length / lmbda unless selected with --where.
- Batch 2026-11-05: `wall_discover.yaml` (DISCOVER + sticky 0.002, 800k steps, config_id
  ef1d440e, --timeout_min 4248). DISCOVER ~5% slower than SUN on Triton, so 1M (~65 h)
  risked the 70.8 h cap. plot_greedy.describe labels DISCOVER without its knob.
  Also: `wall_800k_plain.yaml` (plain DISCOVER 6f63bd1c, SUN p=0 92bf8761, 800k) and
  `wall_1m` seeds 5-9 (1M took 59-61 h; some nodes ran 17% slow on the 500k runs ->
  close to the cap). submit_jobs_triton.py gained --constraint (per-submission node
  features, replacing SLURM_CONSTRAINT).
- 2026-11-09 results: CORRECTION -- the knob is NOT essential on Wall-50x50: plain SUN
  (p=0) at 800k learns the goal 5/5 (0.88; cov99 416-536k, goal median 656k); its 0 at
  500k was the run ending too early. p=0.002 is faster (cov99 360-450k, goal median
  475k); 1M now 8 seeds, 0.90 +- 0.04, 8/8 hold (seeds 5, 6 hit the 70.8 h cap on
  milan4; seeds 7-9 took 65-69 h vs 59-61 h before). DISCOVER never reaches the goal
  (plain 0; + sticky 0.002: small reward in 3/5, 0.04), coverage 96% at 800k.
  plot_greedy.describe now skips DISCOVER + knob; plot_compare --figure wall shows it.
- 2026-11-20: offline test of "go greedy when exploration saturates" on plain-SUN logs
  (`plot_compare.py -f . --figure entropy_rule` -> plots/compare_entropy_rule): rule =
  growth over W ckpts < r x max growth so far. Cumulative entropy is concave, so the
  entropy rule fires early everywhere (Wall 200-320k, before coverage completes);
  the coverage rule times Wall well (360-590k, around goal discovery) but is far too
  late / never on LunarLander, which wants greed early. No exploration-only signal
  fits all envs -> task-return signal (bandit). New greedy_mode values in
  src/actor.py: `reverse_sticky` (greedy first, hazard -> goals) and `bandit`
  (per-episode SW-UCB over sticky hazards `greedy_arms`, reward = episode return;
  uniform while all returns are equal). Configs `sun_greedy_reverse`,
  `sun_greedy_bandit` (timeout factor 2.0 in submit_jobs_triton.py). Sweeps
  `greedy_adaptive.yaml` (MC/CartPole/LunarLander x {bandit, reverse 0.005/0.02/0.1})
  and `wall_adaptive.yaml` (800k, bandit + reverse 0.005/0.02, --timeout_min 4248).
- 2026-11-23 results (greedy_adaptive / wall_adaptive, 5 seeds; `plot_compare.py -f .
  --figure adaptive`, 3rd column = baselines): bandit matches the tuned setting on
  LunarLander (203, picks p=1 in 5/5, faster than the schedule) and Wall (0.88, 5/5,
  goal 576-784k ~ plain SUN), but is slow on MountainCar (-109, 2/5): uniform arms while
  all returns equal -> ~40% greedy steps before any reward. Reverse sticky's best p
  flips per env (0.1 MC/CartPole, 0.005 LunarLander) -> dropped. CartPole: all tie.
  Fix: `bandit_flat: min` (pick the smallest hazard until two episode returns first
  differ; permanent switch) -> `sun_greedy_bandit_explore` (ids MC 673f4948, CartPole
  1e65dd41, LL ae5c6075, Wall@800k 8e44af7d); sweeps `greedy_bandit_explore.yaml`,
  `wall_bandit_explore.yaml`. notes/main.tex section "Choosing the greediness
  automatically" (tab:adaptive, figures adaptive_greedy.pdf, entropy_rule.pdf).
- Next: analyse the pilot (`test` vs greedy value per env and mode), widen to all envs /
  more seeds for the promising modes, then `task_baseline.yaml` for the other methods,
  then other simple modifications (eps floor, score variants, rho), then harder envs.

## Working with this user

- Cluster: give the commands, don't run them (no SSH access). Always say which machine
  a command is for (laptop vs Triton login node).
- Verify against code/logs; say when something is a guess.
- Keep changes scoped to the request; no unrequested refactors. The CSC scripts and
  shared code are the collaborator's (Simone Parisi) -- keep edits there minimal.
