"""
Triton (Aalto) variant of submit_jobs_slurm.py.

Everything -- sweeps, seeds, chunking into job arrays, one srun task per seed,
memory/timeout tables, --skip_duplicates -- is inherited from submit_jobs_slurm.py,
so read its docstring for how jobs are laid out. This file only swaps the
cluster-specific settings:

    - no --account (Triton auto-assigns it) and no --partition (auto-assigned
      from the requested time/resources)
    - $SCRATCH is $WRKDIR (/scratch/work/<user>, 200 GB); $HOME is only 10 GB
    - the job activates the `sun_rl` conda env through the mamba module
    - GPUs are requested with a plain `gpu:1`, no partition
    - CPU jobs are restricted to newer nodes (SLURM_CONSTRAINT)

RUN THIS ON THE TRITON LOGIN NODE, from the repo root, with the env active:

    module load mamba && source activate sun_rl
    python submit_jobs_triton.py --sweep=<file> --data_dir=<dir> --seeds 0-4 --debug

Same arguments as submit_jobs_slurm.py. Use --debug first to submit a single job.

Logs land in $WRKDIR/slurm_out/<jobid>_<element>_<task>_log.{out,err}.
"""

import os

import submit_jobs_slurm as base


try:
    os.environ["SCRATCH"] = os.environ["WRKDIR"]
except KeyError:
    raise SystemExit("$WRKDIR is not set: are you on the Triton login node?")

CONDA_ENV = "sun_rl"

base.SLURM_ACCOUNT = None  # submitit drops None parameters, so no --account
base.SLURM_GPU = dict(slurm_gres="gpu:1")
# Newer CPUs only: the default batch-hsw/bdw (Haswell/Broadwell) nodes are the
# slowest, and SUN runs are single-threaded and CPU-bound. "a|b" = either feature.
# Partitions with these features: batch-milan (milan), batch-csl/batch-pht (csl).
# Set to None to allow any node (shorter queue, slower runs).
SLURM_CONSTRAINT = "milan|csl"
if SLURM_CONSTRAINT:
    base.SLURM_EXTRA = dict(slurm_constraint=SLURM_CONSTRAINT)
base.SLURM_SETUP = [
    "module load mamba",
    f"source activate {CONDA_ENV}",
    "export SCRATCH=$WRKDIR",
    # keep `cd <repo>` and the thread/malloc exports of the base setup
    *base.SLURM_SETUP[base.SLURM_SETUP.index(f"cd {base.SCRIPT_DIR}"):],
]

# Triton's batch-hsw (Haswell) nodes run SUN/DISCOVER at ~8.5 steps/s per seed
# (measured 2026-10-05; ~9x slower than an M-series laptop). The base timeouts were
# too short for the gridworlds: four_rooms_stuck_hard timed out at ~165k/200k steps.
# These are sized as steps / 8.5 * 1.3 margin, divided by the x2.0 factor of the
# sun_greedy_* algorithms in TIMEOUTS_ADJUST_ALGO.
base.TIMEOUTS_ENV.update({
    "four_rooms_stuck_hard": 260,             # 200k steps -> SUN 520 min
    "three_rooms_wall_mini_nonuniform": 130,  # 100k steps -> SUN 260 min
    "travel_field_small": 390,                # 300k steps -> SUN 780 min
    # explore_hard.yaml (2026-10-09). Speeds estimated from local probes scaled by
    # the measured Triton/laptop ratio: SUN ~14 steps/s on the small gridworlds,
    # ~7 on travel_field and ~9.5 on wall_large (DISCOVER ~9.4 / ~7.5).
    "river_swim": 45,                         # 20k steps
    "straight": 45,                           # 30k steps
    "four_rooms_stuck": 260,                  # 200k steps -> SUN 520 min
    "four_rooms_loop": 260,                   # 200k steps
    "travel_field": 1550,                     # 1M steps -> SUN 3100 min (~52 h)
    # wall_large: the 1M-step SUN/DISCOVER runs did not finish in 40/50 h. wall_500k.yaml
    # runs 500k steps; 1500 -> SUN 3000 min, DISCOVER 3750 min, enough down to ~2.8 steps/s.
    "wall_large": 1500,
})
# Goal-conditioned algorithms that learn the task Q (explore_only=False) run at the
# speed of the sun_greedy_* ones (x2.0 in submit_jobs_slurm.py); the base factors
# (1 / 1.3) were sized for pure exploration. Used by mc_ablation.yaml.
base.TIMEOUTS_ADJUST_ALGO.update({
    "sun_ratio_adaptive": 2.0,
    "sun_novelty_adaptive": 2.0,
    "sun_reachability_adaptive": 2.0,
    "sun_ucb_adaptive": 2.0,                  # (not in mc_ablation: identical to the ratio there)
    "adagoal": 2.5,                           # ensemble, like discover
    "sun_greedy_reverse": 2.0,                # same cost per step as the other sun_greedy_*
    "sun_greedy_bandit": 2.0,
    "sun_greedy_bandit_explore": 2.0,
})

# Per-seed memory (MB) for the same environments = replay memory + ~650 MB of
# Python/torch (FourRooms-Hard on Triton: 740 MB peak with a 106 MB buffer) + 10%.
# The replay memory is allocated in full at start-up and stores gridworld observations
# one-hot, so it is (training_steps x n_states) values: exact sizes below are from the
# array shapes, for SUN/DISCOVER (which also store each step's goal; DQN needs ~2/3).
# Do NOT size this from macOS RSS: macOS compresses the still-zero pages and reported
# ~2 GB for wall_large, whose jobs were then OOM-killed on Triton at 3.1 GB/seed.
base.PER_SEED_MEM_MB.update({
    "river_swim": 800,                        # buffer < 1 MB
    "straight": 800,                          # buffer < 1 MB
    "four_rooms_stuck": 900,                  # buffer 106 MB
    "four_rooms_loop": 900,                   # buffer 106 MB
    "travel_field": 3300,                     # buffer 2.3 GB (784 states x 1M steps)
    "wall_large": 8700,                       # buffer 7.2 GB (2500 states x 1M steps)
})

# Queue limits (MAX_RUNNING_JOBS, MAX_SUBMIT_JOBS, MAX_ELEMENTS_PER_ARRAY) are
# inherited. Check them against Triton before large sweeps:
#   sacctmgr show assoc user=$USER format=account%30,partition%20,maxjobs,maxsubmit
#   scontrol show config | grep -i MaxArraySize


if __name__ == "__main__":
    # Extra Triton-only option: one wall-clock limit for every job of this submission,
    # replacing TIMEOUTS_ENV x TIMEOUTS_ADJUST_ALGO (still capped at base.MAX_TIMEOUT).
    # For sweeps whose length differs from what the per-env tables assume, e.g.
    #   python submit_jobs_triton.py --sweep=wall_1m ... --timeout_min 4240
    import argparse
    import sys
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--timeout_min", type=int, default=None)
    # Node features for this submission only, replacing SLURM_CONSTRAINT, e.g. "milan"
    pre.add_argument("--constraint", default=None)
    extra, sys.argv[1:] = pre.parse_known_args(sys.argv[1:])
    if extra.constraint is not None:
        base.SLURM_EXTRA = dict(slurm_constraint=extra.constraint)
        print(f"Node constraint: {extra.constraint} (--constraint)")
    if extra.timeout_min is not None:
        base.TIMEOUTS_ENV = {env: extra.timeout_min for env in base.TIMEOUTS_ENV}
        base.TIMEOUTS_ADJUST_ALGO = {}
        print(f"Every job gets {min(extra.timeout_min, base.MAX_TIMEOUT)} min (--timeout_min)")
    base.main()
