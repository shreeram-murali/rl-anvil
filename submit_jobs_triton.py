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
base.SLURM_SETUP = [
    "module load mamba",
    f"source activate {CONDA_ENV}",
    "export SCRATCH=$WRKDIR",
    # keep `cd <repo>` and the thread/malloc exports of the base setup
    *base.SLURM_SETUP[base.SLURM_SETUP.index(f"cd {base.SCRIPT_DIR}"):],
]

# Queue limits (MAX_RUNNING_JOBS, MAX_SUBMIT_JOBS, MAX_ELEMENTS_PER_ARRAY) are
# inherited. Check them against Triton before large sweeps:
#   sacctmgr show assoc user=$USER format=account%30,partition%20,maxjobs,maxsubmit
#   scontrol show config | grep -i MaxArraySize


if __name__ == "__main__":
    base.main()
