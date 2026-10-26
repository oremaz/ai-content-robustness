#!/usr/bin/env bash
#SBATCH --account=mbm@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
set -euo pipefail
repo_root=${SLURM_SUBMIT_DIR:?Missing submission directory}
manifest=${1:?Supply an image experiment manifest}
stage=${2:?Supply an image experiment stage}
entrypoint=${3:-image_experiments.py}
case "$entrypoint" in
    image_experiments.py|image_attacker_ablation.py) ;;
    *) exit 2 ;;
esac
run_root=$(dirname "$manifest")
cd "$repo_root"
export PYTHONPATH="$run_root/source"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PATH="$repo_root/.venv/bin:$PATH"
exec srun uv --no-cache run --no-sync python "$run_root/source/scripts/$entrypoint" run "$manifest" "$stage"
