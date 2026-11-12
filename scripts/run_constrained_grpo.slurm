#!/usr/bin/env bash
#SBATCH --account=mbm@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
manifest=${1:?}
runtime_dir=$(mktemp -d "/dev/shm/grpo-quality-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf "$runtime_dir"' EXIT
.venv/bin/python scripts/watermark_transport_train.py bootstrap "$manifest" --destination "$runtime_dir/packages"
export PYTHONPATH="$(dirname "$manifest")/source:$runtime_dir/packages" PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$runtime_dir/cache" TRITON_CACHE_DIR="$runtime_dir/triton"
srun .venv/bin/python "$(dirname "$manifest")/generate_constrained_grpo.py" "$manifest"
