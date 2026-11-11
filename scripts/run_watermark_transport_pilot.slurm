#!/usr/bin/env bash
#SBATCH --account=mbm@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --signal=USR1@120
#SBATCH --requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
manifest=${1:?}
source_root=$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["source"])' "$manifest")
export PYTHONPATH="$source_root" PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
runtime_dir=$(mktemp -d "$(dirname "$manifest")/job-cache-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf "$runtime_dir"' EXIT
export XDG_CACHE_HOME="$runtime_dir/cache" TRITON_CACHE_DIR="$runtime_dir/triton"
export CUDA_CACHE_PATH="$runtime_dir/cuda" TORCH_EXTENSIONS_DIR="$runtime_dir/torch_extensions"
set +e
srun .venv/bin/python "$source_root/scripts/watermark_transport_pilot.py" run "$manifest"
status=$?
set -e
if [[ "$status" == 75 && "${SLURM_RESTART_COUNT:-0}" -lt 1 ]]; then
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi
exit "$status"
