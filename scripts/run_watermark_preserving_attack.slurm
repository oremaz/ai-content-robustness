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
maximum_allocations=$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["compute"]["maximum_allocations"])' "$manifest")
runtime_dir=$(mktemp -d "/dev/shm/ai-robustness-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf "$runtime_dir"' EXIT
.venv/bin/python "$source_root/scripts/watermark_transport_train.py" bootstrap "$manifest" --destination "$runtime_dir/packages"
export PYTHONPATH="$source_root:$runtime_dir/packages" PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$runtime_dir/cache" TRITON_CACHE_DIR="$runtime_dir/triton"
export CUDA_CACHE_PATH="$runtime_dir/cuda" TORCH_EXTENSIONS_DIR="$runtime_dir/torch_extensions"
set +e
srun .venv/bin/python "$source_root/scripts/watermark_preserving_attack.py" run "$manifest"
status=$?
set -e
if [[ "$status" == 75 && "${SLURM_RESTART_COUNT:-0}" -lt "$((maximum_allocations - 1))" ]]; then
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi
exit "$status"
