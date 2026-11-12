#!/usr/bin/env bash
#SBATCH --account=mbm@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --signal=USR1@300
#SBATCH --requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
manifest=$(realpath "${1:?}")
stages_file=$(realpath "${2:-$(dirname "$manifest")/stages.json}")
source_root=$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["source"])' "$manifest")
worker_script=$(realpath "${3:-$source_root/scripts/translation_experiments.py}")
max_requeues=$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["max_requeues"])' "$manifest")
if (( ${SLURM_RESTART_COUNT:-0} > max_requeues )); then
    echo 'Requeue limit reached; retained checkpoints require budget review.' >&2
    exit 1
fi
runtime_dir=$(mktemp -d "/dev/shm/translation-${SLURM_JOB_ID}-XXXXXX")
nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" --query-gpu=timestamp,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv -l 10 >"$(dirname "$manifest")/gpu-${SLURM_JOB_ID}-${SLURM_RESTART_COUNT:-0}.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true; rm -rf "$runtime_dir"' EXIT
.venv/bin/python "$source_root/scripts/watermark_transport_train.py" bootstrap "$manifest" --destination "$runtime_dir/packages"
export PYTHONPATH="$source_root:$source_root/scripts:$runtime_dir/packages" PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$runtime_dir/cache" TRITON_CACHE_DIR="$runtime_dir/triton"
export NLTK_DATA="$PWD/.venv/nltk_data:$PWD/results/research_workflow/watermarking/resources/nltk_data"
mapfile -t stages < <(.venv/bin/python -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))))' "$stages_file")
for stage in "${stages[@]}"; do
    set +e
    srun .venv/bin/python -c 'import json,sys,runpy; from unittest.mock import patch; c=json.load(open(sys.argv[3])); script=sys.argv.pop(1)
with patch("importlib.metadata.packages_distributions",return_value=c["package_map"]): runpy.run_path(script,run_name="__main__")' "$worker_script" phase "$manifest" "$stage"
    status=$?
    set -e
    if [[ "$status" == 75 ]]; then
        scontrol requeue "$SLURM_JOB_ID"
        exit 0
    fi
    if [[ "$status" != 0 ]]; then exit "$status"; fi
done
