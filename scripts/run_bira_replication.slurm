#!/usr/bin/env bash
#SBATCH --account=mbm@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --signal=USR1@300
#SBATCH --requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export WANDB_MODE=offline HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export NLTK_DATA="$PWD/.venv/nltk_data:$PWD/results/research_workflow/watermarking/resources/nltk_data"
export HF_HUB_CACHE="$PWD/results/research_workflow/watermarking/resources/hf"
manifest=${1:?}
source_root=$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["source"])' "$manifest")
runtime_archive=$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["runtime_archive"])' "$manifest")
runtime_dir=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/bira-runtime-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf "$runtime_dir"' EXIT
export TRITON_CACHE_DIR="$runtime_dir/triton"
export XDG_CACHE_HOME="$runtime_dir/cache"
export CUDA_CACHE_PATH="$runtime_dir/cuda"
export TORCH_EXTENSIONS_DIR="$runtime_dir/torch_extensions"
tar -xzf "$runtime_archive" -C "$runtime_dir"
export BIRA_PAPER_PYTHON="$runtime_dir/bin/python"
export PYTHONPATH="$source_root:$source_root/scripts"
case_args=()
if [[ "${2:?}" == case-loop ]]; then
    IFS=, read -r -a case_indices <<< "${3:?}"
elif [[ -n "${3:-}" ]]; then
    case_args=(--case "$3")
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    case_args=(--case "$SLURM_ARRAY_TASK_ID")
fi
set +e
if [[ "$2" == case-loop ]]; then
    status=0
    for case_index in "${case_indices[@]}"; do
        srun .venv/bin/python "$source_root/scripts/bira_replication.py" run "$manifest" case --case "$case_index"
        status=$?
        if [[ "$status" != 0 ]]; then break; fi
    done
else
    srun .venv/bin/python "$source_root/scripts/bira_replication.py" run "$manifest" "$2" "${case_args[@]}"
    status=$?
fi
set -e
if [[ "$status" == 75 ]]; then
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi
exit "$status"
