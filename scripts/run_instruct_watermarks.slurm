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
export BIRA_PAPER_PYTHON=$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["paper_python"])' "$manifest")
test -x "$BIRA_PAPER_PYTHON"
runtime_dir=$(mktemp -d "$(dirname "$manifest")/job-cache-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf "$runtime_dir"' EXIT
export TRITON_CACHE_DIR="$runtime_dir/triton" XDG_CACHE_HOME="$runtime_dir/cache"
export CUDA_CACHE_PATH="$runtime_dir/cuda" TORCH_EXTENSIONS_DIR="$runtime_dir/torch_extensions"
export PYTHONPATH="$source_root:$source_root/scripts"
set +e
srun .venv/bin/python "$source_root/scripts/instruct_watermark_experiments.py" run "$manifest" "${2:?}"
status=$?
set -e
if [[ "$status" == 75 ]]; then
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi
exit "$status"
