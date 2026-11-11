#!/usr/bin/env bash
#SBATCH --account=mbm@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --signal=USR1@300
#SBATCH --requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
manifest=${1:?}
source_root=$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["source"])' "$manifest")
paper_python=$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["python"])' "$manifest")
overlay=$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["overlay"])' "$manifest")
cache_dir=$(mktemp -d "$(dirname "$manifest")/job-cache-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf "$cache_dir"' EXIT
export TRITON_CACHE_DIR="$cache_dir/triton" XDG_CACHE_HOME="$cache_dir/cache"
export CUDA_CACHE_PATH="$cache_dir/cuda" TORCH_EXTENSIONS_DIR="$cache_dir/torch_extensions"
export HF_HOME="$cache_dir/hf" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$overlay:$source_root:$source_root/scripts"
set +e
srun "$paper_python" "$source_root/scripts/jina_watermark_similarity.py" run "$manifest" "${2:?}"
status=$?
set -e
if [[ "$status" == 75 ]]; then
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi
exit "$status"
