#!/usr/bin/env bash
#SBATCH --account=mbm@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --signal=USR1@300

set -euo pipefail
repo_root=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$repo_root"
export PYTHONPATH=.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
if [[ "${1:-}" == "meta_adapt" ]]; then
    export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
fi
export LOKY_MAX_CPU_COUNT="${SLURM_CPUS_PER_TASK:-24}"
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export NLTK_DATA="$repo_root/.venv/nltk_data"
export PATH="$repo_root/.venv/bin:$PATH"
export TRITON_CACHE_DIR="$repo_root/.venv/cache/triton"
if [[ "${1:-}" == "--watermark-removal-attack" || "${1:-}" == "--watermark-removal-owner" ]]; then
    removal_input=${2:?Supply a removal task or evaluator manifest}
    removal_source=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["source"])' "$removal_input")
    export PYTHONPATH="$removal_source" PYTHONDONTWRITEBYTECODE=1
    if [[ "$1" == "--watermark-removal-attack" ]]; then
        exec srun uv --no-cache run --no-sync python -m ai_content_detector.watermarking.removal "$removal_input"
    fi
    exec srun uv --no-cache run --no-sync python "$removal_source/scripts/watermark_removal.py" run "$removal_input" "${3:?Supply a case}" "${4:?Supply a stage}"
fi
if [[ "${1:-}" == "--watermark" ]]; then
    manifest=${2:?Supply a watermark experiment manifest}
    source_root=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["source"])' "$manifest")
    export PYTHONPATH="$source_root"
    export PYTHONDONTWRITEBYTECODE=1
    exec srun uv --no-cache run --no-sync python "$source_root/scripts/watermark_experiments.py" run "$manifest" "${3:?Supply a watermark stage}"
fi
if [[ "${1:-}" == "--profile-rl-optimizations" ]]; then
    module load cuda/12.4.1
    export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++
    if ! uv --no-cache run --no-sync python -c 'import torch; import causal_conv1d'; then
        export CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=9.0
        srun uv --no-cache pip install --python "$repo_root/.venv/bin/python" --no-deps \
            --no-build-isolation --reinstall \
            "$repo_root/results/research_workflow/defender_comparison/kernel_build/causal_conv1d-1.5.3.post1.tar.gz"
    fi
    exec srun uv --no-cache run --no-sync python scripts/profile_rl_optimizations.py run "${2:?Supply an optimization profile manifest}"
fi
if [[ "${1:-}" == "--synonym-intensity" ]]; then
    exec srun uv --no-cache run --no-sync python scripts/synonym_intensity.py run "${2:?Supply an intensity manifest}" "${3:?Supply a stage}"
fi
if [[ "${1:-}" == "--synonym-robustness" ]]; then
    exec srun uv --no-cache run --no-sync python scripts/synonym_robustness.py run "${2:?Supply a synonym evaluation manifest}"
fi
if [[ "${1:-}" == "--short-probe" ]]; then
    exec srun uv --no-cache run --no-sync python scripts/short_probes.py run "${2:?Supply a probe manifest}" "${3:?Supply a defender}"
fi
if [[ "${1:-}" == "--architecture-race" ]]; then
    exec srun uv --no-cache run --no-sync python scripts/architecture_races.py run "${2:?Supply a race manifest}" "${3:?Supply a defender}"
fi
if [[ "${1:-}" == "--updated-base" ]]; then
    exec srun uv --no-cache run --no-sync python scripts/evaluate_updated_base.py run "${2:?Supply a base evaluation manifest}" "${3:?Supply a defender branch}"
fi
if [[ "${1:-}" == "--evaluate-checkpoint" ]]; then
    exec srun uv --no-cache run --no-sync python scripts/evaluate_attacker_checkpoint.py "${2:?Supply a group}" "${3:?Supply a checkpoint}" "${4:?Supply an evaluation directory}"
fi
if [[ "${1:-}" == "--profile-baseline" ]]; then
    exec srun uv --no-cache run --no-sync python scripts/profile_baseline.py --continuation-job "${2:?Supply a continuation job}"
fi
if [[ "${1:-}" == "--chain" ]]; then
    exec srun uv --no-cache run --no-sync python scripts/research_chain.py run "${2:?Supply a chain manifest}" "${3:?Supply a stage}"
fi
exec srun uv --no-cache run --no-sync python scripts/research_workflow.py "${1:?Supply a workflow stage}"
