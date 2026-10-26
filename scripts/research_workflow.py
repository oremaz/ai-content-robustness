"""Execute full detection notebooks and the existing evasion experiments."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import runpy
import signal
import hashlib
import shutil
import tarfile
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "research_workflow"
NOTEBOOKS = {
    "text_notebook": "bench-aitextdetect.ipynb",
    "image_notebook": "bench-imai-artifact.ipynb",
}
from ai_content_detector.rl_evasion.defender_comparison.plan import STAGES as COMPARISON_STAGES


def source_provenance():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "source.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with (OUTPUT / "report.lock").open("a") as report_lock:
            fcntl.flock(report_lock, fcntl.LOCK_EX)
            return _source_provenance()


def _source_provenance():
    repositories = {"ai-content-robustness": ROOT, "ml-playground": ROOT.parent / "ml-playground"}
    manifest = {}
    paths = []
    for label, repository in repositories.items():
        names = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=repository).decode().split("\0")
        files = {name: hashlib.sha256((repository / name).read_bytes()).hexdigest()
                 for name in names if name and (repository / name).is_file()}
        manifest[label] = {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(), "files": files}
        paths.extend((repository / name, f"{label}/{name}") for name in sorted(files))
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    directory = OUTPUT / "source" / fingerprint
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "source.tar.gz"
    if not archive.exists():
        with tarfile.open(archive, "w:gz") as bundle:
            for path, name in paths:
                bundle.add(path, arcname=name, recursive=False)
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        for label, repository in repositories.items():
            (directory / f"{label}.patch").write_bytes(subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=repository))
    import importlib.metadata
    versions = {distribution.metadata["Name"]: distribution.version
                for distribution in importlib.metadata.distributions()}
    return {"source_fingerprint": fingerprint, "source_archive": str(archive),
            "source_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "installed_versions": versions}


def validate_prerequisites(stage):
    from scripts.prepare_research import text_configs, record
    manifest = json.loads((OUTPUT / "prerequisites.json").read_text())
    if stage == "text_notebook":
        notebook = json.loads((ROOT / "ai_content_detector/notebooks" / NOTEBOOKS[stage]).read_text())
        required = [item["hf_model"] for item in text_configs(notebook).values()]
        required += ["yaful/MAGE", "artem9k/ai-text-detection-pile", "IntelLabs/AI-Peer-Review-Detection-Benchmark"]
    elif stage == "image_notebook":
        def completed_artifact():
            target = ROOT / "data/artifact-real-fake"
            values = json.loads((target / ".artifact-complete.json").read_text())
            if not values.get("complete") or not values.get("crc_verified"):
                raise ValueError("Full ArtiFact validation is incomplete")
            return {**values, "path": str(target.resolve()), "offline_ready": True}
        record(manifest, "awsaf49/artifact-dataset", completed_artifact)
        required = ["awsaf49/artifact-dataset", "birdy654/cifake-real-and-ai-generated-synthetic-images",
                    "sunnykakar/shoes-dataset-real-and-ai-generated-images"]
        required += [f"vision/{name}" for name in ["binary_efficientnet_b4_ns", "clip_classifier",
                     "timm_convnextv2_tiny", "timm_dinov2_vit_base", "resnet50"]]
    else:
        text_models = ["Qwen/Qwen3.5-9B-Base", "tiiuae/falcon-7b",
                       "tiiuae/falcon-7b-instruct", "openai-community/roberta-base-openai-detector"]
        if stage == "ddpo_image":
            required = ["stable-diffusion-v1-5/stable-diffusion-v1-5", "openai/clip-vit-base-patch32",
                        "Gustavosta/Stable-Diffusion-Prompts"]
        else:
            required = [*text_models, "abisee/cnn_dailymail", "intfloat/e5-base-v2"]
            if stage in COMPARISON_STAGES:
                required += ["google/gemma-1.1-2b-it"]
            if stage == "multispin":
                required += ["rrivera1849/LUAR-MUD", "spacy/en_core_web_sm", "nltk/cmudict"]
            if stage == "benchmark":
                required += ["Hello-SimpleAI/HC3", "tuner007/pegasus_paraphrase", "nltk/wordnet"]
    unavailable = [name for name in required if manifest.get(name, {}).get("status") != "ready"
                   or not manifest.get(name, {}).get("offline_ready")]
    if unavailable:
        raise RuntimeError(f"Full offline prerequisites are not ready: {unavailable}")
    selected = {name: manifest[name] for name in required}
    for name, entry in selected.items():
        path = Path(entry.get("path", ""))
        if path.parent.name == "snapshots" and entry.get("revision"):
            reference = path.parent.parent / "refs/main"
            if reference.is_file() and reference.read_text().strip() != entry["revision"]:
                raise ValueError(f"Cached main revision changed after preparation: {name}")
    return selected


def run_module(module, arguments):
    previous = sys.argv
    try:
        sys.argv = [module, *arguments]
        runpy.run_module(module, run_name="__main__")
    finally:
        sys.argv = previous



def execute_notebook(input_path, output_path):
    import papermill
    from jupyter_client import KernelManager
    from nbclient.exceptions import DeadKernelError
    from ai_content_detector.rl_evasion.checkpointing import CheckpointPause

    manager = KernelManager(kernel_name="ai-content-robustness")
    requested = False
    previous_handlers = {}

    def request_stop(signum, frame):
        nonlocal requested
        requested = True
        provisioner = manager.provisioner
        if provisioner is not None and provisioner.pid is not None:
            try:
                os.kill(provisioner.pid, signum)
            except ProcessLookupError:
                pass

    def before_cell(**kwargs):
        if requested:
            raise CheckpointPause("Notebook scheduler stop requested between cells")

    for signum in (signal.SIGUSR1, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, request_stop)
    try:
        papermill.execute_notebook(
            input_path, output_path, kernel_name="ai-content-robustness",
            km=manager, cwd=str(ROOT), execution_timeout=None,
            log_output=True, on_cell_start=before_cell,
        )
    except DeadKernelError as error:
        if requested:
            raise CheckpointPause("Notebook kernel stopped on scheduler signal; resume from saved training state") from error
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)


def run_stage(stage: str) -> None:
    os.chdir(ROOT)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    provenance = {
        "stage": stage,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "python": sys.version,
        **source_provenance(),
    }
    destination = OUTPUT / f"{stage}.json"
    attempt_path = OUTPUT / "attempts" / f"{stage}-{provenance['slurm_job_id'] or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}.json"
    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(provenance, indent=2) + "\n")
    from scripts.report_research import update_report
    try:
        update_report()
        provenance["prerequisites"] = validate_prerequisites(stage)
        if stage in NOTEBOOKS:
            import nbformat
            notebook = NOTEBOOKS[stage]
            prepared = nbformat.read(ROOT / "ai_content_detector" / "notebooks" / notebook, as_version=4)
            if stage == "text_notebook":
                for cell in prepared.cells:
                    if cell.cell_type == "code" and "def evaluate_qlora_models(" in cell.source:
                        cell.source += "\n\nfrom scripts.notebook_evaluation import install_text_evaluation_cache\ninstall_text_evaluation_cache(globals())\n"
                    if cell.cell_type == "code" and "def peer_review_text" in cell.source:
                        metadata = json.loads((ROOT / "data/peer_review_evaluation.json").read_text())
                        if metadata.get("sampling_cell_sha256") != hashlib.sha256(cell.source.encode()).hexdigest():
                            raise ValueError("Prepared peer-review sample does not match the current notebook sampling code")
                        cell.source = '''import hashlib, json
from pathlib import Path
peer_path = Path("data/peer_review_evaluation.parquet")
peer_metadata = json.loads(peer_path.with_suffix(".json").read_text())
assert hashlib.sha256(peer_path.read_bytes()).hexdigest() == peer_metadata["sha256"]
assert peer_metadata["sampling"]["seed"] == SEED
assert peer_metadata["sampling"]["sample_count"] == 2000
assert peer_metadata["sampling"]["buffer_size"] == 2000
df_external_ood = pd.read_parquet(peer_path)
assert len(df_external_ood) == 2000 and df_external_ood.label.value_counts().to_dict() == {0: 1000, 1: 1000}
X_external_ood = df_external_ood["text"].tolist()
y_external_ood = df_external_ood["label"].values
print(f"External OOD eval set: {len(df_external_ood):,} samples")
print(pd.crosstab(df_external_ood["generator"], df_external_ood["label"]))
'''
            for cell in prepared.cells:
                if cell.cell_type == "code":
                    cell.outputs = []
                    cell.execution_count = None
            if stage == "text_notebook":
                export = '''tables = {"MAGE test": df_all_results, "Pile cross-dataset": df_pile_results,
          "Peer-review external OOD": df_external_ood_results}
expected = set(baseline_names) | set(ALL_QLORA_MODELS)
for frame in tables.values():
    assert set(frame.model) == expected, "An expected text model is missing from evaluation"
payload = {"tables": {name: json.loads(frame.to_json(orient="records")) for name, frame in tables.items()}}
import joblib
Path("results/bench_aitextdetect").mkdir(parents=True, exist_ok=True)
joblib.dump(tfidf, "results/bench_aitextdetect/tfidf_vectorizer.pkl")
'''
            else:
                export = '''tables = {"ArtiFact holdout": artifact_test_metrics, "CIFAKE external": cifake_test_metrics,
          "Shoes external": shoes_metrics}
for frame in tables.values():
    assert set(frame.model) == set(SELECTED_MODELS), "An expected image model is missing from evaluation"
payload = {"tables": {name: json.loads(frame.to_json(orient="records")) for name, frame in tables.items()},
           "intervention": json.loads(robust_comparison.to_json(orient="records")),
           "passes_exploratory_gate": passes_exploratory_gate}
'''
            export = "import json, os\nfrom pathlib import Path\n" + export
            export += f'''payload["job_id"] = os.environ.get("SLURM_JOB_ID")
Path({str(OUTPUT / f"{stage}_measurements.json")!r}).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\\n")
'''
            prepared.cells.append(nbformat.v4.new_code_cell(export))
            input_path = attempt_path.with_suffix(".input.ipynb")
            output_path = attempt_path.with_suffix(".ipynb")
            nbformat.write(prepared, input_path)
            execute_notebook(input_path, output_path)
            from scripts.report_research import finalize_notebook
            finalize_notebook(stage, output_path)
            shutil.copyfile(output_path, OUTPUT / notebook)
            if stage == "text_notebook":
                from scripts.defender_selection import save_notebook_selection
                save_notebook_selection(OUTPUT / "text_notebook_measurements.json",
                                        ROOT / "ai_content_detector/notebooks" / notebook,
                                        OUTPUT / "defender_comparison/notebook_selection.json")
        elif stage in COMPARISON_STAGES:
            from ai_content_detector.rl_evasion.defender_comparison.workflow import run
            from ai_content_detector.rl_evasion.checkpointing import CheckpointPause, StopRequest
            with StopRequest() as stop:
                run(stage, stop=stop)
                if stop.requested:
                    raise CheckpointPause("Comparison stage artifacts saved after scheduler warning")
        elif stage == "benchmark":
            arguments = [
                "--max-samples", "200",
                "--require-all-detectors",
                "--local-only",
                "--detectors", "binoculars", "fast_detect_gpt", "roberta_classifier",
                "--grpo-checkpoint", "results/text_evasion/grpo/final",
                "--multispin-checkpoint", "results/text_evasion/multispin/iteration_5",
            ]
            for dataset in ("hc3", "cnn_dailymail"):
                run_module("ai_content_detector.rl_evasion.benchmarking.benchmark", [
                    *arguments, "--dataset", dataset,
                    "--output-dir", str(OUTPUT / "benchmark" / dataset),
                ])
        else:
            run_module("ai_content_detector.rl_evasion.run_experiments", [
                "--experiment", stage,
            ])
    except BaseException as error:
        from ai_content_detector.rl_evasion.checkpointing import CheckpointPause
        paused = isinstance(error, CheckpointPause) or getattr(error, "ename", None) == "CheckpointPause"
        provenance["status"] = "paused" if paused else "failed"
        provenance["error"] = f"{type(error).__name__}: {error}"
        if paused:
            raise SystemExit(75) from error
        raise
    else:
        provenance["status"] = "completed"
    finally:
        provenance["finished_at"] = datetime.now(timezone.utc).isoformat()
        destination.write_text(json.dumps(provenance, indent=2) + "\n")
        attempt_path.write_text(json.dumps(provenance, indent=2) + "\n")
        update_report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=[
        *NOTEBOOKS, "grpo_text", "multispin", "ddpo_image", "benchmark",
        "arms_race", "meta_adapt", *COMPARISON_STAGES,
    ])
    run_stage(parser.parse_args().stage)
