"""Evaluate a committed standalone attacker without resuming its training."""

from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from ai_content_detector.rl_evasion.checkpointing import CheckpointPause, StopRequest, load_checkpoint
from ai_content_detector.rl_evasion.config import TextEvasionConfig, seed_everything
from ai_content_detector.rl_evasion.defender_comparison import workflow
from ai_content_detector.rl_evasion.defender_comparison.protocol import digest, write_json


def checkpoint_metadata(group, checkpoint):
    checkpoint = Path(checkpoint).resolve()
    root = workflow.OUTPUT / "rl" / group
    if checkpoint.parent != root.resolve():
        raise ValueError("Evaluation checkpoint must belong to the selected standalone branch")
    if not (checkpoint / "restart_complete.pt").is_file():
        raise ValueError("Evaluation requires a fully committed checkpoint")
    context = load_checkpoint(root / "restart_context.pt")
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    if checkpoint.name != f"checkpoint-{state['global_step']}":
        raise ValueError("Checkpoint step differs from trainer state")
    config = TextEvasionConfig(**context["config"])
    files = ("adapter_model.safetensors", "adapter_config.json", "trainer_state.json")
    identity = {name: hashlib.sha256((checkpoint / name).read_bytes()).hexdigest() for name in files}
    manifest = json.loads((workflow.OUTPUT / "prompt_manifest.json").read_text())
    if context["eval_prompts"] != manifest["evaluation"]:
        raise ValueError("Saved attacker evaluation partition differs from the recorded protocol")
    metadata = {"group": group, "checkpoint": str(checkpoint), "global_step": state["global_step"],
                "scheduled_steps": state["max_steps"], "epoch": state["epoch"],
                "config": context["config"], "checkpoint_sha256": identity,
                "implementation_sha256": {
                    str(path.relative_to(workflow.ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (Path(__file__), Path(workflow.__file__),
                                 Path(workflow.__file__).with_name("protocol.py"),
                                 workflow.ROOT / "ai_content_detector/rl_evasion/text_evasion/grpo_trainer.py")},
                "evaluation_inputs_sha256": {
                    name: hashlib.sha256((workflow.OUTPUT / name).read_bytes()).hexdigest()
                    for name in ("base_pairs.json", "validation_thresholds.json", "prompt_manifest.json")}}
    return config, metadata


def evaluate(group, checkpoint, directory, stop=None):
    from peft import load_peft_weights, set_peft_model_state_dict

    directory = Path(directory)
    config, metadata = checkpoint_metadata(group, checkpoint)
    saved = directory / "manifest.json"
    if saved.exists() and json.loads(saved.read_text()) != metadata:
        raise ValueError("Checkpoint evaluation manifest differs from saved progress")
    write_json(saved, metadata)
    pool = workflow.group_pool(workflow.load_plan(), group, directory / "defenders")
    try:
        if digest(workflow.checkpoint_identity(pool.checkpoints)) != config.detector_checkpoint_hash:
            raise ValueError("Warmed defender identity differs from attacker training")
        result_path = directory / "results.json"
        if result_path.exists():
            return json.loads(result_path.read_text())
        seed_everything(config.seed)
        attacker = workflow.ComparisonAttacker(config, pool)
        attacker._setup_model()
        weights = load_peft_weights(str(checkpoint), device="cpu")
        loaded = set_peft_model_state_dict(attacker.model, weights)
        if loaded.unexpected_keys or any("lora_" in name for name in loaded.missing_keys):
            raise ValueError("Checkpoint adapter does not match the evaluation model")
        attacker.model.requires_grad_(False)
        attacker.model.eval()
        if stop is not None and stop.requested:
            raise CheckpointPause("Checkpoint evaluation paused after model setup")
        result = workflow.evaluate_attacker(attacker, pool, directory, f"checkpoint/{digest(metadata)}")
        result.update({"checkpoint": metadata, "completed_at": datetime.now(timezone.utc).isoformat()})
        write_json(result_path, result)
        return result
    finally:
        pool.clear()


def run(group, checkpoint, directory):
    receipt_path = Path(directory) / "receipt.json"
    previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    receipt = {"status": "running", "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
               "attempt": previous.get("attempt", 0) + 1,
               "started_at": datetime.now(timezone.utc).isoformat()}
    write_json(receipt_path, receipt)
    try:
        with StopRequest() as stop:
            evaluate(group, checkpoint, directory, stop=stop)
        from scripts.report_research import update_report
        update_report()
        receipt["status"] = "completed"
    except CheckpointPause as error:
        receipt.update(status="paused", error=str(error))
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(receipt_path, receipt)
    if receipt["status"] == "paused":
        if receipt["attempt"] >= 100:
            raise RuntimeError("Checkpoint evaluation reached its 100-attempt limit")
        if receipt["slurm_job_id"] is None:
            raise CheckpointPause(receipt["error"])
        subprocess.run(["scontrol", "requeue", receipt["slurm_job_id"]], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", choices=("encoder", "decoder", "ensemble"))
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("directory", type=Path)
    arguments = parser.parse_args()
    run(arguments.group, arguments.checkpoint, arguments.directory)
