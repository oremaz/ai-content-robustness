"""Blind input boundaries, immutable artifacts and native clean-training behavior."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from ai_content_detector.watermarking.backbones import inject_lora, restore_generator, state_for_generator, use_backbones
from ai_content_detector.watermarking.protocol import file_hash, write_json
from ai_content_detector.watermarking.removal import (
    artifact_hashes, load_fitting, merge_state, text_bridge, validate_task, verify_artifact,
)


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def task(tmp_path):
    data = tmp_path / "fitting.json"
    write_json(data, {"train": [{"id": "a", "tokens": [1, 2, 3]}],
                      "validation": [{"id": "b", "tokens": [2, 3, 4]}]})
    output = tmp_path / "output"
    output.mkdir()
    return {"case": "model01", "method": "clean_full", "model": {"architecture": "causal_lm", "checkpoint": "/supplied/model"},
            "fitting": {"path": str(data), "sha256": file_hash(data)}, "output": str(output), "seed": 9,
            "source": "/source", "backbones": "/source/backbones", "code_sha256": {},
            "recipe": {"steps": 20, "export_steps": [5, 20], "mode": "full", "lr": 1e-5,
                       "accumulation": 1, "length": 32, "batch": 2, "lora_rank": 4}}


@pytest.mark.parametrize("field", ["clean_model", "key", "message", "detector", "scores", "owner"])
def test_no_clean_worker_rejects_reference_and_owner_fields(task, field):
    task[field] = {"checkpoint": "/private/model"} if field == "clean_model" else "private"
    with pytest.raises(ValueError):
        validate_task(task)


def test_nested_owner_configuration_is_rejected(task):
    task["model"]["config"] = {"nested": {"extractor": "/private/model"}}
    with pytest.raises(ValueError, match="Private evaluator"):
        validate_task(task)


def test_merge_requires_reference_and_a_fixed_bounded_sweep(task):
    task.update(method="merge", recipe={"weights": [.75, .5, .25, 0.]})
    with pytest.raises(ValueError, match="explicit compatible"):
        validate_task(task)
    task["clean_model"] = {"checkpoint": "/clean/model"}
    first = validate_task(task)
    task["recipe"]["weights"] = [.5]
    assert first != validate_task(task)
    task["recipe"]["weights"] = [1.5]
    with pytest.raises(ValueError):
        validate_task(task)


def test_training_bridge_contains_only_clean_fitting_rows(task):
    bridge = text_bridge(task)
    assert "qwen" not in bridge and "key" not in bridge and "resources" not in bridge
    assert bridge["clean_source"] == task["model"]["checkpoint"]
    data = json.loads(Path(bridge["data"]).read_text())
    assert set(data["text"]) == {"removal", "validation"}
    assert data["text"]["removal"][0]["id"] == "a"


def test_data_split_and_hash_contracts(task):
    path = Path(task["fitting"]["path"])
    data = json.loads(path.read_text())
    data["validation"][0]["id"] = "a"
    write_json(path, data)
    with pytest.raises(ValueError, match="changed"):
        load_fitting(task)
    task["fitting"]["sha256"] = file_hash(path)
    with pytest.raises(ValueError, match="leakage"):
        load_fitting(task)
    data["validation"][0]["id"] = "b"
    data["test"] = []
    write_json(path, data)
    task["fitting"]["sha256"] = file_hash(path)
    with pytest.raises(ValueError, match="only training and validation"):
        load_fitting(task)


def test_merging_normalization_and_exact_endpoints():
    marked = {"weight": torch.tensor([.12, -.23], dtype=torch.bfloat16), "counter": torch.tensor(2)}
    clean = {"weight": torch.tensor([.52, .17], dtype=torch.bfloat16), "counter": torch.tensor(2)}
    for weight in (.75, .5, .25):
        observed = merge_state(marked, clean, weight)
        expected = ((weight*marked["weight"].float()) + ((1-weight)*clean["weight"].float())).bfloat16()
        torch.testing.assert_close(observed["weight"], expected)
    for weight, expected in ((0., clean), (1., marked)):
        observed = merge_state(marked, clean, weight)
        for name in expected:
            assert torch.equal(observed[name], expected[name])
            assert observed[name].data_ptr() != expected[name].data_ptr()
    with pytest.raises(ValueError, match="incompatible"):
        merge_state(marked, {"different": torch.ones(2)}, .5)
    clean["counter"] = torch.tensor(4)
    with pytest.raises(ValueError, match="nonfloating"):
        merge_state(marked, clean, .5)


def test_artifact_seal_rejects_changed_or_added_files(tmp_path):
    (tmp_path / "model.pt").write_bytes(b"model weights")
    hashes = artifact_hashes(tmp_path)
    verify_artifact(tmp_path, hashes)
    (tmp_path / "extra.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        verify_artifact(tmp_path, hashes)


def test_owner_cannot_evaluate_until_all_attack_artifacts_are_frozen(task, tmp_path):
    spec = importlib.util.spec_from_file_location("removal_driver_test", ROOT / "scripts/watermark_removal.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = {"cases": {"model01": {"tasks": {}}}}
    for method in module.METHODS:
        current = copy.deepcopy(task)
        current["method"] = method
        if method == "merge":
            current["recipe"] = {"weights": [.5]}
            current["clean_model"] = {"checkpoint": "/clean/model"}
        else:
            current["recipe"]["mode"] = method.removeprefix("clean_")
        current["output"] = str(tmp_path / method)
        path = tmp_path / f"{method}.json"
        write_json(path, current)
        manifest["cases"]["model01"]["tasks"][method] = {"path": str(path), "sha256": file_hash(path)}
        write_json(Path(current["output"]) / "receipt.json", {"status": "running", "signature": validate_task(current)})
    with pytest.raises(ValueError, match="All attack artifacts"):
        module.collect_artifacts(manifest, "model01")


@pytest.mark.parametrize("completed", [False, True])
def test_submission_separates_worker_arguments_and_preserves_parent_lane(tmp_path, monkeypatch, completed):
    spec = importlib.util.spec_from_file_location("removal_submission_test", ROOT / "scripts/watermark_removal.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parent = tmp_path / "parent.json"
    write_json(parent, {"jobs": {"producer": {"job_id": "11"}, "old_removal": {"job_id": "12"}}})
    if completed:
        write_json(tmp_path / "producer.json", {"status": "completed", "job_id": "11"})
    manifest = {"directory": str(tmp_path), "jobs": {}, "qos": "qos_gpu_h100-t3",
                "parents": {"images": str(parent)}, "cases": {"model01": {
                    "parent": "images", "producer": "producer", "lane": "old_removal",
                    "tasks": {m: {"path": str(tmp_path / f"{m}.json")} for m in module.METHODS}}}}
    path = tmp_path / "private_owner.json"
    write_json(path, manifest)
    commands = []
    def submit(command, text):
        commands.append(command)
        return str(100+len(commands))
    monkeypatch.setattr(module.subprocess, "check_output", submit)
    monkeypatch.setattr(module, "report", lambda path: None)
    module.submit(path)
    assert len(commands) == 5
    expected = "afterany:12" if completed else "afterok:11,afterany:12"
    assert f"--dependency={expected}" in commands[0]
    for i, command in enumerate(commands[1:], 1):
        assert f"--dependency=afterok:{100+i}" in command
    for command in commands[1:4]:
        assert "--watermark-removal-attack" in command
        assert str(path) not in command
        assert "--hold" in command and "--qos=qos_gpu_h100-t3" in command
    assert "--watermark-removal-owner" in commands[-1]


def test_native_rar_clean_lora_and_full_backward_and_state_reload():
    upstream = ROOT / "results/research_workflow/watermarking/resources/distseal"
    if not upstream.exists():
        pytest.skip("Native image source unavailable")
    use_backbones(upstream)
    from omegaconf import OmegaConf
    from deps.rar.modeling.rar import RAR
    from deps.rar.modeling.modules.losses import ARLoss
    config = OmegaConf.load(upstream / "configs/generation/rar.yaml")
    config.model.generator.update({"hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 2,
                                   "intermediate_size": 64, "image_seq_len": 16, "condition_num_classes": 4,
                                   "use_checkpoint": False, "dropout": 0., "attn_drop": 0.})
    config.model.vq_model.codebook_size = 32
    torch.manual_seed(44)
    base = RAR(config).eval()
    tokens, labels = torch.randint(32, (2, 16)), torch.tensor([0, 3])
    for mode in ("lora", "full"):
        model = copy.deepcopy(base).requires_grad_(mode == "full")
        if mode == "lora":
            inject_lora(model, 4)
            with torch.no_grad():
                torch.testing.assert_close(model(tokens, model.preprocess_condition(labels)),
                                           base(tokens, base.preprocess_condition(labels)))
        model.train()
        logits, target = model(tokens, model.preprocess_condition(labels), return_labels=True)
        loss, _ = ARLoss(config)(logits, target)
        loss.backward()
        params = [p for p in model.parameters() if p.requires_grad]
        assert torch.isfinite(loss)
        assert sum(float(p.grad.abs().sum()) for p in params if p.grad is not None) > 0
        optimizer = torch.optim.AdamW(params, lr=1e-3, foreach=False)
        optimizer.step()
        restored = copy.deepcopy(base)
        if mode == "lora":
            inject_lora(restored, 4)
        restore_generator(restored, state_for_generator(model, "rar"), "rar")
        model.eval(); restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(tokens, restored.preprocess_condition(labels)),
                                       model(tokens, model.preprocess_condition(labels)))
