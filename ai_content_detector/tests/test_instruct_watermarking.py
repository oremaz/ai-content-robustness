"""Instruction boundaries and TextSeal2 private detection checks."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest
import torch

from ai_content_detector.watermarking.instruct_text import (
    chat_prompt, prepare_questions, textseal2_config, textseal2_score,
)
from ai_content_detector.watermarking.text import forward_kl, response_mask


class Tokenizer:
    vocab_size = 1000

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return "USER:"+messages[0]["content"]+" ASSISTANT:"

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, tokens, **kwargs):
        return "".join(map(chr, tokens))


def test_native_chat_and_response_only_causal_loss():
    assert chat_prompt(Tokenizer(), "Question?") == "USER:Question? ASSISTANT:"
    mask = response_mask(7, 4, "cpu")
    assert mask.tolist() == [[False, False, False, True, True, True]]
    student = torch.randn(1, 6, 11, requires_grad=True)
    teacher = torch.randn_like(student)
    forward_kl(student, teacher, mask).backward()
    assert student.grad[:, :3].abs().sum() == 0
    assert student.grad[:, 3:].abs().sum() > 0
    with pytest.raises(ValueError):
        response_mask(4, 4, "cpu")


def test_question_and_response_splits_are_disjoint_and_repeatable(tmp_path):
    path = tmp_path/"questions.jsonl"
    rows = [{"question": f"Question {i}", "human_answers": [str(i).zfill(4)+"a"*270]} for i in range(400)]
    path.write_text("\n".join(json.dumps(x) for x in rows+rows))
    counts = dict(train=8, removal=8, validation=8, test=8)
    first = prepare_questions(path, Tokenizer(), 7, counts)
    assert first == prepare_questions(path, Tokenizer(), 7, counts)
    selected = [r for group in first["text"].values() for r in group]
    assert len({r["id"] for r in selected}) == 32
    assert len({tuple(r["human_tokens"]) for r in selected}) == 32
    for row in selected:
        assert row["tokens"][:row["prompt_length"]] == Tokenizer().encode(row["prompt"])
        assert len(row["tokens"])-row["prompt_length"] == 256


def test_textseal2_gamma_and_document_local_deduplication():
    source = Path(__file__).resolve().parents[2]/"results/research_workflow/watermarking/resources/textseal2"
    if not source.exists():
        pytest.skip("Released TextSeal2 source unavailable")
    sys.path.insert(0, str(source))
    from scipy.stats import gamma
    from textseal.watermarking.detector import build_detector
    config = textseal2_config()
    assert config.watermark_type == "textseal2" and config.repeated_windows == "skip"
    detector = build_detector(Tokenizer(), config)
    text = "abcd"*30
    scores = detector.get_scores_by_t([text], scoring_method="v2", seen_windows=set())[0]
    result = textseal2_score(detector, text)
    assert result == textseal2_score(detector, text)
    assert result["scored_pairs"] == len(scores) == 4
    assert result["p_value"] == pytest.approx(gamma.sf(sum(scores), len(scores)))
    assert textseal2_score(detector, "")["eligible"] is False


def test_both_experiments_wait_for_preflight_and_submission_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/"scripts"))
    import instruct_watermark_experiments as driver
    config = {"directory": str(tmp_path), "source": str(tmp_path/"source"),
              "input_hashes": {}, "frozen_hashes": {}}
    config["signature"] = driver.digest(config)
    path = tmp_path/"manifest.json"
    path.write_text(json.dumps(config))
    calls = []
    def submit(args, **kwargs):
        calls.append(args)
        return str(100+len(calls))+"\n"
    monkeypatch.setattr(driver.subprocess, "check_output", submit)
    driver.submit(path)
    assert [c[-1] for c in calls] == ["preflight", "textseal2", "latent"]
    assert not any(c.startswith("--dependency") for c in calls[0])
    assert all("--dependency=afterok:101" in c for c in calls[1:])
    driver.submit(path)
    assert len(calls) == 3
    assert driver.GENERATOR == "Qwen/Qwen3.5-4B"
    assert driver.REWRITER == "meta-llama/Meta-Llama-3.1-8B-Instruct"


def test_runtime_extracts_once_on_project_storage(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/"scripts"))
    import instruct_watermark_experiments as driver
    original = tmp_path/"original"
    (original/"bin").mkdir(parents=True)
    (original/"bin/python").symlink_to(sys.executable)
    with tarfile.open(tmp_path/"bira_runtime_py310.tar.gz", "w:gz") as archive:
        archive.add(original, arcname=".")
    monkeypatch.setattr(driver, "RESOURCES", tmp_path)
    python, signature = driver.prepare_runtime()
    assert python.is_file()
    assert python.parent.parent.parent == tmp_path
    monkeypatch.setattr(driver.subprocess, "run", lambda *a, **kw: pytest.fail("Runtime was extracted twice"))
    assert driver.prepare_runtime() == (python, signature)
    assert not list(tmp_path.glob("bira_runtime_build_*"))


def test_launcher_does_not_use_full_node_temporary_storage(tmp_path):
    project = Path(__file__).resolve().parents[2]
    (tmp_path/".venv/bin").mkdir(parents=True)
    (tmp_path/".venv/bin/python").symlink_to(sys.executable)
    manifest = tmp_path/"manifest.json"
    manifest.write_text(json.dumps({"source": str(tmp_path/"source"), "paper_python": sys.executable}))
    stub = tmp_path/"srun"
    stub.write_text('#!/usr/bin/env bash\nset -eu\ntest -d "$(dirname "$XDG_CACHE_HOME")"\nprintf "%s\\n" "$BIRA_PAPER_PYTHON" "$XDG_CACHE_HOME" > "$LAUNCH_RECEIPT"\n')
    stub.chmod(0o755)
    receipt = tmp_path/"launched.txt"
    env = {**os.environ, "PATH": str(tmp_path)+os.pathsep+os.environ["PATH"],
           "SLURM_SUBMIT_DIR": str(tmp_path), "SLURM_JOB_ID": "123",
           "SLURM_TMPDIR": str(tmp_path/"unavailable-node-disk"), "LAUNCH_RECEIPT": str(receipt)}
    result = subprocess.run(["bash", str(project/"scripts/run_instruct_watermarks.slurm"), str(manifest), "preflight"],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    lines = receipt.read_text().splitlines()
    assert lines[0] == sys.executable
    assert lines[1].startswith(str(tmp_path/"job-cache-123-"))
    assert not list(tmp_path.glob("job-cache-*"))
