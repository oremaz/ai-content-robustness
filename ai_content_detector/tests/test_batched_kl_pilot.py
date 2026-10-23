import json
from types import SimpleNamespace

import pytest
import torch

from ai_content_detector.rl_evasion.config import TextEvasionConfig
from ai_content_detector.rl_evasion.text_evasion.batch_layout import (
    apply_layout, apply_trainer_layout, layout_identity, profile_pass, restore_layout, select_measurement,
)
from ai_content_detector.rl_evasion.text_evasion.schedules import warmup_stable_cosine
from scripts import architecture_races as races


def test_500_step_schedule_has_50_warmup_400_stable_and_50_cosine():
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=3e-5)
    scheduler = warmup_stable_cosine(optimizer, num_training_steps=500, num_warmup_steps=50,
                                    num_decay_steps=50, min_lr_ratio=1 / 3)
    rates = [scheduler.get_last_lr()[0]]
    for _ in range(500):
        optimizer.step(); scheduler.step(); rates.append(scheduler.get_last_lr()[0])
    assert rates[0] == 0
    assert rates[50:451] == [3e-5] * 401
    assert rates[475] == pytest.approx(2e-5)
    assert rates[500] == pytest.approx(1e-5)


def test_batched_submission_is_fresh_attacker_only_and_keeps_effective_32(tmp_path, monkeypatch):
    monkeypatch.setattr(races, "ROOT", tmp_path)
    monkeypatch.setattr(races, "OUTPUT", tmp_path)
    monkeypatch.setenv("USER", "fixture-user")
    monkeypatch.setattr(races, "inputs", lambda: ({}, {}))
    replies = iter(["", "201", "202", "203"])
    monkeypatch.setattr(races.subprocess, "check_output", lambda *args, **kwargs: next(replies))
    path = races.submit_batched_pilot(0.01, 32, 4)
    manifest = json.loads(path.read_text())
    protocol = manifest["protocol"]
    training = protocol["execution"]["attacker_training"]
    assert protocol["attacker_steps_per_round"] == 500 and protocol["initialization"] == "base_Qwen"
    assert manifest["pause_after_attacker_round"] == 1 and "continuation" not in manifest
    assert training["per_device_train_batch_size"] * training["gradient_accumulation_steps"] == 32
    assert training["grpo_num_generations"] == 4 and training["grpo_beta"] == 0.01
    assert protocol["execution"]["microbatch_candidates"] == [2, 4, 8, 16, 32]
    for job in manifest["jobs"].values():
        assert "--qos=qos_gpu_h100-dev" in job["command"] and "--time=02:00:00" in job["command"]
        assert "--requeue" in job["command"] and not any(arg.startswith("--dependency=") for arg in job["command"])


def test_profile_selection_and_restart_preserve_batch_and_reject_changed_beta(tmp_path):
    actor = SimpleNamespace(config=TextEvasionConfig(output_dir=str(tmp_path), grpo_beta=0.01,
                            per_device_train_batch_size=8, gradient_accumulation_steps=4))
    candidates = [2, 4, 8, 16, 32]
    identity = layout_identity(actor, candidates)
    measurements = [{"status": "out_of_memory", "microbatch": 32},
                    {"status": "insufficient_headroom", "microbatch": 16, "median_seconds": 1},
                    {"status": "ok", "microbatch": 8, "median_seconds": 2},
                    {"status": "ok", "microbatch": 4, "median_seconds": 3}]
    assert select_measurement(measurements) == 8
    with pytest.raises(RuntimeError):
        select_measurement(measurements[:2])
    (tmp_path / "microbatch_profile.json").write_text(json.dumps({"identity": identity, "selected_microbatch": 16}))
    restore_layout(actor, candidates)
    assert actor.config.per_device_train_batch_size == 16 and actor.config.gradient_accumulation_steps == 2
    assert layout_identity(actor, candidates) == identity
    restore_layout(actor, candidates)
    actor.config.grpo_beta = 0.02
    with pytest.raises(ValueError, match="profile differs"):
        restore_layout(actor, candidates)
    with pytest.raises(ValueError, match="must divide"):
        apply_layout(actor.config, 12, 32)


def tiny_grpo(tmp_path, dtype=torch.float32):
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, set_seed
    from trl import GRPOConfig, GRPOTrainer

    set_seed(42)
    vocabulary = {token: index for index, token in enumerate(["[PAD]", "[EOS]", "[UNK]", "a", "b", "c", "d", "e"])}
    raw = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]")); raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]")
    model = GPT2LMHeadModel(GPT2Config(vocab_size=8, n_positions=64, n_embd=16, n_layer=1, n_head=2,
                                      resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
                                      eos_token_id=1, pad_token_id=0, bos_token_id=1)).to(dtype=dtype)
    model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
                                            target_modules=["c_attn"], task_type="CAUSAL_LM"))
    args = GRPOConfig(output_dir=str(tmp_path), use_cpu=True, bf16=False, report_to="none",
                      per_device_train_batch_size=8, gradient_accumulation_steps=4,
                      num_generations=4, beta=0.01, max_completion_length=4,
                      gradient_checkpointing=False, max_steps=1, disable_tqdm=True, logging_steps=1)
    trainer = GRPOTrainer(model=model, args=args, processing_class=tokenizer,
                          train_dataset=Dataset.from_dict({"prompt": ["a b"] * 8}),
                          reward_funcs=lambda completions, **kwargs: [float(len(text)) for text in completions])
    return trainer


def test_native_grpo_applies_kl_to_a_fixed_base_reference_and_profiles_without_updates(tmp_path):
    from trl.trainer.utils import use_adapter

    trainer = tiny_grpo(tmp_path)
    model = trainer.model
    reference = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if ".ref." in name}
    assert reference and all(not parameter.requires_grad for name, parameter in model.named_parameters() if name in reference)
    prompt = torch.tensor([[3, 4], [4, 5]])
    completion = torch.tensor([[5, 6, 3, 4], [3, 4, 5, 6]])
    ids = torch.cat([prompt, completion], dim=1); mask = torch.ones_like(ids)
    with torch.no_grad(), use_adapter(model, "ref"):
        ref, _ = trainer._get_per_token_logps_and_entropies(model, ids, mask, 4)
        ref_logits = model(ids).logits
    with torch.no_grad(), model.disable_adapter():
        torch.testing.assert_close(ref_logits, model(ids).logits, rtol=0, atol=0)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B.default" in name:
                parameter.normal_(0, 0.5)
    inputs = {"prompt_ids": prompt, "prompt_mask": torch.ones_like(prompt),
              "completion_ids": completion, "completion_mask": torch.ones_like(completion),
              "advantages": torch.zeros(2), "ref_per_token_logps": ref,
              "num_items_in_batch": torch.tensor(8)}
    loss = trainer.compute_loss(model, inputs)
    assert loss.item() > 0 and trainer._metrics["train"]["kl"][-1] > 0
    assert loss.item() == pytest.approx(0.01 * trainer._metrics["train"]["kl"][-1], rel=1e-5)
    loss.backward()
    assert all(model.get_parameter(name).grad is None for name in reference)
    full_gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters()
                      if parameter.grad is not None}
    model.zero_grad(set_to_none=True)
    cfg = TextEvasionConfig(per_device_train_batch_size=1, gradient_accumulation_steps=2)
    apply_trainer_layout(trainer, cfg)
    for index in range(2):
        batch = {name: value[index:index + 1] if value.ndim else value for name, value in inputs.items()}
        trainer.accelerator.backward(trainer.compute_loss(model, batch))
    for name, expected in full_gradients.items():
        torch.testing.assert_close(model.get_parameter(name).grad, expected, rtol=2e-5, atol=1e-8)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    profile_pass(trainer, {key: value for key, value in inputs.items() if key != "ref_per_token_logps"}, 1)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    for name, value in reference.items():
        torch.testing.assert_close(model.get_parameter(name), value, rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_native_grpo_layout_preserves_native_loss_scaling_before_training(tmp_path):
    trainer = tiny_grpo(tmp_path)
    cfg = TextEvasionConfig(per_device_train_batch_size=16, gradient_accumulation_steps=2)
    apply_trainer_layout(trainer, cfg)
    assert trainer.args.generation_batch_size == 32 and trainer.args.steps_per_generation == 2
    assert trainer._train_batch_size == 16 and trainer.accelerator.gradient_accumulation_steps == 1
    trainer.train()
    assert trainer.state.global_step == 1
    assert trainer.args.per_device_train_batch_size * trainer.args.gradient_accumulation_steps == 32
    assert any("kl" in row for row in trainer.state.log_history)


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_restart_loads_root_policy_and_frozen_reference_subdirectory(tmp_path, dtype):
    from ai_content_detector.rl_evasion.text_evasion.reference import configure_reference_restart, reference_identity

    source = tiny_grpo(tmp_path / "source", dtype=dtype)
    identity = reference_identity(source.model)
    with torch.no_grad():
        for name, parameter in source.model.named_parameters():
            if "lora_B.default" in name:
                parameter.fill_(0.5)
    checkpoint = tmp_path / "checkpoint-1"
    source.model.save_pretrained(checkpoint)
    expected = {name: parameter.detach().clone() for name, parameter in source.model.named_parameters()}
    resumed = tiny_grpo(tmp_path / "resumed", dtype=dtype)
    cfg = TextEvasionConfig(output_dir=str(tmp_path / "resumed"), grpo_beta=0.01)
    configure_reference_restart(resumed, cfg)
    resumed._load_from_checkpoint(str(checkpoint))
    assert reference_identity(resumed.model) == identity
    for name, parameter in resumed.model.named_parameters():
        torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)
    assert all(not parameter.requires_grad for name, parameter in resumed.model.named_parameters() if ".ref." in name)
