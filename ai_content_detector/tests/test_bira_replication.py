import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"scripts"))
from bira_official import AttemptCache, attack_args, load_official
from bira_replication import naturalness_passes


@pytest.fixture
def official():
    return load_official(ROOT/"results/research_workflow/watermarking/resources/bira")


def test_released_source_and_twenty_retry_backoff_are_used(official):
    assert official.BIRAAttack.rewrite.__code__.co_filename.endswith("bira/attack_utils/attacks.py")
    assert official.HfModel.generate.__code__.co_filename.endswith("bira/attack_utils/utils.py")
    attack = official.BIRAAttack(attack_args("unused"))
    namespace = official.BIRAAttack.rewrite.__globals__
    class Calculator:
        def __init__(self, model): pass
        def transform_tokens(self, text, threshold_low):
            assert threshold_low == 50
            return [2], ["word"]
    namespace["SelfInformationCalculator"] = Calculator
    betas = []
    class Generator:
        tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [1]*50)
        def generate(self, prompts, logit_processors):
            betas.append(logit_processors[0].beta)
            return ["repeat"]
    attack.gen_model = Generator()
    attack.aux_model = attack.gen_model
    text, meta = attack.rewrite("document")
    assert text == "repeat" and meta == {"generation_iter": 20, "is_degenerated": True}
    assert betas[0] == -4 and betas[-1] == -1.625


def test_released_surprisal_has_correct_causal_shift(official):
    ids = torch.tensor([[0, 1, 2, 3]])
    logits = torch.tensor([[[0., 10., 0., 0.], [0., 10., -10., 0.], [0., 0., 0., 10.], [0., 0., 10., 0.]]])
    class Encoding(dict):
        def to(self, device): return self
    class Tokenizer:
        def __call__(self, text, **kwargs):return Encoding(input_ids=ids)
        def decode(self, token):return str(token)
    model = SimpleNamespace(model=SimpleNamespace(device="cpu", __call__=None), tokenizer=Tokenizer())
    class Network:
        device = "cpu"
        def __call__(self, **kwargs):return SimpleNamespace(logits=logits)
    model.model = Network()
    calculator = official.SelfInformationCalculator(model)
    tokens, information = calculator.calculate_self_information("unused")
    assert tokens == [1, 2, 3] and information.index(max(information)) == 1
    assert calculator.transform_tokens("unused", 90)[0] == [2]
    scores = official.HighSurprisalLogitProcessor([[2]], -4)(ids, torch.zeros(1, 4))
    assert scores.tolist() == [[0., 0., -4., 0.]]


def test_released_generator_resets_seed_zero_each_call(official):
    namespace = official.HfModel.generate.__globals__
    namespace["ProgressLogger"] = SimpleNamespace(start=lambda *args: None, stop=lambda: None)
    generator = object.__new__(official.HfModel)
    generator.seed = 0
    generator.cfg = SimpleNamespace(name="meta-llama/Meta-Llama-3.1-8B-Instruct", response_max_len=1500,
        top_k=50, top_p=.95, use_sampling=True, sampling_temp=.7, length_penalty=1., no_repeat_ngram_size=0,
        repetition_penalty=1.)
    class Network:
        def generate(self, **kwargs):
            return torch.cat((kwargs["input_ids"], torch.randint(2, 20, (1, 8))), -1)
    generator.model = Network()
    generator.tokenizer = SimpleNamespace(eos_token_id=99, batch_decode=lambda rows, **kwargs: [str(rows.tolist())])
    generator._encode_batch = lambda prompts, **kwargs: (["prompt"], {"input_ids": torch.tensor([[1]])})
    generator._get_actual_input_str = lambda inputs, enc: inputs
    torch.manual_seed(21)
    first = generator.generate(["document"])
    torch.manual_seed(98)
    assert generator.generate(["document"]) == first


def test_attempt_replay_preserves_completed_compute_and_rejects_drift(tmp_path):
    class Paused(Exception):pass
    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight = torch.nn.Parameter(torch.zeros(1))
            self.generation_config = SimpleNamespace(eos_token_id=[99])
            self.calls = 0
        def generate(self, **kwargs):
            self.calls += 1
            return torch.tensor([[1, 2, 99]])
    class Generator:
        def __init__(self):self.model = Network()
        def generate(self, prompts, **kwargs):
            self.model.generate(input_ids=torch.tensor([[1]]))
            return ["completed"]
    stop = SimpleNamespace(requested=False)
    path = tmp_path/"attempts.json"
    attack = SimpleNamespace(gen_model=Generator())
    cache = AttemptCache(attack, path, "fixed", stop, Paused)
    identity = {"position": 0, "text": "doc"}
    cache.begin(identity)
    assert attack.gen_model.generate(["doc"]) == ["completed"]
    assert attack.gen_model.model.calls == 1
    resumed = SimpleNamespace(gen_model=Generator())
    replay = AttemptCache(resumed, path, "fixed", stop, Paused)
    replay.begin(identity)
    assert resumed.gen_model.generate(["doc"]) == ["completed"]
    assert resumed.gen_model.model.calls == 0
    assert replay.last["length_capped"] is False
    with pytest.raises(ValueError):replay.begin({"position": 0, "text": "changed"})
    replay.begin({"position": 1, "text": "next"})
    assert replay.calls == []


def test_twenty_sentence_gate_blocks_missing_and_garbled_examples():
    good = {"passed": True, "valid_judgment": True, "obvious_failure": False}
    rows = [dict(good) for _ in range(20)]
    gate = {"documents": 20, "minimum_fluent": 18}
    assert naturalness_passes(rows, gate)
    assert not naturalness_passes(rows[:19], gate)
    rows[0]["obvious_failure"] = True
    assert not naturalness_passes(rows, gate)
    rows[0] = dict(good, passed=False)
    rows[1] = dict(good, passed=False)
    assert naturalness_passes(rows, gate)
    rows[2] = dict(good, passed=False)
    assert not naturalness_passes(rows, gate)


def test_calibration_reuse_rejects_model_data_and_code_drift(tmp_path):
    import copy
    from bira_replication import QWEN, reuse_calibration
    configs = []
    for label in ("old", "new"):
        directory = tmp_path/label
        (directory/"source/scripts").mkdir(parents=True)
        (directory/"source/scripts/bira_replication.py").write_text("def calibrate_qwen(): return 1\ndef load_attack(): return 2\n")
        (directory/"qwen.yaml").write_text("identical")
        (directory/"data.json").write_text(json.dumps({"qwen_calibration": [{"id": "disjoint-c4"}]}))
        configs.append(dict(directory=str(directory), source=str(directory/"source"), seed=0,
            percentile=50, learning_rate=.125, max_iter=20, upstream_revision="pinned",
            models={QWEN: {"path": "clean-rewriter"}}, paper_data=str(directory/"data.json"),
            frozen_hashes={"upstream/attack.py": "fixed", "source/scripts/bira_official.py": "fixed"}, input_hashes={}))
    old, new = configs
    origin = Path(old["directory"])
    (origin/"qwen_beta.json").write_text(json.dumps({"beta": -5, "detector_queries": 0}))
    manifest = origin/"manifest.json"
    manifest.write_text(json.dumps(old))
    reuse_calibration(new, manifest)
    assert new["calibration_reuse"]["beta"] == -5
    assert str(Path(new["directory"])/"qwen_beta.json") in new["input_hashes"]
    changed = copy.deepcopy(new)
    changed["models"][QWEN]["path"] = "another-rewriter"
    with pytest.raises(ValueError, match="rewriter differs"):
        reuse_calibration(changed, manifest)
    Path(new["paper_data"]).write_text(json.dumps({"qwen_calibration": [{"id": "test-data"}]}))
    with pytest.raises(ValueError, match="documents differ"):
        reuse_calibration(new, manifest)
    Path(new["paper_data"]).write_bytes(Path(old["paper_data"]).read_bytes())
    (Path(new["source"])/"scripts/bira_replication.py").write_text("def calibrate_qwen(): return 3\ndef load_attack(): return 2\n")
    with pytest.raises(ValueError, match="driver differs"):
        reuse_calibration(new, manifest)


def test_student_attacks_ignore_failed_standalone_quality_diagnostic(tmp_path, monkeypatch):
    import bira_replication as driver
    config = {"directory": str(tmp_path), "signature": "fixed", "cases": [{"kind": "latent", "name": "student", "runtime": "main"}]}
    calls = []
    (tmp_path/"naturalness_gate.json").write_text(json.dumps({"signature": "fixed", "passed": False}))
    def child(config, manifest, phase, runtime, case_index=None):
        calls.append(phase)
    monkeypatch.setattr(driver, "child", child)
    monkeypatch.setattr(driver, "verify", lambda config: None)
    monkeypatch.setattr(driver.signal, "signal", lambda *args: None)
    driver.run(config, "unused", "auxiliary", None)
    assert calls == ["calibrate"]
    driver.run(config, "unused", "case", 0)
    assert calls == ["calibrate", "probabilities", "vanilla", "bira", "owner", "quality"]
    receipt = json.loads((tmp_path/"student/receipt.json").read_text())
    assert receipt["status"] == "completed"


@pytest.mark.parametrize("quality_only", [False, True])
def test_student_only_submission_uses_completed_preflight_without_expired_dependency(tmp_path, monkeypatch, quality_only):
    import bira_replication as driver
    (tmp_path/"source/scripts").mkdir(parents=True)
    (tmp_path/"source/scripts/run_bira_replication.slurm").write_text("case-loop")
    for runtime in ("paper", "main"):
        (tmp_path/(runtime+"_preflight_receipt.json")).write_text(json.dumps({"status": "completed"}))
    manifest = tmp_path/"manifest.json"
    manifest.write_text(json.dumps({"directory": str(tmp_path), "source": str(tmp_path/"source"),
        "cases": [{"kind": "latent"}], "quality_only": quality_only, "jobs": {"preflight": "old-purged-job"}}))
    submissions = []
    def submit(args, **kwargs):
        submissions.append(args)
        return str(100+len(submissions))+"\n"
    monkeypatch.setattr(driver.subprocess, "check_output", submit)
    driver.submit(manifest)
    assert len(submissions) == (1 if quality_only else 2)
    assert not any(arg.startswith("--dependency") for arg in submissions[0])
    if quality_only:
        assert submissions[0][-1] == "quality"
    else:
        assert "--dependency=afterok:101" in submissions[1]
        assert submissions[1][-2:] == ["case", "0"]


def test_base_quality_control_selects_original_weights_and_cannot_be_attack_target():
    import bira_replication as driver
    config = {"latent_owner": {"directory": "/trained", "distilled_model": "student",
                              "qwen": {"path": "/original/base"}}}
    assert driver.naturalness_model(config) == Path("/trained/student")
    config["quality_model"] = "base"
    assert driver.naturalness_model(config) == Path("/original/base")
    config["quality_model"] = "teacher"
    assert driver.naturalness_model(config) == Path("/original/base")
    with pytest.raises(ValueError, match="quality-only control"):
        driver.prepare(quality_model="base")
    config["quality_model"] = "unknown"
    with pytest.raises(ValueError, match="Unknown quality-control model"):
        driver.naturalness_model(config)


def test_probability_sampler_uses_original_frozen_generation_recipe(tmp_path):
    from bira_replication import original_sampling
    source = tmp_path/"ai_content_detector/watermarking"
    source.mkdir(parents=True)
    (source/"latent_text.py").write_text("def generation_config(manifest):\n    return {'seed': manifest['seed'], 'batch': 8, 'top_p': 1., 'temperature': manifest['latent_training']['temperature']}\n")
    manifest = {"source": str(tmp_path), "seed": 123, "latent_training": {"temperature": .8}}
    config = original_sampling(manifest)
    assert config == {"seed": 123, "batch": 8, "top_p": 1., "temperature": .8, "max_tokens": 256}
    manifest["latent_training"]["temperature"] = .7
    assert original_sampling(manifest)["temperature"] == .7


@pytest.mark.parametrize("kind", ["base", "student", "teacher"])
def test_quality_watermark_hook_is_exclusive_to_teacher_and_removed_on_failure(monkeypatch, kind):
    from contextlib import contextmanager
    import bira_replication as driver
    from ai_content_detector.watermarking import latent_text
    manifest = {"directory": "/owner", "latent_training": {"message": [0, 1]}}
    config = {"quality_model": kind, "latent_owner": manifest}
    model, embedder = object(), object()
    events = []
    def load_pair(owner):
        assert owner is manifest
        events.append("load")
        return embedder, object(), {"step": 5000}
    @contextmanager
    def embedded(network, module, message):
        assert network is model and module is embedder and message == [0, 1]
        events.append("install")
        try:
            yield
        finally:
            events.append("remove")
    monkeypatch.setattr(latent_text, "load_pair", load_pair)
    monkeypatch.setattr(latent_text, "embedded", embedded)
    with pytest.raises(RuntimeError, match="interrupted"):
        with driver.naturalness_context(config, model) as info:
            assert info.get("step") == (5000 if kind == "teacher" else None)
            assert events == (["load", "install"] if kind == "teacher" else [])
            raise RuntimeError("interrupted")
    assert events == (["load", "install", "remove"] if kind == "teacher" else [])
