"""Load the released BIRA classes without their unrelated API/detector imports."""

import ast
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, List, Optional


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def selected_nodes(path, names):
    nodes = []
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
            nodes.append(node)
    found = {n.name if hasattr(n, "name") else n.targets[0].id for n in nodes}
    if found != set(names):
        raise ValueError(f"Official source definitions missing: {set(names)-found}")
    return ast.Module(body=nodes, type_ignores=[])


def load_official(root):
    import numpy as np
    import torch
    import transformers
    import yaml
    from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer, BatchEncoding,
        LlamaTokenizerFast, LogitsProcessor, LogitsProcessorList, PreTrainedTokenizer, T5ForConditionalGeneration)
    root = Path(root)
    spec = importlib.util.spec_from_file_location("bira_paper_logger", root / "attack_utils/logger.py")
    logger = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(logger)
    namespace = dict(__name__="bira_released_runtime", np=np, torch=torch, transformers=transformers, yaml=yaml, os=os, sys=sys,
        json=json, time=time, re=__import__("re"), SimpleNamespace=SimpleNamespace,
        Any=Any, List=List, Optional=Optional, AutoModel=AutoModel, AutoModelForCausalLM=AutoModelForCausalLM,
        AutoTokenizer=AutoTokenizer, BatchEncoding=BatchEncoding, LlamaTokenizerFast=LlamaTokenizerFast,
        LogitsProcessor=LogitsProcessor, LogitsProcessorList=LogitsProcessorList,
        PreTrainedTokenizer=PreTrainedTokenizer, T5ForConditionalGeneration=T5ForConditionalGeneration,
        print=logger.print, ProgressLogger=logger.ProgressLogger)
    definitions = [(root/"attack_utils/utils.py", ["is_decoder_only_model", "is_chat_model", "HfModel"]),
                   (root/"attack_utils/attacks.py", ["SYSPROMPT", "SelfInformationCalculator",
                    "HighSurprisalLogitProcessor", "check_last_ngram_repeats", "WatermarkRemovalAttack", "BIRAAttack"])]
    for path, names in definitions:
        exec(compile(selected_nodes(path, names), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


@contextmanager
def local_models(models):
    import transformers
    classes = [transformers.AutoTokenizer, transformers.AutoModelForCausalLM,
               transformers.AutoModelForSequenceClassification, transformers.AutoModel,
               transformers.BertTokenizer, transformers.BertModel]
    originals = []
    for cls in classes:
        original = cls.from_pretrained
        originals.append((cls, original))
        def factory(original):
            def resolve(name, *args, **kwargs):
                name = models.get(str(name), {}).get("path", name)
                kwargs["local_files_only"] = True
                return original(name, *args, **kwargs)
            return resolve
        cls.from_pretrained = staticmethod(factory(original))
    try:
        yield
    finally:
        for cls, original in originals:
            cls.from_pretrained = original


def attack_args(model_config, beta=-4., vanilla=False, max_iter=20):
    return SimpleNamespace(attack_algorithms="vanilla_paraphrasing" if vanilla else "BIRA", backend="hf",
        model_cfg_path=str(model_config), api_model_name="unused", percentile=50, use_sampling=True,
        learning_rate=.125, max_iter=max_iter, beta=beta)


def qwen_chat_adapter(official, model):
    import types
    original = model._chatify
    def chatify(self, prompt, completion=None, sys_prompt=None):
        if completion is not None:
            return original(prompt, completion, sys_prompt)
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{sys_prompt}\n\n{prompt}"}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
    model._chatify = types.MethodType(chatify, model)
    namespace = official.HfModel._encode_batch.__globals__
    original_is_chat = namespace["is_chat_model"]
    namespace["is_chat_model"] = lambda name: name == "Qwen/Qwen3.5-9B" or original_is_chat(name)


class AttemptCache:
    """Replay completed upstream calls when a timeout interrupts a retry loop."""

    def __init__(self, attack, path, signature, stop, pause_error):
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList
        self.attack, self.path, self.signature = attack, Path(path), signature
        self.stop, self.pause_error = stop, pause_error
        self.calls, self.cursor, self.last = [], 0, None
        self.original = attack.gen_model.generate
        model = attack.gen_model.model
        original_generate = model.generate
        owner = self
        class StopGeneration(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return owner.stop.requested
        def capture(*args, **kwargs):
            kwargs["stopping_criteria"] = StoppingCriteriaList([StopGeneration()])
            result = original_generate(*args, **kwargs)
            if self.stop.requested:
                raise self.pause_error("Interrupted generation; completed attempts retained")
            prompt_length = kwargs["input_ids"].shape[-1]
            tokens = result[0, prompt_length:].tolist()
            eos = model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            self.last = {"generated_tokens": len(tokens), "length_capped": not any(t in eos for t in tokens),
                         "model_dtype": str(next(model.parameters()).dtype)}
            return result
        model.generate = capture
        attack.gen_model.generate = self.generate

    def begin(self, identity):
        self.cursor, self.last = 0, None
        self.identity = identity
        data = json.loads(self.path.read_text()) if self.path.exists() else None
        if data and data["signature"] != self.signature:
            raise ValueError("Attempt replay differs from the frozen protocol")
        if data and data["identity"] != identity:
            if data["identity"]["position"] < identity["position"]:
                data = None
            else:
                raise ValueError("Attempt replay differs from the frozen input")
        self.calls = data["calls"] if data else []

    def generate(self, prompts, *args, **kwargs):
        if self.stop.requested:
            raise self.pause_error("Completed attempts retained")
        processors = kwargs.get("logit_processors")
        beta = processors[0].beta if processors else 0.
        key = {"prompts": prompts, "beta": beta}
        if self.cursor < len(self.calls):
            item = self.calls[self.cursor]
            if item["key"] != key:
                raise ValueError("Upstream retry sequence changed")
            self.last = item["metadata"]
            result = item["result"]
        else:
            started = time.monotonic()
            result = self.original(prompts, *args, **kwargs)
            self.last["seconds"] = time.monotonic()-started
            self.calls.append({"key": key, "metadata": self.last, "result": result})
            save(self.path, {"signature": self.signature, "identity": self.identity, "calls": self.calls})
        self.cursor += 1
        return result
