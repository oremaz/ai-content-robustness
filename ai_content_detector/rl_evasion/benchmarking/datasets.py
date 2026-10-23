"""Data loading for evasion benchmarking.

Provides standardized datasets for evaluating evasion methods:
    - HC3 (Human ChatGPT Comparison Corpus)
    - CNN-DailyMail prompts (in-distribution with RL training)
    - Custom text files
"""

from __future__ import annotations

import json
import logging
import os
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkDataset:
    """Standard format for benchmark evaluation data.

    A dataset with ``ai_texts_available == False`` has no pre-generated AI
    samples — downstream benchmark methods that do *post-hoc* evasion on a
    given AI text (paraphrasing, synonym substitution, etc.) must be skipped
    on such datasets rather than silently fed empty strings.
    """

    name: str
    prompts: List[str]
    ai_texts: List[str]
    human_references: List[str] = field(default_factory=list)
    ai_texts_available: bool = True
    sample_ids: List[str] = field(default_factory=list)
    source_revision: Optional[str] = None

    def __post_init__(self) -> None:
        n = len(self.prompts)
        if n == 0:
            raise ValueError("BenchmarkDataset must contain at least one sample")
        if len(self.ai_texts) != n:
            raise ValueError(
                f"prompts and ai_texts must have equal lengths, got {n} and {len(self.ai_texts)}"
            )
        if self.human_references and len(self.human_references) != n:
            raise ValueError(
                "human_references must be empty or aligned one-to-one with prompts, "
                f"got {len(self.human_references)} and {n}"
            )
        if self.sample_ids and len(self.sample_ids) != n:
            raise ValueError("sample_ids must be empty or aligned with prompts")
        if not self.sample_ids:
            self.sample_ids = [f"{self.name}:{index}" for index in range(n)]
        if self.ai_texts_available and any(not text.strip() for text in self.ai_texts):
            raise ValueError("ai_texts_available=True requires a nonempty AI text for every sample")

    def __len__(self) -> int:
        return len(self.prompts)

    def subset(self, n: int) -> "BenchmarkDataset":
        return BenchmarkDataset(
            name=f"{self.name}[:{n}]",
            prompts=self.prompts[:n],
            ai_texts=self.ai_texts[:n],
            human_references=self.human_references[:n] if self.human_references else [],
            ai_texts_available=self.ai_texts_available,
            sample_ids=self.sample_ids[:n],
            source_revision=self.source_revision,
        )


def load_hc3(
    max_samples: int = 500,
    split: str = "train",
    data_file: Optional[str] = None,
) -> BenchmarkDataset:
    """Load HC3 (Human ChatGPT Comparison Corpus).

    Each sample has a question, a human answer, and a ChatGPT answer.
    This is ideal for benchmarking: the AI text is a direct ChatGPT
    response, and we have a ground-truth human reference.

    Dataset: Hello-SimpleAI/HC3 on HuggingFace.
    """
    from datasets import load_dataset

    logger.info("Loading HC3 dataset (split=%s, max=%d)...", split, max_samples)
    source = Path(data_file) if data_file is not None else Path(__file__).resolve().parents[3] / "data/hc3-all.jsonl"
    if not source.is_file():
        if data_file is not None:
            raise FileNotFoundError(source)
        from huggingface_hub import hf_hub_download
        source = Path(hf_hub_download("Hello-SimpleAI/HC3", "all.jsonl", repo_type="dataset"))
    snapshot_source = source
    if source.is_symlink() and source.parent.parent.name != "snapshots":
        snapshot_source = source.parent / os.readlink(source)
    revision = snapshot_source.parent.name if snapshot_source.parent.parent.name == "snapshots" else "local"
    source = source.resolve()
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    ds = load_dataset("json", data_files={"train": str(source)}, split=split)

    prompts = []
    ai_texts = []
    human_refs = []
    sample_ids = []

    for index, row in enumerate(ds):
        question = row.get("question", "")
        chatgpt_answers = row.get("chatgpt_answers", [])
        human_answers = row.get("human_answers", [])

        if not chatgpt_answers or not human_answers or not question:
            continue

        prompts.append(question)
        # Take first answer from each
        ai_texts.append(chatgpt_answers[0] if isinstance(chatgpt_answers[0], str) else str(chatgpt_answers[0]))
        human_refs.append(human_answers[0] if isinstance(human_answers[0], str) else str(human_answers[0]))
        sample_ids.append(f"HC3:{split}:{index}")

        if len(prompts) >= max_samples:
            break

    logger.info("HC3 loaded: %d samples", len(prompts))
    return BenchmarkDataset(
        name="HC3",
        prompts=prompts,
        ai_texts=ai_texts,
        human_references=human_refs,
        sample_ids=sample_ids,
        source_revision=f"{revision}:sha256:{digest.hexdigest()}",
    )


def load_cnn_dailymail(
    max_samples: int = 500,
    split: str = "test",
    prompt_words: int = 50,
) -> BenchmarkDataset:
    """Load CNN-DailyMail as prompt + continuation pairs.

    This matches the RL training setup: first ~50 words are the prompt,
    the rest is the "source" text. For benchmarking, the source acts as
    a human reference and we evaluate model-generated continuations.
    """
    from datasets import load_dataset

    logger.info("Loading CNN-DailyMail (split=%s, max=%d)...", split, max_samples)
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split=split)

    prompts = []
    ai_texts = []  # Will be filled during benchmarking (generated by each method)
    human_refs = []

    for row in ds:
        article = row["article"]
        words = article.split()
        if len(words) < prompt_words + 20:
            continue

        prompt = " ".join(words[:prompt_words])
        continuation = " ".join(words[prompt_words:])
        prompts.append(prompt)
        ai_texts.append("")  # Placeholder — benchmark runner generates these
        human_refs.append(continuation)

        if len(prompts) >= max_samples:
            break

    logger.info("CNN-DailyMail loaded: %d prompt-continuation pairs", len(prompts))
    return BenchmarkDataset(
        name="CNN-DailyMail",
        prompts=prompts,
        ai_texts=ai_texts,
        human_references=human_refs,
        # No pre-generated AI samples — post-hoc methods should be skipped.
        ai_texts_available=False,
        source_revision=f"abisee/cnn_dailymail:3.0.0:fingerprint:{ds._fingerprint}",
    )


def load_from_files(
    ai_file: str,
    human_file: Optional[str] = None,
    prompts_file: Optional[str] = None,
    max_samples: int = 500,
) -> BenchmarkDataset:
    """Load benchmark data from text files (one text per line).

    Args:
        ai_file: Path to file with AI-generated texts (one per line).
        human_file: Optional path to file with human references.
        prompts_file: Optional path to file with prompts.
        max_samples: Maximum number of samples to load.
    """
    def read_lines(path: str) -> List[str]:
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()][:max_samples]

    ai_texts = read_lines(ai_file)
    human_refs = read_lines(human_file) if human_file else []
    prompts = read_lines(prompts_file) if prompts_file else ai_texts

    if prompts_file and len(prompts) != len(ai_texts):
        raise ValueError(
            f"Prompt and AI files have different nonempty line counts: {len(prompts)} != {len(ai_texts)}"
        )
    if human_file and len(human_refs) != len(ai_texts):
        raise ValueError(
            f"Human and AI files have different nonempty line counts: {len(human_refs)} != {len(ai_texts)}"
        )

    return BenchmarkDataset(
        name=f"custom({os.path.basename(ai_file)})",
        prompts=prompts,
        ai_texts=ai_texts,
        human_references=human_refs,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


DATASET_LOADERS = {
    "hc3": load_hc3,
    "cnn_dailymail": load_cnn_dailymail,
}


def load_dataset_by_name(name: str, **kwargs) -> BenchmarkDataset:
    """Load a benchmark dataset by name."""
    if name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_LOADERS.keys())}")
    return DATASET_LOADERS[name](**kwargs)
