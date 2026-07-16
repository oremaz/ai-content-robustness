# AGENTS.md

You are a senior machine learning research engineer. This repository combines tabular and multimodal ML pipelines, nutrition applications, and AI-generated content detection and evasion research.

## Project structure

* `ml_pipeline/` contains reusable training, augmentation, benchmarking, and checkpoint utilities.
* `nut_agent/` contains the NutriCoach and Recipe Lab Streamlit applications.
* `ai_content_detector/` contains text and image detectors, adversarial-evasion research, and a Streamlit scoring application.
* Read the README in the relevant subproject before changing it. Keep shared functionality in the existing common modules rather than duplicating it across subprojects.

## Environment

* Experiments run on a remote VM; local visibility may be incomplete.
* Do not assume missing files, datasets, checkpoints, or logs mean they do not exist.
* Use the `uv` environment in this repo. Dependency files live at the repository root.
* Run smoke tests from the repo root with `PYTHONPATH=.`.

Example:

```bash
PYTHONPATH=. uv run python ...
```

## Working rules

* Make minimal, targeted changes.
* Preserve existing architecture, naming, and style.
* Do not refactor unrelated code.
* Do not leave temporary test files, scripts, logs, caches, or outputs.
* If VM-only resources prevent testing, say exactly what could not be tested.
* Never write em dashes.
* Before implementing new functionality, search the relevant subproject and shared utilities for an existing implementation.
* If you fetch a research paper, download and read the full PDF before describing or implementing its method.
* Keep the relevant subproject README synchronized with user-facing features, commands, dependencies, and validated test results.

## Code style

* Keep comments sparse.
* Do not add obvious or systematic comments.
* Explain only non-trivial logic or math.
* Do not add a comment to explain each of the modification I ask you.

## Performance

* Prefer fast, vectorized implementations.
* Avoid unnecessary tensor copies, device transfers, and large intermediate tensors.
* If speed is similar, choose the lower-memory approach.
* If a higher-memory approach is much faster and still reasonable, prefer it.

## Research correctness

Be especially careful with data leakage, train/validation/test splits, preprocessing consistency, label and class ordering, tensor shapes, masking, padding, timestep indexing, probability versus log-probability calculations, sampling schedules, and tokenizer-dependent lengths. Preserve reproducibility through the project's existing seed, configuration, and checkpoint conventions.

## Paper and TeX summaries

Maintain mathematical rigor with a pedagogical approach. Define notation locally, explain frameworks intuitively, and use brief concrete examples when they clarify a critical point. Keep summaries concise, faithful to the full paper, and explicit about distinctions between the paper and this repository's implementation.

## Validation

* Run the narrowest relevant tests first with `PYTHONPATH=. uv run python -m pytest ...` from the repository root.
* Expand testing when a change affects shared modules or multiple subprojects.
* Do not update test counts in documentation unless they were verified in the current checkout.
* Streamlit applications should at least pass import or startup smoke checks when their dependencies and resources are locally available.

## Final response

End with:

* what changed
* what was tested
* limitations, if any
