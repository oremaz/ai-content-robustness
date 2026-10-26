"""Write a concise, evidence-based summary without loading models."""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def atomic_text(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def report(path):
    manifest = json.loads(Path(path).read_text())
    root = Path(manifest["directory"])
    with (root / "summary.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        baseline = read_json(root / "baseline_validation/metrics.json")
        final = read_json(root / "test_results.json")
        best = read_json(root / "long/selected.json")
        latest = read_json(root / "long/latest_validation.json")
        lines = ["# SD1.5 DDPO evasion: long-run findings", ""]
        if final:
            initial, attacked = final["initial"], final["selected"]
            lines += ["**Final result:** the validation-selected attacker was evaluated against warmed DDA on held-out prompts.", ""]
            lines += [f"On {attacked['num_images']} untouched test prompts, evasion was **{attacked['evasion_rate']:.1%}** for the selected attacker versus **{initial['evasion_rate']:.1%}** for initial SD1.5. "
                      f"Mean CLIP alignment was {attacked['clip_mean']:.4f} versus {initial['clip_mean']:.4f}.", ""]
        elif baseline and best:
            n = baseline["num_images"]
            lines += [f"**Interim result:** the best checkpoint meeting the validation CLIP requirement reaches "
                      f"**{best['evasion_rate']:.1%} evasion** versus **{baseline['evasion_rate']:.1%}** initially "
                      f"({round(best['evasion_rate'] * n)}/{n} versus {round(baseline['evasion_rate'] * n)}/{n}). "
                      "Held-out confirmation is pending.", ""]
        else:
            lines += ["Results are pending; no conclusion about evasion is available yet.", ""]
        lines += ["The run trains SD1.5 UNet LoRA against warmed DDA for up to 12,800 generated images. "
                  "Reward: 0.64 evasion + 0.26 CLIP + 0.10 bounded reference similarity. "
                  "Checkpoint selection uses 200 validation prompts and requires at least 95% of initial mean CLIP alignment.", ""]
        if latest and best:
            long_status = (read_json(root / "long.json") or {}).get("status", "pending")
            lines += [f"**Training ({long_status}):** latest validation at {latest['images_seen']:,} images: "
                      f"**{latest['evasion_rate']:.1%} evasion**. "
                      + (f"It retains {latest['clip_mean'] / baseline['clip_mean']:.1%} of initial CLIP alignment and "
                         + ("meets" if latest["clip_eligible"] else "misses") + " the 95% selection requirement. " if baseline else "")
                      + f"The selected checkpoint is from {best['images_seen']:,} training images.", ""]
        diagnostic = read_json(root / "diagnostic_metrics.json")
        if diagnostic:
            lines += [f"Policy replay error was {diagnostic['max_replay_logprob_error']:.3g}; the initial policy gradient was nonzero. "
                      "These checks passed, but do not prove that every aspect of optimization is correct.", ""]
        lines += ["**Limits:** one seed and multiple checkpoints selected on a small validation set. "
                  + ("This result concerns the tested recipe and defender, not the impossibility of diffusion-model evasion."
                     if final else "The final 500-prompt held-out evaluation is pending. CLIP alignment is a proxy, not a guarantee of visual quality."), "",
                  "[Latest long-run validation](long/latest_validation.json) · [Selected checkpoint metrics](long/selected.json)", ""]
        atomic_text(root / "summary.md", "\n".join(lines))
        repository = Path(manifest["repository"])
        relative = (root / "summary.md").relative_to(repository)
        atomic_text(repository / "docs/image_attacker_ablation_summary.md",
                    "# SD1.5 attacker results\n\n[Concise findings](../" + str(relative) + ").\n")
        atomic_text(repository / "docs/image_attacker_ablation_results.md",
                    "# SD1.5 attacker results\n\n[Long-run findings](../" + str(relative) + ").\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    report(parser.parse_args().manifest)


if __name__ == "__main__":
    main()
