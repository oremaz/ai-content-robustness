"""Paired watermark removal accounting with explicit preservation constraints."""

from __future__ import annotations

from collections import Counter
import math


def numeric_gate(metrics, bounds):
    """All declared metrics must be finite and lie in their inclusive interval."""
    failures = []
    for name, limits in bounds.items():
        low, high = limits
        if low is not None and high is not None and low > high:
            raise ValueError(f"Reversed preservation bounds: {name}")
        value = metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            failures.append(name + ":missing_or_nonfinite")
        elif low is not None and value < low or high is not None and value > high:
            failures.append(name + ":outside_bounds")
    if not bounds:
        raise ValueError("At least one preservation constraint is required")
    return {"passed": not failures, "failures": failures}


def joint_outcomes(originals, attacked, preservation):
    """Inputs map identical sample IDs to Boolean decisions and gate records."""
    if not originals or set(originals) != set(attacked) or set(originals) != set(preservation):
        raise ValueError("Original, attack and preservation sample IDs must match exactly")
    if any(type(value) is not bool for pool in (originals, attacked) for value in pool.values()):
        raise ValueError("Owner decisions must be explicit Booleans")
    for record in preservation.values():
        if type(record.get("passed")) is not bool or not isinstance(record.get("failures"), list):
            raise ValueError("Preservation requires Boolean passed and a failures list")
        if record["passed"] == bool(record["failures"]):
            raise ValueError("Preservation pass and failure reasons disagree")
    initial = {key for key, value in originals.items() if value}
    removed = {key for key in initial if not attacked[key]}
    passed = {key for key, value in preservation.items() if value["passed"]}
    initial_passed = initial & passed
    joint = removed & passed
    failures = Counter(reason for row in preservation.values() for reason in row["failures"])

    def fraction(numerator, denominator):
        return {"count": len(numerator), "n": len(denominator),
                "rate": len(numerator) / len(denominator) if denominator else None}

    return {"n": len(originals), "originally_detected": len(initial),
            "raw_removal": fraction(removed, initial),
            "preservation": fraction(passed, originals),
            "preservation_among_original_positives": fraction(initial_passed, initial),
            "joint_removal": fraction(joint, initial),
            "removal_given_preservation": fraction(joint, initial_passed),
            "preservation_failure_reasons": dict(sorted(failures.items())),
            "joint_success_ids": sorted(joint)}


def select_preserving_candidate(records, minimum):
    """Largest passing update count, then smallest anchor; no detector input."""
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 1:
        raise ValueError("A positive preservation count is required")
    names = set()
    eligible = []
    for row in records:
        if row["name"] in names:
            raise ValueError("Duplicate attack candidate")
        names.add(row["name"])
        step, weight, passed, count = row["step"], row["anchor_weight"], row["preserved"], row["n"]
        if type(step) is not int or step < 1 or type(passed) is not int or type(count) is not int or not 0 <= passed <= count:
            raise ValueError("Invalid attack update or preservation count")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
            raise ValueError("Invalid prediction-anchor weight")
        if passed >= minimum:
            eligible.append(row)
    return max(eligible, key=lambda r: (r["step"], -r["anchor_weight"])) if eligible else None


def preserving_fallback(originals, candidates, preservation):
    """Return an adapted output only when its declared quality gate passes."""
    if not originals or set(originals) != set(candidates) or set(originals) != set(preservation):
        raise ValueError("Fallback output and quality IDs must match exactly")
    selected, fallback_ids = {}, []
    for key, gate in preservation.items():
        if type(gate.get("passed")) is not bool or not isinstance(gate.get("failures"), list):
            raise ValueError("Fallback requires an explicit preservation gate")
        if gate["passed"] == bool(gate["failures"]):
            raise ValueError("Fallback pass and failure reasons disagree")
        selected[key] = candidates[key] if gate["passed"] else originals[key]
        if not gate["passed"]:
            fallback_ids.append(key)
    return selected, fallback_ids
