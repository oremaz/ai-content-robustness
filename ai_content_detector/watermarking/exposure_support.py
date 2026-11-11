"""Maximize worst-case token evidence under a probability coverage constraint."""

import math

import torch


@torch.no_grad()
def maximin_support(probabilities, scores, k=10, coverage_loss=.1):
    """Select IDs only; retained probabilities are never altered or renormalized."""
    if probabilities.ndim != 1 or probabilities.shape != scores.shape or not len(probabilities):
        raise ValueError("Expected aligned nonempty probability and evidence vectors")
    if type(k) is not int or k < 1 or not math.isfinite(coverage_loss) or not 0 <= coverage_loss < 1:
        raise ValueError("Invalid exposure size or relative coverage loss")
    p, s = probabilities.double(), scores.double()
    if not torch.isfinite(p).all() or not torch.isfinite(s).all() or (p < 0).any() or p.sum() <= 0:
        raise ValueError("Invalid exposure probabilities or evidence")
    active = torch.where(p > 0)[0]
    k = min(k, len(active))
    baseline = p[active].topk(k).values.sum()
    target = (1-coverage_loss)*baseline
    order = active[s[active].argsort(descending=True, stable=True)]
    ordered = p[order]
    if k==len(active):
        prefix=ordered.cumsum(0)
    else:
        prefix = torch.zeros_like(ordered)
        for _ in range(k):
            prefix = torch.cummax(torch.cat((prefix.new_zeros(1), prefix[:-1]))+ordered, 0).values
    tolerance = 32*torch.finfo(torch.float64).eps*max(1., float(p.sum()))
    boundary = torch.where(prefix >= target-tolerance)[0][0]
    floor = s[order[boundary]]
    eligible = active[s[active] >= floor]
    selected = eligible[p[eligible].topk(min(k, len(eligible))).indices]
    retained = p[selected].sum()
    if retained < target-tolerance:
        raise FloatingPointError("Selected exposure violates its coverage budget")
    return selected, {"baseline_topk_mass": float(baseline), "retained_mass": float(retained),
                      "relative_mass_loss": max(0., float(1-retained/baseline)),
                      "score_floor": float(s[selected].min()), "coverage_loss_budget": coverage_loss,
                      "feasibility_tolerance": tolerance, "candidates": len(selected)}


def evidence_lower_bound(tokens, floors, in_support, window=3, tolerance=1.e-6):
    """V2 unique context/token accounting; unguarded positions contribute zero."""
    if len(tokens) != len(floors) or len(tokens) != len(in_support) or type(window) is not int or window < 1:
        raise ValueError("Evidence-path lengths or context width disagree")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Invalid evidence tolerance")
    if any(type(x) is not bool for x in in_support):
        raise ValueError("Support membership must be explicit Booleans")
    if any(x is not None and (not isinstance(x,(int,float)) or isinstance(x,bool) or not math.isfinite(x) or x < 0) for x in floors):
        raise ValueError("Invalid evidence floor")
    if any(member and floor is None for member,floor in zip(in_support,floors)):
        raise ValueError("A support member requires an evidence floor")
    seen, lower, guarded = set(), [], 0
    for position in range(window,len(tokens)):
        pair = tuple(tokens[position-window:position+1])
        if pair in seen:
            continue
        seen.add(pair)
        if in_support[position]:
            lower.append(max(0.,floors[position]-tolerance))
            guarded += 1
    return {"score_lower_bound":math.fsum(lower),"scored_pairs":len(seen),
            "guarded_scored_pairs":guarded,"per_token_tolerance":tolerance}
