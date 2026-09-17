"""Preregistered analysis of the main grid (docs/main-grid-preregistration.md).

Reads an aggregate directory produced by scripts/aggregate_comparisons.py and writes:
  analysis.json  - primary H1 test (graph - query interaction, stress vs complete, 95%),
                   mechanism decomposition (capped/ranked/delayed, Bonferroni 98.3%),
                   H2 arrangement complementarity (95%), null checks, cell means.
  analysis.md    - the same as tables.
Nothing here selects metrics or contrasts after the fact; the rules are the registered ones.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from frontier_bench.evaluation import arrangement_summary, condition_summary, paired_summary


def fmt(x):
    return "n/a" if x is None else f"{x:+.2f}"


def interval(d):
    return "n/a" if d.get("interval") is None else f"[{d['interval'][0]:+.2f}, {d['interval'][1]:+.2f}]"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args(argv)
    rows = json.loads((Path(args.aggregate) / "summaries.json").read_text(encoding="utf-8"))
    completeness = json.loads((Path(args.aggregate) / "completeness.json").read_text(encoding="utf-8"))
    seeds = sorted({r["world_seed"] for r in rows})
    complete_rows = [r for r in rows if r["status"] == "completed"]
    conditions = sorted({r["condition"] for r in rows})
    reference = next(c for c in conditions if c.endswith("_complete"))

    # H1 primary and mechanism decomposition: graph - query interaction versus complete.
    gq95 = condition_summary(complete_rows, baseline="query", samples=args.samples, confidence=0.95, seed=args.seed, reference_condition=reference)
    gq983 = condition_summary(complete_rows, baseline="query", samples=args.samples, confidence=0.9833, seed=args.seed, reference_condition=reference)
    graph95 = next(c for c in gq95["contrasts"] if c["policy"] == "graph")
    graph983 = next(c for c in gq983["contrasts"] if c["policy"] == "graph")
    stress = next(c for c in conditions if c.endswith("_stress"))
    primary = graph95["interaction_vs_reference"][stress]
    primary_verdict = "supported" if primary["interval"] and (primary["interval"][0] > 0 or primary["interval"][1] < 0) else "not supported"
    mechanisms = {c: graph983["interaction_vs_reference"][c] for c in conditions if c not in (reference, stress)}
    # Paired SD for calibration.
    per_block = defaultdict(lambda: defaultdict(list))
    for r in complete_rows:
        if r["policy"] in ("graph", "query"):
            per_block[(r["world_seed"], r["condition"])][r["policy"]].append(r["reference_relevant_accounts"])
    diffs = defaultdict(list)
    for (seed, cond), v in per_block.items():
        if len(v.get("graph", [])) == 3 and len(v.get("query", [])) == 3:
            diffs[cond].append(sum(v["graph"]) / 3 - sum(v["query"]) / 3)
    import statistics as st
    paired_sd = {c: (st.pstdev(d) if len(d) > 1 else None) for c, d in diffs.items()}

    # H2: arrangement complementarity, and per-condition breakdown within distributed worlds.
    h2 = arrangement_summary(complete_rows, mixture="fixed", components=("graph", "query"), samples=args.samples, confidence=0.95, seed=args.seed)
    h2_by_condition = {}
    for cond in conditions:
        sub = [r for r in complete_rows if r["condition"] == cond]
        h2_by_condition[cond] = arrangement_summary(sub, mixture="fixed", components=("graph", "query"), samples=args.samples, confidence=0.95, seed=args.seed)["by_arrangement"]
    dist = h2["by_arrangement"].get("distributed", {}).get("fixed_minus_best_component", {})
    h2_verdict = "supported" if dist.get("interval") and dist["interval"][0] > 0 else "not supported"

    # Every strategy versus fixed per condition (descriptive) and versus random (null check).
    vs_fixed = condition_summary(complete_rows, baseline="fixed", samples=args.samples, confidence=0.95, seed=args.seed, reference_condition=reference)
    vs_random = condition_summary(complete_rows, baseline="random", samples=args.samples, confidence=0.95, seed=args.seed, reference_condition=reference)
    pooled = paired_summary(complete_rows, baseline="fixed", samples=args.samples, confidence=0.95, seed=args.seed)

    cells = defaultdict(list)
    for r in complete_rows:
        cells[(r["policy"], r["condition"], r["arrangement"])].append(r["reference_relevant_accounts"])
    cell_means = {f"{p}|{c}|{a}": sum(v) / len(v) for (p, c, a), v in cells.items()}
    route = defaultdict(lambda: defaultdict(int))
    for r in complete_rows:
        for fam, n in r.get("new_accounts_by_family", {}).items():
            route[r["policy"]][fam] += n

    result = {"seeds": seeds, "runs": len(rows), "completed_runs": len(complete_rows), "completeness": completeness,
              "reference_condition": reference, "H1_primary": {"contrast": "graph - query", "condition": stress, **primary, "verdict": primary_verdict},
              "H1_mechanisms_98.3": mechanisms, "graph_minus_query_by_condition_95": graph95["by_condition"], "paired_sd_graph_minus_query": paired_sd,
              "H2": {"arrangements": h2["by_arrangement"], "by_condition": h2_by_condition, "verdict_distributed": h2_verdict},
              "vs_fixed": vs_fixed, "vs_random": vs_random, "pooled_vs_fixed": pooled, "cell_means": cell_means, "new_accounts_by_family": route}
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")

    lines = [f"# Main grid analysis ({len(complete_rows)}/{len(rows)} completed runs, seeds {seeds[0]}-{seeds[-1]}, n={len(seeds)} blocks)", "",
             f"Reference condition: `{reference}`. Intervals: percentile bootstrap over world-seed blocks ({args.samples} resamples).", "",
             "## H1 primary: (graph - query) interaction, stress vs complete (95%)", "",
             f"mean {fmt(primary['mean'])}, interval {interval(primary)}, blocks {primary['seed_blocks']}, sign +{primary['positive']}/0{primary['zero']}/-{primary['negative']}, paired SD in stress {paired_sd.get(stress)} -> **{primary_verdict}**", "",
             "## H1 mechanisms: (graph - query) interaction vs complete (98.3%, Bonferroni over three)", "", "| condition | mean | interval | + / 0 / - |", "|---|---:|---|---|"]
    for c, d in mechanisms.items():
        lines.append(f"| {c} | {fmt(d['mean'])} | {interval(d)} | {d['positive']}/{d['zero']}/{d['negative']} |")
    lines += ["", "## graph - query by condition (95%)", "", "| condition | mean | interval | blocks |", "|---|---:|---|---:|"]
    for c, d in graph95["by_condition"].items():
        lines.append(f"| {c} | {fmt(d['mean'])} | {interval(d)} | {d['seed_blocks']} |")
    lines += ["", "## H2: fixed - best(graph, query) by arrangement (95%)", "", "| arrangement | fixed - graph | fixed - query | fixed - best | interval (best) | + / 0 / - |", "|---|---:|---:|---:|---|---|"]
    for a, d in h2["by_arrangement"].items():
        b = d["fixed_minus_best_component"]
        lines.append(f"| {a} | {fmt(d['fixed_minus_graph']['mean'])} | {fmt(d['fixed_minus_query']['mean'])} | {fmt(b['mean'])} | {interval(b)} | {b['positive']}/{b['zero']}/{b['negative']} |")
    lines.append(f"\nDistributed-arrangement verdict: **{h2_verdict}**")
    lines += ["", "## fixed - best(graph, query) in distributed worlds by condition (95%)", "", "| condition | mean | interval | + / 0 / - |", "|---|---:|---|---|"]
    for c, d in h2_by_condition.items():
        b = d.get("distributed", {}).get("fixed_minus_best_component", {})
        if b:
            lines.append(f"| {c} | {fmt(b['mean'])} | {interval(b)} | {b['positive']}/{b['zero']}/{b['negative']} |")
    lines += ["", "## Every strategy minus fixed, by condition (95%) and interaction vs complete", ""]
    for contrast in vs_fixed["contrasts"]:
        lines.append(f"**{contrast['policy']}**")
        lines += ["", "| condition | mean | interval | interaction mean | interaction interval |", "|---|---:|---|---:|---|"]
        for c, d in contrast["by_condition"].items():
            i = contrast["interaction_vs_reference"].get(c)
            lines.append(f"| {c} | {fmt(d['mean'])} | {interval(d)} | {fmt(i['mean']) if i else '—'} | {interval(i) if i else '—'} |")
        lines.append("")
    lines += ["## Cell means (new unique reference-relevant accounts)", "", "| policy | condition | " + " | ".join(sorted({a for (_, _, a) in cells})) + " |", "|---|---|" + "---:|" * len({a for (_, _, a) in cells})]
    arrangements = sorted({a for (_, _, a) in cells})
    for p in ("random", "graph", "query", "fixed", "round_robin"):
        for c in conditions:
            lines.append(f"| {p} | {c} | " + " | ".join(f"{cell_means.get(f'{p}|{c}|{a}', float('nan')):.2f}" for a in arrangements) + " |")
    lines += ["", "## New accounts by surfacing family (pooled)", "", "| policy | graph | query | other |", "|---|---:|---:|---:|"]
    for p, fam in route.items():
        lines.append(f"| {p} | {fam.get('graph', 0)} | {fam.get('query', 0)} | {fam.get('other', 0)} |")
    (out / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:12]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
