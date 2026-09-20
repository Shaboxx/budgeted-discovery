# Evaluation: what the evidence supports

## Historical main grid

Execution was recorded on September 17, 2026 under policy contract 1.1.0. The complete main grid was 20 independent world seeds (201–220), three arrangements (concentrated, distributed, weak), five observation conditions, and five strategies: **1,500 completed runs; zero failed, skipped, or interrupted jobs**. Each seed × arrangement block shares the exact world content across all 25 strategy/condition combinations. There are 60 distinct world-content hashes.

The bundled `data/historical/summaries.json` contains one explicitly selected set of fields per recorded run. During release preparation, all 1,500 source run manifests and every artifact checksum listed by those manifests were checked. `scripts/verify_historical.py` independently verifies the released grid, identities, pairing, and budget bounds. This is an audit of historical evidence, not a claim that all simulations were rerun for the public release.

Worlds contain 120 generated accounts, four communities, eight time snapshots, generated text, and fixed reference-relevance labels. TADC-SBM supplies evolving topology; NDlib supplies diffusion inputs to synthetic content generation. The provider exposes partial observations; the evaluator alone uses reference truth. Operational feedback uses a profile-only text proxy with one-tick delay and 0.1 noise.

Every strategy has 24 requests, 32 cost units, and four request slots per tick. Initial exposure is shared and excluded from the incremental outcome. Request failures and follow-up pages consume budget. Pending assessments are not counted as negative feedback.

| Condition | Observation mechanism |
|---|---|
| Complete | Full response scope, no induced failures, no extra ranking noise or lag |
| Capped | Page size capped at five |
| Ranked | Cap plus ranking noise 0.05 and recency weight 0.15 |
| Delayed | Full scope, one-tick availability lag |
| Stress | Cap, ranking perturbation, one-tick lag, 5% keyed request failures |

## Hypotheses, baseline, and uncertainty

The primary metric is new unique reference-relevant accounts reached by the end of a run. Comparisons use paired worlds. The independent statistical unit is a world seed: arrangements and conditions are repeated measurements within that seed, not independent replications. Intervals use 2,000 percentile bootstrap resamples of the 20 seed blocks.

**H1:** `(graph − query) under stress − (graph − query) under complete observation` is −2.73 accounts, 95% interval [−4.35, −1.22]. The registered interaction is supported in this simulator. The capped mechanism contributes −3.73 [−5.37, −2.26] at the 98.3% confidence level used for the three mechanism comparisons. Ranking perturbation added no distinguishable effect beyond the cap in this experiment.

**H2:** the fixed mixture minus `max(graph, query)`, paired within each world and condition then averaged within seed, is +0.71 [−0.65, +2.02] in distributed-target worlds. The registered pooled complementarity hypothesis is **not supported**. Its complete-observation subgroup is +3.25 [+1.45, +5.10], while the stress subgroup is −1.25 [−2.55, 0.00]. A conditional subgroup benefit is not a universal winning strategy.

Graph-versus-query gaps by condition are +13.65 (complete), +9.92 (capped), +9.87 (ranked), +15.22 (delayed), and +10.92 (stress). These are differences in synthetic account counts, not percent improvements or business value. Random acquisition is a baseline; graph and query are the mixture's component baselines. The frozen mixture's graph probability of 0.75 came from development seeds 11 and 12, separate from the test seeds.

The supplied analysis script recreates the complete tables and intervals from the historical rows. See [the reproduced analysis](../reports/historical-analysis/analysis.md).

## Newly reproduced public example

`scripts/demo.py` reruns five strategies against the bundled seed-201, concentrated-target synthetic world under the stress observation condition. It uses the same declared acquisition contract, budgets, and frozen mixture weight. The example executes new decisions, checks receipts, writes Parquet curves, and replays all five runs. Its output is a small engineering demonstration; one world cannot estimate generalization or confidence intervals.

## Limitations and status

- All data and provider restrictions are synthetic. No live APIs, production costs, or observed business outcomes are represented.
- The small world size and vocabulary favor neighborhood expansion. Target arrangement matters little at this scale; a hypothesis requiring separated regions is weakly realized.
- Reference relevance is a constructed target. Operational text feedback is a different, noisy quantity.
- The separately implemented online linear policy and cost-aware heuristic are excluded from this historical main grid. Tests establish their software behavior, not outperformance.
- The released source includes the generation modules, but full TADC-SBM/NDlib regeneration needs an additional Linux environment with graph-tool and compatible generation dependencies. That environment is outside this portable release's verified quick start and CI. The bundled world avoids silently substituting a different generator.
- The historical release contains selected per-run summaries, not the full event archive. Full replay is demonstrated on newly produced public runs. Source hashes make code changes observable; timing and memory usage may vary by host.
- Policies see an allowlisted observation view, but arbitrary Python policy code is not security sandboxed.
