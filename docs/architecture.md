# Implementation guide

The package keeps the `frontier_bench` import namespace and exposes the `budgeted-discovery` command. Its runtime is portable Python; the verified example consumes an exported world.

| Component | Responsibility |
|---|---|
| `world.py`, `schemas.py`, `config.py` | Typed artifacts, stable identities, configuration validation, checksums |
| `provider.py` | Causal lexical retrieval, graph expansion, pagination, availability, keyed failures, prices |
| `knowledge.py` | Observed evidence, candidate admission, query lineage, feedback state |
| `environment.py` | Gymnasium action table, request/cost constraints, time and feedback delivery |
| `policies.py` | Five headline strategies plus separate heuristic and online linear experimental policies |
| `runner.py` | Bounded execution, append-only logs, saved summaries, deterministic replay |
| `evaluation.py` | Truth-based evaluation and paired seed-block bootstrap comparisons |
| `reporting.py` | Checksum-verified offline HTML/SVG reports from recorded artifacts |
| `measurement.py` | Separate controlled temporal-measurement diagnostics |
| `generator.py`, `diffusion.py`, `content.py` | Optional synthetic-generation implementation; excluded from portable environment verification |

Data quality is part of the experiment contract. Policies can use only surfaced evidence; future posts and evaluator truth must not influence their action tables. A failed request differs from a successful zero-result request. Pagination and revisits retain query ancestry and incur explicit costs. Duplicate entities do not receive discovery credit twice.

The tests use hand-specified synthetic controls to check evidence isolation, causal retrieval, duplicate accounting, delayed assessment, action budgets, deterministic policy streams, exact replay, artifact tampering, CLI failure codes, and paired statistical units. These controls verify software properties independently of whether an acquisition strategy wins.

The archived main-grid contract narrows graph actions to neighbors/pages/revisits and round-robin rotation to graph/query. The generic package retains older defaults for compatibility. `scripts/demo.py` sets the main-grid choices explicitly; do not assume an arbitrary default CLI run is the headline experiment.
