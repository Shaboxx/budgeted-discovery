# Data contracts and provenance

## Included data

`data/example-world/` is a canonical synthetic world generated for the September 17, 2026 experiment: seed 201, concentrated targets, evolving topology, 120 accounts, eight snapshots. All identifiers, profiles, posts, graph edges, and labels are generated. They do not describe real people. This file set is copied intact so its manifest checksums remain verifiable. No external dataset is redistributed.

`data/historical/summaries.json` is a field-limited projection of the 1,500 completed main-grid run summaries recorded on September 17, 2026. It retains experimental identifiers, paired-world hashes, status, outcome, request/cost totals, policy settings, and selected quality diagnostics. Runtime host information and archive filesystem locations are omitted. It contains no learned-policy or heuristic-policy runs.

`data/historical/completeness.json` records the verified execution scope. The public results are supplied under the repository's applicable data/license terms; third-party software notices apply to their respective components.

## Canonical world

| File | Contract |
|---|---|
| `world.json` | Schema version, account list, exact edge snapshots, generated posts, generator metadata and resolved configuration |
| `truth.json` | Evaluator-only reference-relevant account IDs, memberships, regions, diffusion states, target definition |
| `typed-events.jsonl` | One typed relationship per line with event/availability times; interaction rows expire at the next snapshot |
| `diagnostics.json` | Account, edge, isolation, component and degree checks for every snapshot |
| `manifest.json` | SHA-256 file checksums and separate world/content identities |

Account IDs must be unique. Edge endpoints and post authors must reference known accounts. Undirected edges cannot repeat within a snapshot; self-loops are excluded. Post availability cannot precede the event time. Explicit terms support time-causal retrieval. Loading rejects corrupt file checksums, invalid schema versions, and inconsistent identities. World-content hashes exclude experiment metadata so identical worlds can be paired across policies and observation conditions.

## Per-run outputs

The runner creates `manifest.json`, resolved configuration, initial exposure, `decisions.jsonl`, `executions.jsonl`, `outcomes.jsonl`, `evaluation.jsonl`, `curves.parquet`, `summary.json`, and final visible state. Requests include observation/execution status and cost. Later assessments revise recorded feedback without turning pending accounts into negatives. Evaluation truth is logged separately from learner feedback.

Replay verifies artifact checksums and reconstructs decisions, execution receipts, rewards, final visible state, and reference evaluation. Report generation checks raw artifacts before rebuilding derived tables; modifying a summary or evaluation trace causes verification to fail.

## Provenance and citations

The exported world metadata records TADC-SBM and NDlib generation settings and versions. The local runner uses [Gymnasium](https://gymnasium.farama.org/), [BM25S](https://github.com/xhluca/bm25s), [River](https://riverml.xyz/), [NetworkX](https://networkx.org/), [Apache Arrow](https://arrow.apache.org/), and [DuckDB](https://duckdb.org/). The optional generation implementation targets [TADC-SBM](https://pypi.org/project/tadc-sbm/) and [NDlib](https://ndlib.readthedocs.io/).

`nol_reference.py` contains an independent basic online-gradient implementation informed by [Tim LaRock's NOL reference](https://github.com/tlarock/nol), source revision `39c9ec8bc8e05a91c623511302978d2de479c0ff`. Its MIT attribution is retained in source and notices. It is not a reproduction of all NOL methods or published experiments. See [third-party notices](../THIRD_PARTY_NOTICES.md).
