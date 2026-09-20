# Public example: reproduced September 20, 2026

This is output from the released Python implementation running offline on the bundled synthetic seed-201 world: 120 accounts, eight time snapshots, concentrated target arrangement, stress observation condition. It is five newly executed runs, separate from the historical 1,500-run record.

```text
Five strategies completed; five event-log replays verified.
policy          relevant  requests  cost
random                10        24  28.0
graph                 12        24  24.0
query                  2        22  32.0
fixed                  8        24  28.0
round_robin            3        24  30.0
```

`relevant` is the count of newly reached unique reference-relevant synthetic accounts, excluding common initial exposure. This one-world ordering is illustrative and should not be substituted for the paired 20-seed historical analysis. The query strategy reaches the cost limit before using every request.

Run `python scripts/demo.py --output runs/demo` to reproduce the experiment and open its HTML report. The [saved HTML report](demo/report.html) includes request/cost curves, run summaries and links to all five saved manifests and raw event artifacts. Download the repository and open the HTML locally; GitHub's file viewer shows its source. The report is self-contained and makes no external requests.

The example uses the frozen graph weight 0.75 and separate operational profile feedback; evaluator truth remains unavailable to policy decisions. All five newly generated runs passed exact event-log replay. Source fingerprints and dependency versions appear in the saved manifests. Timing and sampled memory are host-dependent diagnostics, not performance benchmarks.

The portable verification suite covers the core provider, evidence isolation, cost accounting, delayed feedback, policy behavior, replay, report integrity and CLI behavior. Full native graph generation is not part of this portable check.
