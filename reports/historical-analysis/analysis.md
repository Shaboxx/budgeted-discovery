# Main grid analysis (1500/1500 completed runs, seeds 201-220, n=20 blocks)

Reference condition: `evolving_complete`. Intervals: percentile bootstrap over world-seed blocks (2000 resamples).

## H1 primary: (graph - query) interaction, stress vs complete (95%)

mean -2.73, interval [-4.35, -1.22], blocks 20, sign +4/01/-15, paired SD in stress 2.601014759238743 -> **supported**

## H1 mechanisms: (graph - query) interaction vs complete (98.3%, Bonferroni over three)

| condition | mean | interval | + / 0 / - |
|---|---:|---|---|
| evolving_capped | -3.73 | [-5.37, -2.26] | 2/0/18 |
| evolving_delayed | +1.57 | [-0.22, +3.47] | 11/1/8 |
| evolving_ranked | -3.78 | [-5.37, -2.26] | 2/0/18 |

## graph - query by condition (95%)

| condition | mean | interval | blocks |
|---|---:|---|---:|
| evolving_capped | +9.92 | [+8.78, +11.07] | 20 |
| evolving_complete | +13.65 | [+12.18, +15.13] | 20 |
| evolving_delayed | +15.22 | [+14.05, +16.50] | 20 |
| evolving_ranked | +9.87 | [+8.82, +10.95] | 20 |
| evolving_stress | +10.92 | [+9.78, +11.98] | 20 |

## H2: fixed - best(graph, query) by arrangement (95%)

| arrangement | fixed - graph | fixed - query | fixed - best | interval (best) | + / 0 / - |
|---|---:|---:|---:|---|---|
| concentrated | +0.58 | +12.68 | +0.50 | [-0.83, +1.85] | 10/2/8 |
| distributed | +0.71 | +12.95 | +0.71 | [-0.65, +2.02] | 14/0/6 |
| weak | +0.23 | +11.63 | +0.21 | [-0.68, +1.10] | 10/1/9 |

Distributed-arrangement verdict: **not supported**

## fixed - best(graph, query) in distributed worlds by condition (95%)

| condition | mean | interval | + / 0 / - |
|---|---:|---|---|
| evolving_capped | +0.20 | [-1.65, +2.25] | 9/0/11 |
| evolving_complete | +3.25 | [+1.45, +5.10] | 15/1/4 |
| evolving_delayed | +0.85 | [-1.05, +2.75] | 10/3/7 |
| evolving_ranked | +0.50 | [-1.40, +2.50] | 10/2/8 |
| evolving_stress | -1.25 | [-2.55, +0.00] | 6/1/13 |

## Every strategy minus fixed, by condition (95%) and interaction vs complete

**graph**

| condition | mean | interval | interaction mean | interaction interval |
|---|---:|---|---:|---|
| evolving_capped | -0.08 | [-1.13, +1.05] | +2.48 | [+1.08, +3.95] |
| evolving_complete | -2.57 | [-3.92, -1.17] | — | — |
| evolving_delayed | -0.17 | [-1.28, +0.97] | +2.40 | [+1.53, +3.35] |
| evolving_ranked | -0.25 | [-1.23, +0.85] | +2.32 | [+1.00, +3.78] |
| evolving_stress | +0.53 | [-0.25, +1.35] | +3.10 | [+1.93, +4.15] |

**query**

| condition | mean | interval | interaction mean | interaction interval |
|---|---:|---|---:|---|
| evolving_capped | -10.00 | [-11.05, -9.02] | +6.22 | [+4.87, +7.77] |
| evolving_complete | -16.22 | [-17.65, -14.73] | — | — |
| evolving_delayed | -15.38 | [-16.53, -14.12] | +0.83 | [-1.02, +2.75] |
| evolving_ranked | -10.12 | [-11.08, -9.12] | +6.10 | [+4.72, +7.38] |
| evolving_stress | -10.38 | [-11.45, -9.28] | +5.83 | [+4.08, +7.62] |

**random**

| condition | mean | interval | interaction mean | interaction interval |
|---|---:|---|---:|---|
| evolving_capped | -5.98 | [-7.15, -4.88] | +4.70 | [+3.22, +6.05] |
| evolving_complete | -10.68 | [-12.42, -8.87] | — | — |
| evolving_delayed | -10.12 | [-11.58, -8.38] | +0.57 | [-0.97, +2.25] |
| evolving_ranked | -5.80 | [-7.08, -4.57] | +4.88 | [+3.47, +6.30] |
| evolving_stress | -6.70 | [-7.80, -5.67] | +3.98 | [+2.32, +5.67] |

**round_robin**

| condition | mean | interval | interaction mean | interaction interval |
|---|---:|---|---:|---|
| evolving_capped | -1.23 | [-2.33, -0.10] | +1.05 | [-0.32, +2.52] |
| evolving_complete | -2.28 | [-3.55, -1.12] | — | — |
| evolving_delayed | -1.50 | [-2.62, -0.45] | +0.78 | [+0.10, +1.50] |
| evolving_ranked | -1.40 | [-2.52, -0.23] | +0.88 | [-0.53, +2.30] |
| evolving_stress | -1.92 | [-3.02, -0.82] | +0.37 | [-1.00, +1.68] |

## Cell means (new unique reference-relevant accounts)

| policy | condition | concentrated | distributed | weak |
|---|---|---:|---:|---:|
| random | evolving_capped | 8.30 | 6.45 | 7.50 |
| random | evolving_complete | 10.75 | 9.60 | 10.30 |
| random | evolving_delayed | 9.50 | 7.30 | 8.35 |
| random | evolving_ranked | 8.45 | 6.70 | 8.15 |
| random | evolving_stress | 5.90 | 5.20 | 5.50 |
| graph | evolving_capped | 13.40 | 13.65 | 12.90 |
| graph | evolving_complete | 18.60 | 17.50 | 18.90 |
| graph | evolving_delayed | 18.60 | 17.50 | 18.90 |
| graph | evolving_ranked | 13.40 | 13.65 | 12.90 |
| graph | evolving_stress | 12.75 | 13.50 | 12.05 |
| query | evolving_capped | 3.50 | 2.70 | 4.00 |
| query | evolving_complete | 5.45 | 2.80 | 5.80 |
| query | evolving_delayed | 2.20 | 4.10 | 3.05 |
| query | evolving_ranked | 3.55 | 2.75 | 4.05 |
| query | evolving_stress | 1.55 | 2.25 | 1.75 |
| fixed | evolving_capped | 13.00 | 13.85 | 13.35 |
| fixed | evolving_complete | 21.50 | 20.75 | 20.45 |
| fixed | evolving_delayed | 19.25 | 18.35 | 17.90 |
| fixed | evolving_ranked | 13.05 | 14.15 | 13.50 |
| fixed | evolving_stress | 12.85 | 12.25 | 11.60 |
| round_robin | evolving_capped | 13.00 | 11.45 | 12.05 |
| round_robin | evolving_complete | 18.70 | 18.15 | 19.00 |
| round_robin | evolving_delayed | 17.05 | 16.75 | 17.20 |
| round_robin | evolving_ranked | 13.10 | 11.25 | 12.15 |
| round_robin | evolving_stress | 10.80 | 10.05 | 10.10 |

## New accounts by surfacing family (pooled)

| policy | graph | query | other |
|---|---:|---:|---:|
| random | 4996 | 4587 | 0 |
| graph | 17612 | 0 | 0 |
| query | 0 | 4830 | 0 |
| fixed | 14572 | 2048 | 0 |
| round_robin | 11628 | 2785 | 0 |
