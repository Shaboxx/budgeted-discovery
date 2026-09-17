"""Separate observation-coverage diagnostics, never acquisition reward.

Fixtures are explicitly deterministic controls; they do not replace TADC-SBM or
NDlib acceptance. Actual SimulatedProvider receipts drive observed counts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .provider import SimulatedProvider
from .schemas import Action, World, canonical, digest

MEASUREMENT_VERSION = "1.0.0"
PANEL = ("acct:000", "acct:001")
SCENARIOS = ("coverage_expansion", "ranked_cap_change", "delayed_indexing_backfill", "genuine_activity_change")


@dataclass(frozen=True)
class WindowCount:
    start: int
    end: int
    count: int
    panel: tuple[str, ...]
    expected_exposures: int
    complete_exposures: int
    interpretation: str = "all_panel_posts_v1"
    statuses: tuple[str, ...] = ()
    watermark_complete: bool = True


def compare_windows(prior: WindowCount, recent: WindowCount, threshold: float = 0.25,
                    minimum_exposures: int = 2) -> dict:
    """Compare the same panel estimand, explicitly retaining unknowns and zero.

    Completeness is a declared provider/lag condition, not inferred from a high
    observed post count. These checks do not establish population representation.
    """
    result = {"prior_count": prior.count, "recent_count": recent.count, "growth_ratio": None,
              "state": "comparable", "qualified_alert": False, "alert_kind": None,
              "threshold": threshold, "estimand": "post count within frozen panel and event window",
              "prior": asdict(prior), "recent": asdict(recent)}
    if prior.panel != recent.panel or prior.interpretation != recent.interpretation:
        result.update(state="not_comparable", reason="panel membership or interpretation changed")
    elif prior.end - prior.start != recent.end - recent.start or prior.end >= recent.start:
        result.update(state="not_comparable", reason="event windows overlap or have different durations")
    elif "partial" in prior.statuses + recent.statuses:
        result.update(state="partial", reason="one or more returned scopes were truncated")
    elif not prior.watermark_complete or not recent.watermark_complete:
        result.update(state="not_comparable", reason="indexing completeness watermark has not passed")
    elif prior.complete_exposures < minimum_exposures:
        result.update(state="insufficient_baseline", reason="too few complete baseline observations")
    elif (prior.complete_exposures < prior.expected_exposures or
          recent.complete_exposures < recent.expected_exposures):
        result.update(state="partial", reason="one or more expected panel exposures are missing")
    elif prior.count == 0:
        # A complete observed zero is different from an unobserved baseline. An
        # emergence alert is possible, but no percentage growth is manufactured.
        result.update(state="zero_baseline", reason="complete baseline observed zero posts")
        if recent.count >= 2:
            result.update(qualified_alert=True, alert_kind="new_activity_after_observed_zero")
    else:
        ratio = (recent.count - prior.count) / prior.count
        result.update(growth_ratio=ratio, qualified_alert=ratio >= threshold,
                      alert_kind="panel_activity_increase" if ratio >= threshold else None,
                      reason="same frozen panel, interpretation, windows, and complete response scope")
    return result


def deterministic_fixture(scenario: str) -> World:
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown diagnostic scenario {scenario!r}")
    accounts = [{"id": f"acct:{i:03d}", "profile": f"Synthetic maker {i}", "attributes": {}, "available_at": 0}
                for i in range(6)]
    posts = []
    for tick in range(4):
        for account in accounts:
            rate = 2 if scenario == "genuine_activity_change" and tick >= 2 else 1
            for ordinal in range(rate):
                available = tick + 2 if scenario == "delayed_indexing_backfill" and tick < 2 else tick
                posts.append({"id": f"post:{account['id']}:{tick}:{ordinal}", "author": account["id"],
                              "event_at": tick, "available_at": available,
                              "text": "craft table workshop", "terms": ["craft", "table", "workshop"],
                              "hashtags": ["craft"], "mentions": []})
    return World(accounts, [[] for _ in range(6)], posts,
                 {"reference_relevance": {a["id"]: True for a in accounts},
                  "change_tick": 2 if scenario == "genuine_activity_change" else None},
                 {"generator": "deterministic-measurement-fixture-v1", "scenario": scenario,
                  "not_tadc_or_ndlib": True, "frozen_panel": list(PANEL),
                  "event_windows": [[0, 1], [2, 3]]})


def _provider_config(scenario: str) -> dict:
    return {"mode": "restricted" if scenario == "ranked_cap_change" else "complete",
            "page_size": 2, "failure_probability": 0.0, "unavailable_operations": [],
            "ranking_noise": 0.2 if scenario == "ranked_cap_change" else 0.0,
            "recency_weight": 0.0, "latency": 0,
            "costs": {op: 1.0 for op in ("inspect", "posts", "neighbors", "search", "page", "revisit")}}


def _collect(world: World, scenario: str, arm: str, seed: int) -> dict:
    provider = SimulatedProvider(world, _provider_config(scenario), seed)
    records, unique_posts = [], {}
    calls_per_tick = 2
    lag = 2 if scenario == "delayed_indexing_backfill" and arm == "panel" else 0
    expected_max_index_delay = 2 if scenario == "delayed_indexing_backfill" else 0
    for event_tick in range(4):
        observation_tick = event_tick + lag
        for slot in range(calls_per_tick):
            if arm == "panel":
                action = Action("posts", subject=PANEL[slot], since=event_tick, until=event_tick, limit=2)
            elif scenario == "coverage_expansion":
                index = 0 if event_tick < 2 else (event_tick - 2) * 2 + slot
                action = Action("posts", subject=f"acct:{index:03d}", since=event_tick, until=event_tick, limit=2)
            elif scenario == "ranked_cap_change":
                action = Action("search", query=("craft",), since=event_tick, until=event_tick,
                                limit=1 if event_tick < 2 else 2)
            elif scenario == "delayed_indexing_backfill":
                # Expanding the lookback explicitly retrieves newly indexed old
                # events. Their publication timestamps are never rewritten.
                action = Action("posts", subject=PANEL[slot], since=0, until=event_tick, limit=20)
            else:
                action = Action("posts", subject=PANEL[slot], since=event_tick, until=event_tick, limit=2)
            receipt = provider.execute(action, observation_tick)
            new_ids = []
            for post in receipt.posts:
                if post["id"] not in unique_posts:
                    unique_posts[post["id"]] = {**post, "first_observed_at": receipt.response_tick}
                    new_ids.append(post["id"])
            records.append({"scenario": scenario, "arm": arm, "event_tick": event_tick,
                            "slot": slot, "action": action.to_dict(), "receipt": receipt.to_dict(),
                            "new_post_ids": new_ids,
                            "index_watermark_complete": observation_tick >= event_tick + expected_max_index_delay})
    raw_counts = [sum(1 for p in unique_posts.values() if start <= p["first_observed_at"] <= end)
                  for start, end in ((0, 1), (2, 3))]
    event_counts = [sum(1 for p in unique_posts.values() if start <= p["event_at"] <= end)
                    for start, end in ((0, 1), (2, 3))]
    return {"records": records, "posts": list(unique_posts.values()), "raw_observation_counts": raw_counts,
            "event_time_counts": event_counts, "requests": len(records),
            "cost": sum(r["receipt"]["cost"] for r in records), "fixed_observation_lag": lag}


def _window_from_collection(collection: dict, start: int, end: int, panel: tuple[str, ...]) -> WindowCount:
    records = [r for r in collection["records"] if start <= r["event_tick"] <= end]
    # This construction is for the scheduled panel: exactly one declared request
    # per (member,event tick). Repeated hits do not masquerade as extra exposures.
    complete = {(r["action"]["subject"], r["event_tick"]) for r in records
                if r["receipt"]["execution_status"] == "completed"
                and r["receipt"]["observation_status"] in ("complete", "observed_zero")}
    count = sum(1 for post in collection["posts"] if post["author"] in panel and start <= post["event_at"] <= end)
    return WindowCount(start, end, count, panel, len(panel) * (end - start + 1), len(complete),
                       statuses=tuple(r["receipt"]["observation_status"] for r in records),
                       watermark_complete=all(r["index_watermark_complete"] for r in records))


def _truth_count(world: World, panel: tuple[str, ...], start: int, end: int) -> int:
    # Evaluation-only reference for exactly the same frozen panel estimand.
    return sum(1 for p in world.posts if p["author"] in panel and start <= p["event_at"] <= end)


def diagnostic_suite(output_dir: str | Path, config: dict[str, Any] | None = None) -> dict:
    config = config or {}
    evaluation = config.get("evaluation", config)
    threshold = float(evaluation.get("growth_threshold", 0.25))
    seed = int(config.get("seed", 41))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results, all_records = [], []
    for scenario in SCENARIOS:
        world = deterministic_fixture(scenario)
        panel = _collect(world, scenario, "panel", seed)
        adaptive = _collect(world, scenario, "adaptive", seed)
        prior = _window_from_collection(panel, 0, 1, PANEL)
        recent = _window_from_collection(panel, 2, 3, PANEL)
        comparison = compare_windows(prior, recent, threshold)
        truth_counts = [_truth_count(world, PANEL, start, end) for start, end in ((0, 1), (2, 3))]
        onset = world.truth["change_tick"]
        changed = onset is not None
        first_available = min((p["available_at"] for p in world.posts if changed and p["event_at"] >= onset and p["author"] in PANEL), default=None)
        first_discovery = min((p["first_observed_at"] for p in panel["posts"] if changed and p["event_at"] >= onset), default=None)
        alert_at = max(r["receipt"]["response_tick"] for r in panel["records"]) if comparison["qualified_alert"] else None
        raw = adaptive["raw_observation_counts"]
        raw_ratio = (raw[1] - raw[0]) / raw[0] if raw[0] else None
        if scenario == "coverage_expansion":
            qualification = "not_comparable: source cohort changed"
        elif scenario == "ranked_cap_change":
            qualification = "partial: ranked response cap changed"
        elif scenario == "delayed_indexing_backfill":
            qualification = "not_comparable: indexing delay and historical backfill changed observation counts"
        else:
            qualification = "comparable: declared same panel and complete event scopes"
        all_records.extend(panel["records"] + adaptive["records"])
        results.append({"scenario": scenario, "fixture_generator": world.metadata["generator"],
                        "world_checksum": digest(asdict(world)), "world_size": {"accounts": len(world.accounts), "posts": len(world.posts), "ticks": world.ticks},
                        "fixed_panel": list(PANEL), "panel_reference_counts": truth_counts,
                        "panel_comparison": comparison, "panel_observed_counts": panel["event_time_counts"],
                        "adaptive_raw_observation_counts": raw, "adaptive_revised_event_counts": adaptive["event_time_counts"],
                        "adaptive_raw_growth_ratio": raw_ratio, "adaptive_raw_apparent_increase": raw[1] > raw[0],
                        "adaptive_qualification": qualification,
                        "budget": {"panel_requests": panel["requests"], "adaptive_requests": adaptive["requests"],
                                   "panel_cost": panel["cost"], "adaptive_cost": adaptive["cost"],
                                   "allocation_rule": "2 requests per event tick for each arm; all requests cost 1"},
                        "panel_fixed_lag": panel["fixed_observation_lag"],
                        "event": {"underlying_change": changed, "onset": onset, "first_observable": first_available,
                                  "first_discovery": first_discovery, "first_qualified_alert": alert_at,
                                  "missed": bool(changed and alert_at is None),
                                  "false_alert": bool(not changed and alert_at is not None),
                                  "onset_to_alert": alert_at - onset if alert_at is not None and onset is not None else None}})
    result = {"measurement_version": MEASUREMENT_VERSION, "kind": "deterministic_integrity_diagnostics",
              "seed": seed, "config_hash": digest({"threshold": threshold, "seed": seed}), "scenarios": results,
              "total_provider_requests": len(all_records),
              "false_alerts": sum(r["event"]["false_alert"] for r in results),
              "true_events": sum(r["event"]["underlying_change"] for r in results),
              "missed_events": sum(r["event"]["missed"] for r in results),
              "limitations": ["Deterministic fixtures establish accounting behavior, not a stochastic false-alarm rate.",
                              "Frozen panel conclusions apply only to that panel, not the whole population.",
                              "Delayed case uses a known synthetic maximum indexing delay and a fixed lag; real-provider completeness may be unknown.",
                              "Diagnostic acquisition schedules are prescribed controls, not learned policies."]}
    records_path = output / "measurement-receipts.jsonl"
    records_path.write_text("".join(canonical(record) + "\n" for record in all_records), encoding="utf-8")
    result["artifacts"] = {"records": str(records_path), "summary": str(output / "measurement-diagnostics.json")}
    (output / "measurement-diagnostics.json").write_text(canonical(result) + "\n", encoding="utf-8")
    return result
