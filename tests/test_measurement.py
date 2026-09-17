from dataclasses import replace
import json

import pytest

from frontier_bench.measurement import WindowCount, compare_windows, deterministic_fixture, diagnostic_suite


def windows(prior_count=4, recent_count=4):
    prior = WindowCount(0, 1, prior_count, ("a", "b"), 4, 4)
    recent = WindowCount(2, 3, recent_count, ("a", "b"), 4, 4)
    return prior, recent


def test_fixed_panel_truth_and_matched_budget_diagnostics(tmp_path):
    result = diagnostic_suite(tmp_path)
    assert result["total_provider_requests"] == 64
    assert result["false_alerts"] == 0
    assert result["true_events"] == 1
    assert result["missed_events"] == 0
    for scenario in result["scenarios"]:
        assert scenario["panel_observed_counts"] == scenario["panel_reference_counts"]
        assert scenario["budget"]["panel_requests"] == scenario["budget"]["adaptive_requests"] == 8
        assert scenario["budget"]["panel_cost"] == scenario["budget"]["adaptive_cost"] == 8
        if not scenario["event"]["underlying_change"]:
            assert scenario["adaptive_raw_apparent_increase"]
            assert not scenario["panel_comparison"]["qualified_alert"]
    change = result["scenarios"][-1]
    assert change["panel_comparison"]["growth_ratio"] == 1.0
    assert change["event"] == {"underlying_change": True, "onset": 2, "first_observable": 2,
                                "first_discovery": 2, "first_qualified_alert": 3, "missed": False,
                                "false_alert": False, "onset_to_alert": 1}


def test_zero_and_missing_baseline_do_not_fabricate_growth_ratios():
    prior, recent = windows(0, 4)
    result = compare_windows(prior, recent)
    assert result["state"] == "zero_baseline"
    assert result["growth_ratio"] is None
    assert result["alert_kind"] == "new_activity_after_observed_zero"
    result = compare_windows(replace(prior, complete_exposures=0), recent)
    assert result["state"] == "insufficient_baseline"
    assert result["growth_ratio"] is None
    assert not result["qualified_alert"]


def test_partial_and_watermark_incomplete_are_explicit():
    prior, recent = windows(4, 8)
    assert compare_windows(prior, replace(recent, statuses=("partial",)))["state"] == "partial"
    result = compare_windows(replace(prior, watermark_complete=False), recent)
    assert result["state"] == "not_comparable"
    assert not result["qualified_alert"]


def test_membership_interpretation_duration_and_overlap_cannot_change_silently():
    prior, recent = windows(4, 8)
    for altered in (replace(recent, panel=("a", "c")), replace(recent, interpretation="new_labels"),
                    replace(recent, end=4), replace(recent, start=1, end=2)):
        assert compare_windows(prior, altered)["state"] == "not_comparable"


def test_backfill_preserves_old_event_time_and_index_availability(tmp_path):
    result = diagnostic_suite(tmp_path)
    records = [json.loads(line) for line in (tmp_path / "measurement-receipts.jsonl").read_text().splitlines()]
    backfilled = []
    for record in records:
        receipt = record["receipt"]
        for post in receipt["posts"]:
            assert post["available_at"] <= receipt["scope_tick"]
            if record["scenario"] == "delayed_indexing_backfill" and record["arm"] == "adaptive":
                if post["event_at"] < receipt["response_tick"]:
                    backfilled.append(post)
    assert backfilled
    delayed = next(s for s in result["scenarios"] if s["scenario"] == "delayed_indexing_backfill")
    assert delayed["adaptive_raw_observation_counts"] == [0, 8]
    assert delayed["adaptive_revised_event_counts"] == [4, 4]
    assert delayed["panel_fixed_lag"] == 2


def test_missed_event_is_reported_instead_of_removed(tmp_path):
    result = diagnostic_suite(tmp_path, {"growth_threshold": 2.0})
    assert result["true_events"] == 1
    assert result["missed_events"] == 1
    event = result["scenarios"][-1]["event"]
    assert event["first_qualified_alert"] is None
    assert event["onset_to_alert"] is None


def test_fixture_explicitly_separate_from_required_generator():
    fixture = deterministic_fixture("coverage_expansion")
    assert fixture.metadata["not_tadc_or_ndlib"]
    assert len(fixture.accounts) == 6
    assert len(fixture.posts) == 24
    with pytest.raises(ValueError):
        deterministic_fixture("invented")
