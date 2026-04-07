import pytest

from trace_collector.aggregation import (
    TraceAggregator,
    TraceEventSample,
    _percentile,
)


def _event(trace_id: int, stage: str, stamp_ns: int, source: str = "test") -> TraceEventSample:
    return TraceEventSample(trace_id=trace_id, stage=stage, stamp_ns=stamp_ns, source=source)


def test_complete_trace_emits_summary_with_expected_durations() -> None:
    agg = TraceAggregator(trace_timeout_ms=1000)
    events = [
        _event(1, "obs_ready", 1_000_000_000),
        _event(1, "infer_start", 1_010_000_000),
        _event(1, "infer_end", 1_030_000_000),
        _event(1, "command_published", 1_035_000_000),
        _event(1, "command_received", 1_050_000_000),
        _event(1, "command_applied", 1_060_000_000),
        _event(1, "first_state_change", 1_120_000_000),
    ]
    summaries = []
    for event in events:
        summaries.extend(agg.add_event(event, now_ns=event.stamp_ns))

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.complete is True
    assert summary.timed_out is False
    assert summary.missing_stages == []
    assert summary.durations_ms["infer_start->infer_end"] == 20.0
    assert summary.durations_ms["obs_ready->first_state_change"] == 120.0
    assert summary.event_count == 7
    assert agg.open_trace_count() == 0


def test_out_of_order_events_still_finalize_once_complete() -> None:
    agg = TraceAggregator(trace_timeout_ms=1000)
    out_of_order = [
        _event(10, "command_received", 100),
        _event(10, "obs_ready", 80),
        _event(10, "infer_start", 90),
        _event(10, "infer_end", 95),
        _event(10, "command_published", 96),
        _event(10, "first_state_change", 140),
        _event(10, "command_applied", 110),
    ]
    summaries = []
    for item in out_of_order:
        summaries.extend(agg.add_event(item, now_ns=item.stamp_ns))
    assert len(summaries) == 1
    assert summaries[0].complete is True
    assert summaries[0].trace_id == 10


def test_duplicate_stage_keeps_first_timestamp() -> None:
    agg = TraceAggregator(required_stages=["obs_ready", "first_state_change"])
    agg.add_event(_event(7, "obs_ready", 1_000), now_ns=1_000)
    agg.add_event(_event(7, "obs_ready", 2_000), now_ns=2_000)  # duplicate; ignored
    summaries = agg.add_event(_event(7, "first_state_change", 3_000), now_ns=3_000)

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.stages_ns["obs_ready"] == 1_000
    assert summary.event_count == 2


def test_timeout_emits_incomplete_when_enabled() -> None:
    agg = TraceAggregator(trace_timeout_ms=1, emit_incomplete=True)
    agg.add_event(_event(3, "obs_ready", 1_000_000), now_ns=1_000_000)
    timed_out = agg.flush_expired(now_ns=3_000_000)

    assert len(timed_out) == 1
    assert timed_out[0].complete is False
    assert timed_out[0].timed_out is True
    assert "first_state_change" in timed_out[0].missing_stages


def test_timeout_drops_incomplete_when_disabled() -> None:
    agg = TraceAggregator(trace_timeout_ms=1, emit_incomplete=False)
    agg.add_event(_event(4, "obs_ready", 1_000_000), now_ns=1_000_000)
    timed_out = agg.flush_expired(now_ns=3_000_000)

    assert timed_out == []
    assert agg.open_trace_count() == 0


def test_max_open_traces_evicts_oldest() -> None:
    agg = TraceAggregator(max_open_traces=2, emit_incomplete=True, trace_timeout_ms=1000)
    agg.add_event(_event(1, "obs_ready", 100), now_ns=100)
    agg.add_event(_event(2, "obs_ready", 200), now_ns=200)
    summaries = agg.add_event(_event(3, "obs_ready", 300), now_ns=300)

    assert len(summaries) == 1
    assert summaries[0].trace_id == 1
    assert summaries[0].complete is False
    assert agg.open_trace_count() == 2


def test_window_stats_reports_percentiles_for_completed_traces() -> None:
    agg = TraceAggregator(required_stages=["obs_ready", "first_state_change"])
    traces = [
        (1, 1_000_000, 21_000_000),   # 20 ms
        (2, 2_000_000, 42_000_000),   # 40 ms
        (3, 3_000_000, 63_000_000),   # 60 ms
    ]
    for trace_id, start_ns, end_ns in traces:
        agg.add_event(_event(trace_id, "obs_ready", start_ns), now_ns=start_ns)
        agg.add_event(_event(trace_id, "first_state_change", end_ns), now_ns=end_ns)

    stats = agg.window_stats(now_ns=70_000_000, window_seconds=10.0)
    latency = stats["latency_ms"]["obs_ready->first_state_change"]
    assert stats["complete_traces"] == 3
    assert round(latency["p50_ms"], 3) == 40.0
    assert round(latency["p95_ms"], 3) == 58.0
    assert round(latency["mean_ms"], 3) == 40.0


def test_constructor_rejects_empty_required_stages() -> None:
    with pytest.raises(ValueError):
        TraceAggregator(required_stages=[])


def test_invalid_events_are_ignored() -> None:
    agg = TraceAggregator(required_stages=["obs_ready", "first_state_change"])
    assert agg.add_event(_event(0, "obs_ready", 1)) == []
    assert agg.add_event(_event(1, "", 1)) == []
    assert agg.add_event(_event(1, "obs_ready", -1), now_ns=0) == []
    assert agg.open_trace_count() == 0


def test_flush_all_returns_all_open_traces() -> None:
    agg = TraceAggregator(required_stages=["obs_ready", "first_state_change"], emit_incomplete=True)
    agg.add_event(_event(100, "obs_ready", 10), now_ns=10)
    agg.add_event(_event(200, "obs_ready", 20), now_ns=20)

    out = agg.flush_all(timed_out=True)
    assert len(out) == 2
    assert sorted(item.trace_id for item in out) == [100, 200]
    assert agg.open_trace_count() == 0


def test_window_stats_prunes_old_entries() -> None:
    agg = TraceAggregator(required_stages=["obs_ready", "first_state_change"])
    agg.add_event(_event(1, "obs_ready", 1_000_000), now_ns=1_000_000)
    agg.add_event(_event(1, "first_state_change", 2_000_000), now_ns=2_000_000)
    stats = agg.window_stats(now_ns=5_000_000_000, window_seconds=0.001)
    assert stats["total_traces"] == 0


def test_percentile_edge_cases() -> None:
    assert _percentile([], 50) == 0.0
    assert _percentile([10.0], 95) == 10.0
    assert _percentile([10.0, 20.0, 30.0], 0) == 10.0
    assert _percentile([10.0, 20.0, 30.0], 100) == 30.0
