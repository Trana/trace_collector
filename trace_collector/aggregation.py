from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import statistics
from typing import Iterable


DEFAULT_REQUIRED_STAGES = [
    "obs_ready",
    "infer_start",
    "infer_end",
    "command_published",
    "command_received",
    "command_applied",
    "first_state_change",
]


DEFAULT_DURATION_PAIRS = [
    ("obs_ready", "infer_start"),
    ("infer_start", "infer_end"),
    ("infer_end", "command_published"),
    ("command_published", "command_received"),
    ("command_received", "command_applied"),
    ("command_applied", "first_state_change"),
    ("obs_ready", "first_state_change"),
]


@dataclass(frozen=True)
class TraceEventSample:
    trace_id: int
    stage: str
    stamp_ns: int
    source: str = ""


@dataclass
class TraceSummary:
    trace_id: int
    complete: bool
    timed_out: bool
    missing_stages: list[str]
    stages_ns: dict[str, int]
    stage_sources: dict[str, str]
    durations_ms: dict[str, float]
    first_seen_ns: int
    last_seen_ns: int
    event_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "trace_id": self.trace_id,
            "complete": self.complete,
            "timed_out": self.timed_out,
            "missing_stages": list(self.missing_stages),
            "stages_ns": dict(self.stages_ns),
            "stage_sources": dict(self.stage_sources),
            "durations_ms": dict(self.durations_ms),
            "first_seen_ns": self.first_seen_ns,
            "last_seen_ns": self.last_seen_ns,
            "event_count": self.event_count,
        }


@dataclass
class _OpenTrace:
    trace_id: int
    first_seen_ns: int
    last_seen_ns: int
    stages_ns: dict[str, int] = field(default_factory=dict)
    stage_sources: dict[str, str] = field(default_factory=dict)
    event_count: int = 0


class TraceAggregator:
    def __init__(
        self,
        *,
        required_stages: Iterable[str] | None = None,
        duration_pairs: Iterable[tuple[str, str]] | None = None,
        trace_timeout_ms: int = 600,
        max_open_traces: int = 4096,
        emit_incomplete: bool = True,
    ) -> None:
        raw_stages = DEFAULT_REQUIRED_STAGES if required_stages is None else required_stages
        stages = [str(item).strip() for item in raw_stages if str(item).strip()]
        if not stages:
            raise ValueError("required_stages must not be empty")
        self._required_stages = stages
        self._required_set = set(stages)
        self._duration_pairs = list(duration_pairs or DEFAULT_DURATION_PAIRS)
        self._timeout_ns = max(1, int(trace_timeout_ms)) * 1_000_000
        self._max_open_traces = max(1, int(max_open_traces))
        self._emit_incomplete = bool(emit_incomplete)

        self._open_traces: dict[int, _OpenTrace] = {}
        self._completed: deque[tuple[int, TraceSummary]] = deque()

    @property
    def required_stages(self) -> list[str]:
        return list(self._required_stages)

    def open_trace_count(self) -> int:
        return len(self._open_traces)

    def add_event(self, event: TraceEventSample, *, now_ns: int | None = None) -> list[TraceSummary]:
        if event.trace_id <= 0:
            return []
        stage = event.stage.strip()
        if not stage:
            return []
        event_ns = int(event.stamp_ns if event.stamp_ns > 0 else (now_ns if now_ns is not None else 0))
        if event_ns <= 0:
            return []

        summaries: list[TraceSummary] = []
        if event.trace_id not in self._open_traces and len(self._open_traces) >= self._max_open_traces:
            evicted = self._evict_oldest(now_ns=event_ns)
            if evicted is not None:
                summaries.append(evicted)

        trace = self._open_traces.get(event.trace_id)
        if trace is None:
            trace = _OpenTrace(trace_id=event.trace_id, first_seen_ns=event_ns, last_seen_ns=event_ns)
            self._open_traces[event.trace_id] = trace

        trace.last_seen_ns = max(trace.last_seen_ns, event_ns)
        if stage not in trace.stages_ns:
            trace.stages_ns[stage] = event_ns
            trace.stage_sources[stage] = event.source
            trace.event_count += 1

        if self._is_complete(trace):
            summaries.append(self._finalize(trace.trace_id, timed_out=False))

        return [summary for summary in summaries if summary is not None]

    def flush_expired(self, *, now_ns: int) -> list[TraceSummary]:
        now_ns = int(now_ns)
        if now_ns <= 0:
            return []
        expired: list[int] = []
        for trace_id, trace in self._open_traces.items():
            if now_ns - trace.first_seen_ns >= self._timeout_ns:
                expired.append(trace_id)

        out: list[TraceSummary] = []
        for trace_id in expired:
            summary = self._finalize(trace_id, timed_out=True)
            if summary is not None:
                out.append(summary)
        return out

    def flush_all(self, *, timed_out: bool = True) -> list[TraceSummary]:
        out: list[TraceSummary] = []
        for trace_id in list(self._open_traces.keys()):
            summary = self._finalize(trace_id, timed_out=timed_out)
            if summary is not None:
                out.append(summary)
        return out

    def window_stats(self, *, now_ns: int, window_seconds: float) -> dict[str, object]:
        window_ns = max(1, int(window_seconds * 1_000_000_000))
        now_ns = int(now_ns)
        while self._completed and (now_ns - self._completed[0][0]) > window_ns:
            self._completed.popleft()

        summaries = [summary for _, summary in self._completed]
        complete = [summary for summary in summaries if summary.complete]
        incomplete = [summary for summary in summaries if not summary.complete]

        durations_by_key: dict[str, list[float]] = {}
        for summary in complete:
            for key, value in summary.durations_ms.items():
                durations_by_key.setdefault(key, []).append(float(value))

        latency_stats = {}
        for key, values in durations_by_key.items():
            values_sorted = sorted(values)
            latency_stats[key] = {
                "count": len(values_sorted),
                "p50_ms": _percentile(values_sorted, 50.0),
                "p95_ms": _percentile(values_sorted, 95.0),
                "mean_ms": float(statistics.fmean(values_sorted)),
            }

        return {
            "window_seconds": float(window_seconds),
            "total_traces": len(summaries),
            "complete_traces": len(complete),
            "incomplete_traces": len(incomplete),
            "open_traces": len(self._open_traces),
            "latency_ms": latency_stats,
        }

    def _evict_oldest(self, *, now_ns: int) -> TraceSummary | None:
        if not self._open_traces:
            return None
        oldest = min(self._open_traces.values(), key=lambda item: item.first_seen_ns)
        return self._finalize(oldest.trace_id, timed_out=True)

    def _is_complete(self, trace: _OpenTrace) -> bool:
        return self._required_set.issubset(trace.stages_ns.keys())

    def _finalize(self, trace_id: int, *, timed_out: bool) -> TraceSummary | None:
        trace = self._open_traces.pop(trace_id, None)
        if trace is None:
            return None
        missing = [stage for stage in self._required_stages if stage not in trace.stages_ns]
        complete = len(missing) == 0
        if not complete and not self._emit_incomplete:
            return None
        summary = TraceSummary(
            trace_id=trace.trace_id,
            complete=complete,
            timed_out=bool(timed_out),
            missing_stages=missing,
            stages_ns=dict(trace.stages_ns),
            stage_sources=dict(trace.stage_sources),
            durations_ms=_compute_durations_ms(trace.stages_ns, self._duration_pairs),
            first_seen_ns=trace.first_seen_ns,
            last_seen_ns=trace.last_seen_ns,
            event_count=trace.event_count,
        )
        self._completed.append((summary.last_seen_ns, summary))
        return summary


def _compute_durations_ms(
    stages_ns: dict[str, int], duration_pairs: Iterable[tuple[str, str]]
) -> dict[str, float]:
    out: dict[str, float] = {}
    for first, second in duration_pairs:
        if first not in stages_ns or second not in stages_ns:
            continue
        delta_ns = int(stages_ns[second]) - int(stages_ns[first])
        out[f"{first}->{second}"] = float(delta_ns) / 1_000_000.0
    return out


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    p = min(100.0, max(0.0, float(percentile)))
    pos = (p / 100.0) * (len(sorted_values) - 1)
    lower = int(pos)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return float(sorted_values[lower])
    weight = pos - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)
