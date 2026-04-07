from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("marvin_trace_msgs.msg")

import rclpy

from trace_collector.aggregation import TraceAggregator
from trace_collector.motion_detection import MotionFirstChangeDetector
from trace_collector.trace_collector_node import TraceCollectorNode


class _PubCapture:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def publish(self, msg) -> None:  # noqa: ANN001
        self.messages.append(str(msg.data))


def _make_node(mode: str = "both"):
    node = SimpleNamespace()
    node._publish_mode = mode
    node._aggregator = TraceAggregator(
        required_stages=["obs_ready", "first_state_change"],
        trace_timeout_ms=1,
        emit_incomplete=True,
    )
    node._summary_pub = _PubCapture()
    node._stats_pub = _PubCapture()
    node._stats_window_seconds = 10.0
    node._now_value = 1_000_000_000
    node._now_ns = lambda: node._now_value
    node._motion_detector = MotionFirstChangeDetector(
        velocity_threshold=0.05,
        position_threshold=0.01,
        hold_samples=2,
        timeout_ms=1000,
    )
    node._publish_summary = lambda payload: TraceCollectorNode._publish_summary(node, payload)
    node._publish_summaries = lambda summaries: TraceCollectorNode._publish_summaries(node, summaries)
    node.get_logger = lambda: SimpleNamespace(debug=lambda *_args, **_kwargs: None)
    return node


def _msg(trace_id: int, stage: str, sec: int, nanosec: int, source: str = "test"):
    return SimpleNamespace(
        trace_id=trace_id,
        stage=stage,
        source=source,
        stamp=SimpleNamespace(sec=sec, nanosec=nanosec),
    )


def test_on_trace_event_publishes_summary_when_trace_completes() -> None:
    node = _make_node(mode="both")
    TraceCollectorNode._on_trace_event(node, _msg(42, "obs_ready", 1, 0))
    TraceCollectorNode._on_trace_event(node, _msg(42, "first_state_change", 1, 50_000_000))

    assert len(node._summary_pub.messages) == 1
    payload = json.loads(node._summary_pub.messages[0])
    assert payload["trace_id"] == 42
    assert payload["complete"] is True
    assert "obs_ready->first_state_change" in payload["durations_ms"]


def test_on_trace_event_uses_now_when_stamp_is_zero() -> None:
    node = _make_node(mode="both")
    node._now_value = 2_000_000_000
    TraceCollectorNode._on_trace_event(node, _msg(7, "obs_ready", 0, 0))
    TraceCollectorNode._on_trace_event(node, _msg(7, "first_state_change", 0, 0))
    payload = json.loads(node._summary_pub.messages[0])
    assert payload["stages_ns"]["obs_ready"] == 2_000_000_000


def test_sweep_timer_publishes_timeout_summary() -> None:
    node = _make_node(mode="both")
    TraceCollectorNode._on_trace_event(node, _msg(9, "obs_ready", 1, 0))
    node._now_value = 1_002_000_000  # > 1ms timeout from first event
    TraceCollectorNode._on_sweep_timer(node)

    assert len(node._summary_pub.messages) == 1
    payload = json.loads(node._summary_pub.messages[0])
    assert payload["timed_out"] is True
    assert payload["complete"] is False


def test_joint_state_motion_detection_emits_first_state_change_summary() -> None:
    node = _make_node(mode="both")
    TraceCollectorNode._on_trace_event(node, _msg(500, "obs_ready", 1, 0))
    TraceCollectorNode._on_trace_event(node, _msg(500, "command_applied", 1, 5_000_000))

    js1 = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=10_000_000)),
        name=["j1", "j2"],
        position=[0.0, 0.0],
        velocity=[0.0, 0.0],
    )
    js2 = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=20_000_000)),
        name=["j1", "j2"],
        position=[0.0, 0.0],
        velocity=[0.1, 0.0],
    )
    js3 = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=30_000_000)),
        name=["j1", "j2"],
        position=[0.0, 0.0],
        velocity=[0.12, 0.0],
    )
    TraceCollectorNode._on_joint_state(node, js1)
    TraceCollectorNode._on_joint_state(node, js2)
    TraceCollectorNode._on_joint_state(node, js3)

    assert len(node._summary_pub.messages) == 1
    payload = json.loads(node._summary_pub.messages[0])
    assert payload["trace_id"] == 500
    assert payload["complete"] is True
    assert payload["stages_ns"].get("first_state_change", 0) > 0


def test_stats_timer_respects_publish_mode() -> None:
    per_trace = _make_node(mode="per_trace")
    TraceCollectorNode._on_stats_timer(per_trace)
    assert per_trace._stats_pub.messages == []

    windowed = _make_node(mode="windowed")
    TraceCollectorNode._on_stats_timer(windowed)
    assert len(windowed._stats_pub.messages) == 1
    payload = json.loads(windowed._stats_pub.messages[0])
    assert payload["total_traces"] == 0


def test_now_ns_returns_positive_time() -> None:
    assert TraceCollectorNode._now_ns() > 0


def test_node_initializes_with_rclpy_context() -> None:
    rclpy.init(args=[])
    node = TraceCollectorNode()
    try:
        assert node._publish_mode in {"per_trace", "windowed", "both"}
        assert node._trace_event_topic == "/trace/events"
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_main_calls_shutdown_on_keyboard_interrupt(monkeypatch) -> None:
    events = {"destroyed": False, "shutdown": False}

    class DummyNode:
        def destroy_node(self) -> None:
            events["destroyed"] = True

    monkeypatch.setattr("trace_collector.trace_collector_node.TraceCollectorNode", lambda: DummyNode())
    monkeypatch.setattr("trace_collector.trace_collector_node.rclpy.init", lambda args=None: None)
    monkeypatch.setattr(
        "trace_collector.trace_collector_node.rclpy.spin",
        lambda _node: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    monkeypatch.setattr("trace_collector.trace_collector_node.rclpy.ok", lambda: True)
    monkeypatch.setattr(
        "trace_collector.trace_collector_node.rclpy.shutdown",
        lambda: events.__setitem__("shutdown", True),
    )

    from trace_collector.trace_collector_node import main

    main([])
    assert events["destroyed"] is True
    assert events["shutdown"] is True
