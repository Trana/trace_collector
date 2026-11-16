from __future__ import annotations

import json
import math
import os
import time

import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from marvin_trace_msgs.msg import TraceEvent

from .aggregation import DEFAULT_REQUIRED_STAGES, TraceAggregator, TraceEventSample
from .motion_detection import JointSample, MotionFirstChangeDetector


class TraceCollectorNode(Node):
    def __init__(self) -> None:
        super().__init__("trace_collector")

        self.declare_parameter("trace_event_topic", "/trace/events")
        self.declare_parameter("summary_topic", "/trace/summary")
        self.declare_parameter("stats_topic", "/trace/stats")
        self.declare_parameter("required_stages", DEFAULT_REQUIRED_STAGES)
        self.declare_parameter("trace_timeout_ms", 600)
        self.declare_parameter("max_open_traces", 4096)
        self.declare_parameter("emit_incomplete", True)
        self.declare_parameter("publish_mode", "both")  # per_trace|windowed|both
        self.declare_parameter("sweep_period_ms", 50)
        self.declare_parameter("stats_window_seconds", 30.0)
        self.declare_parameter("stats_publish_period_ms", 1000)
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("motion_velocity_threshold", 0.05)
        self.declare_parameter("motion_position_threshold", 0.01)
        self.declare_parameter("motion_hold_samples", 2)
        self.declare_parameter("motion_timeout_ms", 300)
        self.declare_parameter("capture_seconds", float(os.environ.get("MARVIN_TRACE_CAPTURE_SECONDS", "0")))
        seconds = float(self.get_parameter("capture_seconds").value)
        if not math.isfinite(seconds) or seconds < 0 or seconds > 300:
            raise ValueError("capture_seconds must be between 0 and 300")
        self._capture_started = time.monotonic()
        self._capture_seconds = seconds
        if seconds:
            self._capture_timer = self.create_timer(
                0.1, self._check_capture_deadline, clock=Clock(clock_type=ClockType.STEADY_TIME)
            )

        self._trace_event_topic = str(self.get_parameter("trace_event_topic").value)
        self._summary_topic = str(self.get_parameter("summary_topic").value)
        self._stats_topic = str(self.get_parameter("stats_topic").value)
        self._joint_state_topic = str(self.get_parameter("joint_state_topic").value)
        self._required_stages = [str(item) for item in self.get_parameter("required_stages").value]
        self._publish_mode = str(self.get_parameter("publish_mode").value).strip().lower()
        if self._publish_mode not in {"per_trace", "windowed", "both"}:
            self.get_logger().warn(f"Unsupported publish_mode={self._publish_mode}, fallback to both")
            self._publish_mode = "both"

        self._aggregator = TraceAggregator(
            required_stages=self._required_stages,
            trace_timeout_ms=int(self.get_parameter("trace_timeout_ms").value),
            max_open_traces=int(self.get_parameter("max_open_traces").value),
            emit_incomplete=bool(self.get_parameter("emit_incomplete").value),
        )
        self._motion_detector = MotionFirstChangeDetector(
            velocity_threshold=float(self.get_parameter("motion_velocity_threshold").value),
            position_threshold=float(self.get_parameter("motion_position_threshold").value),
            hold_samples=int(self.get_parameter("motion_hold_samples").value),
            timeout_ms=int(self.get_parameter("motion_timeout_ms").value),
        )

        self._summary_pub = self.create_publisher(String, self._summary_topic, 10)
        self._stats_pub = self.create_publisher(String, self._stats_topic, 10)
        # Use BEST_EFFORT so collector can receive trace events from mixed publishers
        # (some nodes publish TraceEvent with BEST_EFFORT, others with RELIABLE).
        self._trace_sub = self.create_subscription(
            TraceEvent, self._trace_event_topic, self._on_trace_event, qos_profile_sensor_data
        )
        self._joint_state_sub = self.create_subscription(
            JointState, self._joint_state_topic, self._on_joint_state, qos_profile_sensor_data
        )

        sweep_period_ms = max(5, int(self.get_parameter("sweep_period_ms").value))
        self._sweep_timer = self.create_timer(sweep_period_ms / 1000.0, self._on_sweep_timer)

        stats_period_ms = max(50, int(self.get_parameter("stats_publish_period_ms").value))
        self._stats_window_seconds = float(self.get_parameter("stats_window_seconds").value)
        self._stats_timer = self.create_timer(stats_period_ms / 1000.0, self._on_stats_timer)

        self.get_logger().info(
            "trace_collector started "
            f"(trace_event_topic={self._trace_event_topic}, summary_topic={self._summary_topic}, "
            f"stats_topic={self._stats_topic}, publish_mode={self._publish_mode}, "
            f"required_stages={self._required_stages}, joint_state_topic={self._joint_state_topic})"
        )

    def _on_trace_event(self, msg: TraceEvent) -> None:
        event_ns = int(msg.stamp.sec) * 1_000_000_000 + int(msg.stamp.nanosec)
        if event_ns <= 0:
            event_ns = self._now_ns()
        sample = TraceEventSample(
            trace_id=int(msg.trace_id),
            stage=str(msg.stage),
            stamp_ns=event_ns,
            source=str(msg.source),
        )
        summaries = self._aggregator.add_event(sample, now_ns=self._now_ns())
        self._publish_summaries(summaries)
        if sample.stage == "command_applied":
            self._motion_detector.arm_trace(trace_id=sample.trace_id, command_applied_ns=event_ns)
        elif sample.stage == "first_state_change":
            self._motion_detector.disarm_trace(trace_id=sample.trace_id)

    def _check_capture_deadline(self):
        if time.monotonic() - self._capture_started >= self._capture_seconds:
            self._publish_summaries(self._aggregator.flush_all())
            self.get_logger().info("Bounded trace capture completed")
            rclpy.shutdown()

    def _on_joint_state(self, msg: JointState) -> None:
        sample_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        if sample_ns <= 0:
            sample_ns = self._now_ns()
        sample = JointSample(
            stamp_ns=sample_ns,
            names=[str(item) for item in msg.name],
            positions=[float(item) for item in msg.position],
            velocities=[float(item) for item in msg.velocity],
        )
        detections, timed_out = self._motion_detector.update_joint_sample(sample)
        summaries = []
        for trace_id, event_ns in detections:
            synthetic_event = TraceEventSample(
                trace_id=int(trace_id),
                stage="first_state_change",
                stamp_ns=int(event_ns),
                source="trace_collector_motion",
            )
            summaries.extend(self._aggregator.add_event(synthetic_event, now_ns=self._now_ns()))
        self._publish_summaries(summaries)
        if timed_out:
            self.get_logger().debug(f"Motion detection timeout for trace_ids={timed_out}")

    def _on_sweep_timer(self) -> None:
        summaries = self._aggregator.flush_expired(now_ns=self._now_ns())
        self._publish_summaries(summaries)

    def _on_stats_timer(self) -> None:
        if self._publish_mode not in {"windowed", "both"}:
            return
        stats = self._aggregator.window_stats(
            now_ns=self._now_ns(), window_seconds=self._stats_window_seconds
        )
        self._stats_pub.publish(String(data=json.dumps(stats, separators=(",", ":"))))

    def _publish_summary(self, payload: dict[str, object]) -> None:
        self._summary_pub.publish(String(data=json.dumps(payload, separators=(",", ":"))))

    def _publish_summaries(self, summaries) -> None:  # noqa: ANN001
        if self._publish_mode not in {"per_trace", "both"}:
            return
        for summary in summaries:
            self._publish_summary(summary.to_dict())

    @staticmethod
    def _now_ns() -> int:
        return int(time.time() * 1_000_000_000)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = TraceCollectorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
