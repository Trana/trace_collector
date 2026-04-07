from trace_collector.motion_detection import JointSample, MotionFirstChangeDetector


def _sample(
    stamp_ns: int,
    positions: list[float],
    velocities: list[float],
    names: list[str] | None = None,
) -> JointSample:
    return JointSample(
        stamp_ns=stamp_ns,
        names=names or ["j1", "j2"],
        positions=positions,
        velocities=velocities,
    )


def test_velocity_based_detection_after_hold_samples() -> None:
    detector = MotionFirstChangeDetector(
        velocity_threshold=0.05, position_threshold=100.0, hold_samples=2, timeout_ms=1000
    )
    detector.update_joint_sample(_sample(1_000_000, [0.0, 0.0], [0.0, 0.0]))
    detector.arm_trace(trace_id=1, command_applied_ns=1_000_000)

    detections, timeouts = detector.update_joint_sample(_sample(2_000_000, [0.0, 0.0], [0.06, 0.0]))
    assert detections == []
    assert timeouts == []

    detections, timeouts = detector.update_joint_sample(_sample(3_000_000, [0.0, 0.0], [0.07, 0.0]))
    assert detections == [(1, 3_000_000)]
    assert timeouts == []
    assert detector.pending_count() == 0


def test_position_based_detection_when_velocity_not_available() -> None:
    detector = MotionFirstChangeDetector(
        velocity_threshold=100.0, position_threshold=0.01, hold_samples=1, timeout_ms=1000
    )
    detector.update_joint_sample(_sample(1_000_000, [0.0, 0.0], []))
    detector.arm_trace(trace_id=2, command_applied_ns=1_000_000)

    detections, _ = detector.update_joint_sample(_sample(2_000_000, [0.02, 0.0], []))
    assert detections == [(2, 2_000_000)]


def test_baseline_is_initialized_after_arm_if_no_sample_yet() -> None:
    detector = MotionFirstChangeDetector(
        velocity_threshold=100.0, position_threshold=0.01, hold_samples=1, timeout_ms=1000
    )
    detector.arm_trace(trace_id=3, command_applied_ns=1_000_000)

    detections, _ = detector.update_joint_sample(_sample(2_000_000, [1.0, 1.0], []))
    assert detections == []
    detections, _ = detector.update_joint_sample(_sample(3_000_000, [1.02, 1.0], []))
    assert detections == [(3, 3_000_000)]


def test_timeout_removes_pending_trace() -> None:
    detector = MotionFirstChangeDetector(timeout_ms=2, hold_samples=1)
    detector.update_joint_sample(_sample(1_000_000, [0.0, 0.0], [0.0, 0.0]))
    detector.arm_trace(trace_id=4, command_applied_ns=1_000_000)

    detections, timed_out = detector.update_joint_sample(_sample(5_000_000, [0.0, 0.0], [0.0, 0.0]))
    assert detections == []
    assert timed_out == [4]
    assert detector.pending_count() == 0


def test_disarm_trace() -> None:
    detector = MotionFirstChangeDetector()
    detector.arm_trace(trace_id=9, command_applied_ns=1_000_000)
    assert detector.pending_count() == 1
    detector.disarm_trace(trace_id=9)
    assert detector.pending_count() == 0

