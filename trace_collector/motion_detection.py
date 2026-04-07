from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class JointSample:
    stamp_ns: int
    names: list[str]
    positions: list[float]
    velocities: list[float]


@dataclass
class _PendingMotion:
    trace_id: int
    command_applied_ns: int
    baseline_positions: dict[str, float] | None = None
    consecutive_hits: int = 0


class MotionFirstChangeDetector:
    def __init__(
        self,
        *,
        velocity_threshold: float = 0.05,
        position_threshold: float = 0.01,
        hold_samples: int = 2,
        timeout_ms: int = 300,
    ) -> None:
        self._velocity_threshold = abs(float(velocity_threshold))
        self._position_threshold = abs(float(position_threshold))
        self._hold_samples = max(1, int(hold_samples))
        self._timeout_ns = max(1, int(timeout_ms)) * 1_000_000
        self._latest_sample: JointSample | None = None
        self._pending: dict[int, _PendingMotion] = {}

    def arm_trace(self, *, trace_id: int, command_applied_ns: int) -> None:
        if trace_id <= 0:
            return
        baseline = None
        if self._latest_sample is not None:
            baseline = _positions_map(self._latest_sample.names, self._latest_sample.positions)
        self._pending[trace_id] = _PendingMotion(
            trace_id=trace_id,
            command_applied_ns=max(0, int(command_applied_ns)),
            baseline_positions=baseline,
            consecutive_hits=0,
        )

    def disarm_trace(self, *, trace_id: int) -> None:
        self._pending.pop(int(trace_id), None)

    def pending_count(self) -> int:
        return len(self._pending)

    def update_joint_sample(self, sample: JointSample) -> tuple[list[tuple[int, int]], list[int]]:
        self._latest_sample = sample
        if not self._pending:
            return ([], [])

        now_ns = int(sample.stamp_ns)
        detections: list[tuple[int, int]] = []
        timed_out: list[int] = []

        velocity_moved = _velocity_triggered(sample.velocities, threshold=self._velocity_threshold)
        positions_map = _positions_map(sample.names, sample.positions)

        for trace_id in list(self._pending.keys()):
            state = self._pending.get(trace_id)
            if state is None:
                continue
            if state.baseline_positions is None:
                state.baseline_positions = dict(positions_map)

            if now_ns > 0 and state.command_applied_ns > 0 and now_ns - state.command_applied_ns >= self._timeout_ns:
                timed_out.append(trace_id)
                self._pending.pop(trace_id, None)
                continue

            position_moved = _position_triggered(
                positions_map,
                baseline_positions=state.baseline_positions,
                threshold=self._position_threshold,
            )

            if velocity_moved or position_moved:
                state.consecutive_hits += 1
            else:
                state.consecutive_hits = 0

            if state.consecutive_hits >= self._hold_samples:
                detections.append((trace_id, now_ns))
                self._pending.pop(trace_id, None)

        return (detections, timed_out)


def _positions_map(names: Iterable[str], positions: Iterable[float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, pos in zip(names, positions):
        if name:
            out[str(name)] = float(pos)
    return out


def _velocity_triggered(velocities: Iterable[float], *, threshold: float) -> bool:
    if threshold <= 0.0:
        return False
    for value in velocities:
        if abs(float(value)) >= threshold:
            return True
    return False


def _position_triggered(
    positions: dict[str, float],
    *,
    baseline_positions: dict[str, float] | None,
    threshold: float,
) -> bool:
    if threshold <= 0.0 or not baseline_positions:
        return False
    for name, value in positions.items():
        if name not in baseline_positions:
            continue
        if abs(float(value) - float(baseline_positions[name])) >= threshold:
            return True
    return False

