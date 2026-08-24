"""Analytic inverse kinematics for a two-link planar robot arm."""

import math

_TOLERANCE = 1e-9


def _clamp_cosine(value):
    return max(-1.0, min(1.0, value))


def inverse_kinematics_2link(l1, l2, x, y, elbow="up"):
    """Solve joint angles that place the end effector at ``(x, y)``.

    Uses the law of cosines for the second joint and the two-argument
    arctangent for the first joint.

    Args:
        l1: Length of the first (proximal) link, must be positive.
        l2: Length of the second (distal) link, must be positive.
        x: Target x coordinate.
        y: Target y coordinate.
        elbow: Configuration branch, either ``"up"`` or ``"down"``.

    Returns:
        Tuple ``(q1, q2)`` of joint angles in radians such that
        :func:`~src.kinematics.forward_kinematics.end_effector_position`
        maps them back to ``(x, y)``.

    Raises:
        ValueError: If the target lies outside the reachable annulus
            ``abs(l1 - l2) <= r <= l1 + l2`` or inputs are invalid.

    Example:
        >>> import math
        >>> q1, q2 = inverse_kinematics_2link(1.0, 1.0, 1.0, 1.0)
        >>> round(q1, 6), round(q2, 6)
        (0.0, 1.570796)
    """
    if l1 <= 0 or l2 <= 0:
        raise ValueError("link lengths must be positive")
    if elbow not in ("up", "down"):
        raise ValueError('elbow must be "up" or "down"')

    distance_sq = x * x + y * y
    reach_max = l1 + l2
    reach_min = abs(l1 - l2)
    distance = math.sqrt(distance_sq)
    if distance > reach_max + _TOLERANCE or distance < reach_min - _TOLERANCE:
        raise ValueError(
            f"target ({x}, {y}) is unreachable for links {l1}, {l2}"
        )

    cos_q2 = _clamp_cosine(
        (distance_sq - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    )
    q2 = math.acos(cos_q2)
    if elbow == "down":
        q2 = -q2

    q1 = math.atan2(y, x) - math.atan2(
        l2 * math.sin(q2), l1 + l2 * math.cos(q2)
    )
    return q1, q2


if __name__ == "__main__":
    target = (1.2, 0.8)
    q1, q2 = inverse_kinematics_2link(1.0, 1.0, *target)
    print(f"target {target} -> q1={q1:.4f} rad, q2={q2:.4f} rad")
