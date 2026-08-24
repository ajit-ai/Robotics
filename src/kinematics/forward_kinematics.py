"""Forward kinematics for a planar (2D) serial robot arm.

Each joint rotates about the z-axis; links have fixed lengths. Joint angles
are absolute offsets applied sequentially, so the pose of link ``i`` is the
cumulative sum of the first ``i`` joint angles.
"""

import math


def _validate(link_lengths, joint_angles):
    if len(link_lengths) != len(joint_angles):
        raise ValueError("link_lengths and joint_angles must have the same length")
    if not link_lengths:
        raise ValueError("at least one link is required")
    if any(length <= 0 for length in link_lengths):
        raise ValueError("link lengths must be positive")


def forward_kinematics(link_lengths, joint_angles):
    """Compute every joint position of a planar serial arm.

    Args:
        link_lengths: Sequence of positive link lengths [l1, l2, ...].
        joint_angles: Sequence of joint angles in radians [q1, q2, ...].

    Returns:
        List of (x, y) tuples starting with the base (0, 0) and ending with
        the end-effector position. Length is ``len(link_lengths) + 1``.

    Raises:
        ValueError: If inputs are empty, mismatched, or contain a
            non-positive link length.

    Example:
        >>> forward_kinematics([1.0, 1.0], [0.0, 0.0])[-1]
        (2.0, 0.0)
    """
    _validate(link_lengths, joint_angles)
    x = y = 0.0
    cumulative_angle = 0.0
    positions = [(x, y)]
    for length, angle in zip(link_lengths, joint_angles):
        cumulative_angle += angle
        x += length * math.cos(cumulative_angle)
        y += length * math.sin(cumulative_angle)
        positions.append((x, y))
    return positions


def end_effector_position(link_lengths, joint_angles):
    """Return only the (x, y) position of the arm's end effector."""
    return forward_kinematics(link_lengths, joint_angles)[-1]


if __name__ == "__main__":
    links = [1.0, 1.0]
    for angles in ([math.pi / 4, math.pi / 4], [0.0, math.pi / 2]):
        ee = end_effector_position(links, angles)
        print(f"angles={angles} -> end effector at ({ee[0]:.3f}, {ee[1]:.3f})")
