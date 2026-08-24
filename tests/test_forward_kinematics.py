import math
import cmath

import pytest

from src.kinematics.forward_kinematics import end_effector_position
from src.kinematics.forward_kinematics import forward_kinematics


class TestForwardKinematics:
    def test_straight_arm_lies_along_x_axis(self):
        ee = end_effector_position([1.0, 2.0], [0.0, 0.0])
        assert ee == pytest.approx((3.0, 0.0))

    def test_single_link_rotates_full_length(self):
        ee = end_effector_position([1.0], [math.pi / 2])
        assert ee[0] == pytest.approx(0.0, abs=1e-12)
        assert ee[1] == pytest.approx(1.0)

    def test_returns_base_plus_one_point_per_link(self):
        positions = forward_kinematics([1.0, 1.0, 1.0], [0.1, -0.2, 0.3])
        assert len(positions) == 4
        assert positions[0] == (0.0, 0.0)

    def test_matches_complex_number_reference_model(self):
        lengths = [0.8, 1.1, 0.6]
        angles = [0.7, -0.4, 1.2]
        expected = sum(
            l * cmath.exp(1j * sum(angles[: i + 1])) for i, l in enumerate(lengths)
        )
        x, y = end_effector_position(lengths, angles)
        assert x == pytest.approx(expected.real, rel=1e-12)
        assert y == pytest.approx(expected.imag, rel=1e-12)

    def test_end_effector_never_exceeds_total_reach(self):
        lengths = [1.0, 0.75, 0.5]
        angles = [2.1, 0.9, -3.0]
        x, y = end_effector_position(lengths, angles)
        assert math.hypot(x, y) <= sum(lengths) + 1e-12

    def test_joint_angles_are_cumulative(self):
        positions = forward_kinematics([1.0, 1.0], [math.pi / 2, math.pi / 2])
        second_x, second_y = positions[1]
        assert second_x == pytest.approx(0.0, abs=1e-12)
        assert second_y == pytest.approx(1.0)
        assert positions[2][0] == pytest.approx(-1.0, abs=1e-12)
        assert positions[2][1] == pytest.approx(1.0)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            forward_kinematics([1.0, 1.0], [0.0])

    def test_empty_arm_raises(self):
        with pytest.raises(ValueError, match="one link"):
            forward_kinematics([], [])

    def test_non_positive_link_raises(self):
        with pytest.raises(ValueError, match="positive"):
            forward_kinematics([1.0, -0.5], [0.0, 0.0])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
