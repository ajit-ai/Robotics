import math

import pytest

from src.kinematics.forward_kinematics import end_effector_position
from src.kinematics.inverse_kinematics import inverse_kinematics_2link

L1 = 1.5
L2 = 0.8


class TestInverseKinematics:
    @pytest.mark.parametrize("elbow", ["up", "down"])
    @pytest.mark.parametrize(
        "x, y",
        [
            (1.0, 1.0),
            (2.29, 0.0),
            (0.5, -1.2),
            (-1.8, 0.6),
            (-0.75, 0.05),
            (-0.7, -0.7),
        ],
    )
    def test_round_trip_fk_of_ik_recovers_target(self, elbow, x, y):
        q1, q2 = inverse_kinematics_2link(L1, L2, x, y, elbow=elbow)
        fx, fy = end_effector_position([L1, L2], [q1, q2])
        assert fx == pytest.approx(x, abs=1e-9)
        assert fy == pytest.approx(y, abs=1e-9)

    def test_elbow_branches_are_mirror_solutions(self):
        x, y = 1.2, 0.8
        q1_up, q2_up = inverse_kinematics_2link(L1, L2, x, y, "up")
        q1_down, q2_down = inverse_kinematics_2link(L1, L2, x, y, "down")
        assert q2_up > 0 > q2_down
        assert q2_up == pytest.approx(-q2_down)
        assert q1_up + q1_down == pytest.approx(2.0 * math.atan2(y, x))

    def test_fully_extended_target_gives_zero_bend(self):
        q1, q2 = inverse_kinematics_2link(L1, L2, L1 + L2, 0.0)
        assert abs(q2) < 1e-6
        assert abs(q1) < 1e-6
        fx, fy = end_effector_position([L1, L2], [q1, q2])
        assert (fx, fy) == pytest.approx((L1 + L2, 0.0), abs=1e-9)

    def test_boundary_reach_is_accepted_with_tolerance(self):
        slightly_outside_x = L1 + L2 + 1e-10
        q1, q2 = inverse_kinematics_2link(L1, L2, slightly_outside_x, 0.0)
        assert abs(q2) < 1e-4

    @pytest.mark.parametrize("x, y", [(3.0, 0.5), (0.0, 5.0)])
    def test_far_target_raises(self, x, y):
        with pytest.raises(ValueError, match="unreachable"):
            inverse_kinematics_2link(L1, L2, x, y)

    def test_inner_hole_is_unreachable_for_unequal_links(self):
        with pytest.raises(ValueError, match="unreachable"):
            inverse_kinematics_2link(L1, L2, 0.1, 0.0)

    def test_equal_links_cover_origin(self):
        q1, q2 = inverse_kinematics_2link(1.0, 1.0, 0.0, 0.0)
        fx, fy = end_effector_position([1.0, 1.0], [q1, q2])
        assert fx == pytest.approx(0.0, abs=1e-9)
        assert fy == pytest.approx(0.0, abs=1e-9)

    def test_invalid_elbow_raises(self):
        with pytest.raises(ValueError, match="elbow"):
            inverse_kinematics_2link(L1, L2, 1.0, 1.0, elbow="sideways")

    def test_non_positive_links_raise(self):
        with pytest.raises(ValueError, match="positive"):
            inverse_kinematics_2link(0.0, L2, 1.0, 1.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
