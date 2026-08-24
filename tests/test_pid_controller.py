import pytest

from src.control.pid_controller import PIDController


def simulate(pid, plant_gain=1.0, disturbance=0.0, setpoint=1.0, steps=2000, dt=0.01):
    state = 0.0
    for _ in range(steps):
        u = pid.update(setpoint, state)
        state += dt * (plant_gain * u + disturbance)
    return state


class TestPIDBasics:
    def test_pure_proportional_output(self):
        pid = PIDController(kp=2.0, ki=0.0, kd=0.0, dt=0.1)
        assert pid.update(1.0, 0.25) == pytest.approx(2.0 * 0.75)

    def test_first_update_has_no_derivative_kick_from_history(self):
        pid = PIDController(kp=1.0, ki=0.0, kd=3.0, dt=0.1)
        assert pid.update(2.0, 2.0) == pytest.approx(0.0)

    def test_integral_accumulates_over_steps(self):
        pid = PIDController(kp=0.0, ki=2.0, kd=0.0, dt=0.1)
        for _ in range(5):
            output = pid.update(1.0, 0.0)
        assert pid.integral == pytest.approx(0.5)
        assert output == pytest.approx(2.0 * 0.5)

    def test_derivative_reacts_to_error_change(self):
        pid = PIDController(kp=1.0, ki=0.0, kd=1.0, dt=0.1)
        first = pid.update(1.0, 0.0)
        second = pid.update(1.0, 0.6)
        assert first == pytest.approx(1.0)
        expected = 1.0 * 0.4 + 1.0 * ((0.4 - 1.0) / 0.1)
        assert second == pytest.approx(expected)

    def test_reset_clears_internal_state(self):
        pid = PIDController(kp=1.0, ki=5.0, kd=1.0, dt=0.1)
        for _ in range(10):
            pid.update(1.0, 0.0)
        pid.reset()
        assert pid.integral == 0.0
        assert pid.update(1.0, 1.0) == 0.0


class TestAntiWindup:
    def test_integral_frozen_while_saturated_in_error_direction(self):
        pid = PIDController(
            kp=0.0, ki=1000.0, kd=0.0, dt=1.0, output_limits=(-1.0, 1.0)
        )
        for _ in range(50):
            output = pid.update(10.0, 0.0)
        assert output == pytest.approx(1.0)
        assert pid.integral == 0.0

    def test_integral_frozen_for_opposite_saturation(self):
        pid = PIDController(
            kp=0.0, ki=1000.0, kd=0.0, dt=1.0, output_limits=(-1.0, 1.0)
        )
        for _ in range(50):
            pid.update(-10.0, 0.0)
        output = pid.update(-10.0, 0.0)
        assert output == pytest.approx(-1.0)
        assert pid.integral == 0.0

    def test_unsaturated_errors_accumulate_normally(self):
        pid = PIDController(
            kp=0.0, ki=10.0, kd=0.0, dt=1.0, output_limits=(-1.0, 1.0)
        )
        pid.update(0.05, 0.0)
        assert pid.integral == pytest.approx(0.05)


class TestClosedLoopBehaviour:
    def test_converges_on_integrator_plant(self):
        pid = PIDController(kp=8.0, ki=2.0, kd=0.05, dt=0.01)
        final_state = simulate(pid, steps=3000)
        assert abs(final_state - 1.0) < 1e-3

    def test_integral_action_rejects_constant_disturbance(self):
        pi = PIDController(kp=5.0, ki=20.0, kd=0.0, dt=0.01)
        p_only = PIDController(kp=5.0, ki=0.0, kd=0.0, dt=0.01)
        pi_state = simulate(pi, disturbance=-0.5, steps=4000)
        p_state = simulate(p_only, disturbance=-0.5, steps=4000)
        assert abs(pi_state - 1.0) < 1e-3
        assert abs(p_state - 1.0) > 0.05

    def test_output_limits_are_respected_in_closed_loop(self):
        pid = PIDController(
            kp=50.0, ki=0.0, kd=0.0, dt=0.01, output_limits=(-0.5, 0.5)
        )
        state = 0.0
        for _ in range(500):
            u = pid.update(1.0, state)
            assert -0.5 <= u <= 0.5
            state += 0.01 * u
        assert abs(state - 1.0) < 1e-6


class TestValidation:
    @pytest.mark.parametrize("dt", [0.0, -1.0])
    def test_non_positive_dt_raises(self, dt):
        with pytest.raises(ValueError, match="dt"):
            PIDController(1.0, 1.0, 1.0, dt=dt)

    def test_inverted_limits_raise(self):
        with pytest.raises(ValueError, match="output_limits"):
            PIDController(1.0, 0.0, 0.0, dt=0.1, output_limits=(2.0, -2.0))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
