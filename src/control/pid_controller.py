"""Discrete PID controller with anti-windup and output saturation.

Implements the parallel form ``u = kp*e + ki*integral(e) + kd*de/dt`` with
conditional-integration anti-windup: the integral term is frozen while the
output is saturated and would be driven further into saturation.
"""

import math


class PIDController:
    """A reusable single-loop PID controller.

    Args:
        kp: Proportional gain.
        ki: Integral gain.
        kd: Derivative gain.
        dt: Fixed control period in seconds, must be positive.
        output_limits: Optional ``(min, max)`` clamp applied to the command.

    Raises:
        ValueError: If ``dt`` is not positive or limits are inverted.
    """

    def __init__(self, kp, ki, kd, dt, output_limits=None):
        if dt <= 0:
            raise ValueError("dt must be positive")
        if output_limits is not None and output_limits[0] >= output_limits[1]:
            raise ValueError("output_limits must be (min, max) with min < max")
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.output_limits = output_limits
        self._integral = 0.0
        self._previous_error = 0.0
        self._has_previous = False

    @property
    def integral(self):
        """Current accumulated integral of the error."""
        return self._integral

    def update(self, setpoint, measurement):
        """Compute the control command for one time step.

        Args:
            setpoint: Desired value of the process variable.
            measurement: Current value of the process variable.

        Returns:
            The clamped control output for this step.
        """
        error = setpoint - measurement
        derivative = 0.0
        if self._has_previous:
            derivative = (error - self._previous_error) / self.dt

        proposed_integral = self._integral + error * self.dt
        raw_output = (
            self.kp * error
            + self.ki * proposed_integral
            + self.kd * derivative
        )
        saturated = self._clamp(raw_output)
        windup_risk = (
            saturated != raw_output
            and (raw_output > 0) == (error > 0)
        )
        if not windup_risk:
            self._integral = proposed_integral
        output = saturated

        self._previous_error = error
        self._has_previous = True
        return output

    def reset(self):
        """Clear all internal state as if the controller were brand new."""
        self._integral = 0.0
        self._previous_error = 0.0
        self._has_previous = False

    def _clamp(self, value):
        if self.output_limits is None:
            return value
        return max(self.output_limits[0], min(self.output_limits[1], value))


if __name__ == "__main__":
    pid = PIDController(kp=2.0, ki=0.5, kd=0.1, dt=0.01, output_limits=(-10.0, 10.0))
    position = 0.0
    velocity = 0.0
    setpoint = 1.0
    for _ in range(2000):
        acceleration = pid.update(setpoint, position) - 2.0 * velocity
        velocity += acceleration * pid.dt
        position += velocity * pid.dt
    print(f"final position={position:.4f} (setpoint {setpoint})")
