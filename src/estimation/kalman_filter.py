"""Scalar Kalman filter for 1D state estimation.

Estimates a slowly varying scalar state (e.g. battery voltage, range to a
beacon) from noisy measurements using the standard predict/correct cycle:

    predict: ``x`` unchanged, ``P <- P + Q``
    correct: ``K = P / (P + R)``, ``x <- x + K * innovation``, ``P <- (1 - K) P``
"""


class KalmanFilter1D:
    """Optimal recursive estimator for a scalar random-walk process.

    Args:
        process_variance: ``Q``, variance added by the process each step.
        measurement_variance: ``R``, variance of the measurement noise.
        initial_state: Initial estimate of the state.
        initial_variance: Initial uncertainty ``P`` of the estimate.

    Raises:
        ValueError: If any variance argument is negative.
    """

    def __init__(
        self,
        process_variance,
        measurement_variance,
        initial_state=0.0,
        initial_variance=1e6,
    ):
        if process_variance < 0:
            raise ValueError("process_variance must be non-negative")
        if measurement_variance <= 0:
            raise ValueError("measurement_variance must be positive")
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.state = float(initial_state)
        self.variance = float(initial_variance)

    def predict(self):
        """Propagate the state one step, growing uncertainty by ``Q``."""
        self.variance += self.process_variance
        return self.state

    def correct(self, measurement):
        """Fuse one measurement into the current prediction.

        Args:
            measurement: Noisy observation of the true state.

        Returns:
            The posterior state estimate.
        """
        kalman_gain = self.variance / (self.variance + self.measurement_variance)
        innovation = measurement - self.state
        self.state += kalman_gain * innovation
        self.variance *= 1.0 - kalman_gain
        return self.state

    def step(self, measurement):
        """Run one full predict + correct cycle; returns the new estimate."""
        self.predict()
        return self.correct(measurement)

    def filter_sequence(self, measurements):
        """Filter an iterable of measurements, returning all estimates."""
        return [self.step(measurement) for measurement in measurements]


if __name__ == "__main__":
    import random

    rng = random.Random(7)
    kf = KalmanFilter1D(process_variance=1e-5, measurement_variance=0.1)
    truth = 5.0
    for _ in range(100):
        estimate = kf.step(truth + rng.gauss(0.0, 0.32))
    print(f"estimate={estimate:.4f} variance={kf.variance:.6f}")
