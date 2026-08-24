import random

import pytest

from src.estimation.kalman_filter import KalmanFilter1D

PROCESS_VARIANCE = 1e-5
MEASUREMENT_VARIANCE = 0.1
TRUE_STATE = 5.0


class TestKalmanFilter1D:
    def test_estimate_converges_to_constant_truth(self):
        rng = random.Random(123)
        kf = KalmanFilter1D(PROCESS_VARIANCE, MEASUREMENT_VARIANCE)
        estimate = 0.0
        for _ in range(200):
            estimate = kf.step(TRUE_STATE + rng.gauss(0.0, MEASUREMENT_VARIANCE**0.5))
        assert estimate == pytest.approx(TRUE_STATE, abs=0.1)

    def test_filtered_rmse_beats_raw_measurements(self):
        rng = random.Random(2024)
        noise_sigma = 0.32
        kf = KalmanFilter1D(PROCESS_VARIANCE, MEASUREMENT_VARIANCE)
        squared_raw = squared_filtered = 0.0
        count = 500
        for i in range(count):
            noisy = TRUE_STATE + rng.gauss(0.0, noise_sigma)
            filtered = kf.step(noisy)
            if i >= 50:
                squared_raw += (noisy - TRUE_STATE) ** 2
                squared_filtered += (filtered - TRUE_STATE) ** 2
        assert (squared_filtered / count) < (squared_raw / count)

    def test_posterior_variance_matches_recurrence_and_shrinks(self):
        kf = KalmanFilter1D(PROCESS_VARIANCE, MEASUREMENT_VARIANCE, initial_variance=100.0)
        p = 100.0
        previous = None
        for measurement in [1.0, 2.0, 3.0, 4.0]:
            kf.predict()
            kf.correct(measurement)
            predicted = p + PROCESS_VARIANCE
            gain = predicted / (predicted + MEASUREMENT_VARIANCE)
            p = (1.0 - gain) * predicted
            assert kf.variance == pytest.approx(p)
            if previous is not None:
                assert kf.variance < previous
            previous = kf.variance

    def test_high_initial_uncertainty_trusts_first_measurement(self):
        kf = KalmanFilter1D(
            PROCESS_VARIANCE, MEASUREMENT_VARIANCE, initial_state=0.0, initial_variance=1e6
        )
        estimate = kf.step(7.5)
        assert estimate == pytest.approx(7.5, abs=1e-3)

    def test_low_initial_uncertainty_ignores_first_measurement(self):
        kf = KalmanFilter1D(
            process_variance=0.0,
            measurement_variance=MEASUREMENT_VARIANCE,
            initial_state=2.0,
            initial_variance=1e-12,
        )
        estimate = kf.step(50.0)
        assert estimate == pytest.approx(2.0, abs=1e-6)

    def test_step_equals_predict_then_correct(self):
        combined = KalmanFilter1D(PROCESS_VARIANCE, MEASUREMENT_VARIANCE, initial_state=1.0)
        split = KalmanFilter1D(PROCESS_VARIANCE, MEASUREMENT_VARIANCE, initial_state=1.0)
        for measurement in (3.0, -1.0, 8.0):
            via_step = combined.step(measurement)
            split.predict()
            via_split = split.correct(measurement)
            assert via_step == pytest.approx(via_split)
            assert combined.variance == pytest.approx(split.variance)

    def test_filter_sequence_returns_per_step_estimates(self):
        measurements = [1.0, 1.2, 0.8, 1.1]
        batch = KalmanFilter1D(PROCESS_VARIANCE, MEASUREMENT_VARIANCE).filter_sequence(
            measurements
        )
        incremental = KalmanFilter1D(PROCESS_VARIANCE, MEASUREMENT_VARIANCE)
        manual = [incremental.step(z) for z in measurements]
        assert len(batch) == len(measurements)
        assert batch == pytest.approx(manual)

    @pytest.mark.parametrize("process_variance", [-1e-3, -1.0])
    def test_negative_process_variance_raises(self, process_variance):
        with pytest.raises(ValueError, match="process_variance"):
            KalmanFilter1D(process_variance, MEASUREMENT_VARIANCE)

    @pytest.mark.parametrize("measurement_variance", [0.0, -0.5])
    def test_non_positive_measurement_variance_raises(self, measurement_variance):
        with pytest.raises(ValueError, match="measurement_variance"):
            KalmanFilter1D(PROCESS_VARIANCE, measurement_variance)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
