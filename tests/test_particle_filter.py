import random

import pytest

from src.estimation.particle_filter import ParticleFilter1D

TRUE_STATE = 5.0


def make_filter(rng, **overrides):
    kwargs = dict(
        num_particles=2000,
        process_noise_std=0.05,
        measurement_noise_std=0.32,
        initial_state_range=(-10.0, 10.0),
        rng=rng,
    )
    kwargs.update(overrides)
    return ParticleFilter1D(**kwargs)


class TestTracking:
    def test_converges_to_constant_truth(self):
        rng = random.Random(99)
        pf = make_filter(rng)
        estimate = 0.0
        for _ in range(150):
            estimate = pf.step(measurement=TRUE_STATE + rng.gauss(0.0, 0.32))
        assert abs(estimate - TRUE_STATE) < 0.25

    def test_filtered_rmse_beats_raw_measurements(self):
        rng = random.Random(2024)
        pf = make_filter(rng)
        squared_raw = squared_filtered = 0.0
        count = 500
        for i in range(count):
            noisy = TRUE_STATE + rng.gauss(0.0, 0.32)
            filtered = pf.step(measurement=noisy)
            if i >= 50:
                squared_raw += (noisy - TRUE_STATE) ** 2
                squared_filtered += (filtered - TRUE_STATE) ** 2
        assert (squared_filtered / count) < (squared_raw / count)

    def test_tracks_drifting_state(self):
        rng = random.Random(31)
        pf = make_filter(
            rng, process_noise_std=0.5, measurement_noise_std=1.0
        )
        truth = -4.0
        worst_error = 0.0
        for _ in range(200):
            truth += 0.05
            estimate = pf.step(measurement=truth + rng.gauss(0.0, 1.0))
            worst_error = max(worst_error, abs(estimate - truth))
        assert worst_error < 2.0

    def test_deterministic_with_same_seed(self):
        def run(seed):
            rng = random.Random(seed)
            pf = make_filter(rng)
            return [
                pf.step(measurement=3.0 + rng.gauss(0.0, 0.32))
                for _ in range(50)
            ]

        assert run(11) == run(11)

    def test_different_seeds_diverge(self):
        def final_estimate(seed):
            rng = random.Random(seed)
            pf = make_filter(rng)
            return pf.step(measurement=rng.uniform(-9.9, 9.9))

        assert final_estimate(1) != final_estimate(2)


class TestWeightedUpdate:
    def test_weights_normalized_after_update(self):
        pf = make_filter(random.Random(5))
        pf.update(2.0)
        assert sum(pf.weights) == pytest.approx(1.0)
        assert all(w >= 0.0 for w in pf.weights)

    def test_measurement_pulls_weighted_mean_toward_itself(self):
        pf = make_filter(
            random.Random(6),
            initial_state_range=(-1.0, 1.0),
            measurement_noise_std=0.5,
        )
        before = pf.estimate
        pf.update(7.0)
        after = pf.estimate
        assert abs(before) < 0.1
        assert after > 0.5

    def test_resample_restores_uniform_weights_and_size(self):
        pf = make_filter(random.Random(8))
        pf.update(2.0)
        pf.resample()
        assert len(pf.particles) == 2000
        assert pf.weights == [1.0 / 2000] * 2000
        assert pf.effective_sample_size == pytest.approx(2000)

    def test_resample_prefers_high_weight_regions(self):
        pf = make_filter(
            random.Random(9), initial_state_range=(-10.0, 10.0)
        )
        pf.predict()
        for _ in range(20):
            pf.update(8.0)
            pf.resample()
        mean_position = sum(pf.particles) / len(pf.particles)
        assert mean_position > 6.0

    def test_ess_bounded_by_population(self):
        pf = make_filter(random.Random(10))
        pf.update(3.0)
        assert 1.0 <= pf.effective_sample_size <= 2000 + 1e-9

    def test_degenerate_weights_fall_back_to_uniform(self):
        pf = make_filter(
            random.Random(12),
            initial_state_range=(0.0, 1.0),
            measurement_noise_std=1e-300,
        )
        pf.weights[0] = 0.0
        pf.update(1000.0)
        assert sum(pf.weights) == pytest.approx(1.0)


class TestValidation:
    @pytest.mark.parametrize("num_particles", [0, -5])
    def test_non_positive_particles_raise(self, num_particles):
        with pytest.raises(ValueError, match="num_particles"):
            make_filter(None, num_particles=num_particles)

    @pytest.mark.parametrize("field", ["process_noise_std", "measurement_noise_std"])
    def test_non_positive_noise_raises(self, field):
        with pytest.raises(ValueError, match=field):
            make_filter(None, **{field: 0.0})

    def test_inverted_initial_range_raises(self):
        with pytest.raises(ValueError, match="initial_state_range"):
            make_filter(None, initial_state_range=(5.0, -5.0))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
