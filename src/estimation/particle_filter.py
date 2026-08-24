"""Bootstrap particle filter for 1D state estimation.

A particle filter approximates the posterior belief over the state with a
cloud of weighted samples. Each cycle:

    1. predict  - propagate every particle through the motion model,
    2. update   - reweight particles by measurement likelihood
                  ``p(z | x) ~ exp(-0.5 * ((z - x) / sigma_z)^2)``,
    3. resample - draw a fresh unweighted population via systematic
                  resampling to fight weight degeneracy.

The weighted mean of the particles is the state estimate.
"""

import math


class ParticleFilter1D:
    """Sequential Monte Carlo estimator for a scalar random-walk state.

    Args:
        num_particles: Population size ``N``, must be positive.
        process_noise_std: Std-dev of the per-step motion noise.
        measurement_noise_std: Std-dev of the sensor noise.
        initial_state_range: ``(lo, hi)`` rectangle from which the initial
            particles are drawn uniformly.
        rng: Optional ``random.Random`` instance for reproducibility.

    Raises:
        ValueError: For non-positive particle counts or noise std-devs.
    """

    def __init__(
        self,
        num_particles,
        process_noise_std,
        measurement_noise_std,
        initial_state_range=(-10.0, 10.0),
        rng=None,
    ):
        if num_particles <= 0:
            raise ValueError("num_particles must be positive")
        if process_noise_std <= 0:
            raise ValueError("process_noise_std must be positive")
        if measurement_noise_std <= 0:
            raise ValueError("measurement_noise_std must be positive")
        lo, hi = initial_state_range
        if lo > hi:
            raise ValueError("initial_state_range must be (lo, hi) with lo <= hi")

        import random

        self._rng = rng if rng is not None else random.Random()
        self.num_particles = int(num_particles)
        self.process_noise_std = float(process_noise_std)
        self.measurement_noise_std = float(measurement_noise_std)
        self.particles = [
            self._rng.uniform(lo, hi) for _ in range(self.num_particles)
        ]
        self.weights = [1.0 / self.num_particles] * self.num_particles

    @property
    def estimate(self):
        """Weighted mean of the particle population."""
        return sum(w * p for w, p in zip(self.weights, self.particles))

    @property
    def effective_sample_size(self):
        """``1 / sum(w_i^2)``; equals ``N`` for uniform weights."""
        return 1.0 / sum(w * w for w in self.weights)

    def predict(self, control=0.0):
        """Advance every particle one step of the motion model."""
        self.particles = [
            p + control + self._rng.gauss(0.0, self.process_noise_std)
            for p in self.particles
        ]

    def update(self, measurement):
        """Multiply weights by the Gaussian likelihood and renormalize."""
        sigma = self.measurement_noise_std
        two_sigma_sq = 2.0 * sigma * sigma
        for i, p in enumerate(self.particles):
            error = measurement - p
            if two_sigma_sq == 0.0:
                likelihood = 1.0 if error == 0.0 else 0.0
            elif error * error / two_sigma_sq > 700.0:
                likelihood = 0.0
            else:
                likelihood = math.exp(-error * error / two_sigma_sq)
            self.weights[i] *= likelihood
        total = sum(self.weights)
        if not (total > 0.0 and math.isfinite(total)):
            uniform = 1.0 / self.num_particles
            self.weights = [uniform] * self.num_particles
            return
        self.weights = [w / total for w in self.weights]

    def resample(self):
        """Systematic resampling; resets all weights to ``1/N``."""
        n = self.num_particles
        cumulative = []
        running = 0.0
        for w in self.weights:
            running += w
            cumulative.append(running)
        offset = self._rng.random() / n
        new_particles = []
        index = 0
        for j in range(n):
            target = offset + j / n
            while cumulative[index] < target:
                index += 1
            new_particles.append(self.particles[index])
        self.particles = new_particles
        self.weights = [1.0 / n] * n

    def step(self, control=0.0, measurement=None):
        """Run one full predict -> update -> resample cycle."""
        self.predict(control)
        if measurement is not None:
            self.update(measurement)
        self.resample()
        return self.estimate


if __name__ == "__main__":
    import random

    rng = random.Random(7)
    pf = ParticleFilter1D(
        2000, process_noise_std=0.05, measurement_noise_std=0.32,
        initial_state_range=(-10.0, 10.0), rng=rng,
    )
    truth = 5.0
    for _ in range(150):
        estimate = pf.step(measurement=truth + rng.gauss(0.0, 0.32))
    print(f"estimate={estimate:.4f} ess={pf.effective_sample_size:.0f}")
