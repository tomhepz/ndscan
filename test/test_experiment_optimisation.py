"""Tests for optional host-runtime optimiser backends."""

import unittest

try:
    import torch
    from mock_environment import HasEnvironmentCase

    from ndscan.experiment import (
        AskTellOptimiserPointSource,
        ExecutionPolicy,
        ExpFragment,
        ExplicitBatchExplorationStrategy,
        FloatChannel,
        FloatParam,
        HostScanSession,
        LocalLengthscaleExplorationStrategy,
        MhcsExplorationStrategy,
        NuboBatchBayesianOptimisationBackend,
        NuboBayesianOptimisationState,
        OptimiserObservation,
        ScanRequest,
        ScheduledExplorationStrategy,
        extract_scalar_channel_objective,
    )

    _OPTIMISATION_DEPS_AVAILABLE = True
except ModuleNotFoundError:
    _OPTIMISATION_DEPS_AVAILABLE = False


@unittest.skipUnless(
    _OPTIMISATION_DEPS_AVAILABLE,
    "Optional Bayesian optimisation dependencies are not installed",
)
class OptimisationStrategyTest(unittest.TestCase):
    def setUp(self):
        self.bounds_2d = torch.tensor(
            [[0.0, 0.0], [1.0, 1.0]],
            dtype=torch.get_default_dtype(),
        )

    def test_scheduled_explicit_strategy_only_emits_on_selected_batches(self):
        strategy = ScheduledExplorationStrategy(
            every_batches=3,
            strategy=ExplicitBatchExplorationStrategy(
                {
                    0: [(0.25, 0.75)],
                    3: [(0.75, 0.25)],
                }
            ),
        )

        def state(batch_index: int) -> NuboBayesianOptimisationState:
            return NuboBayesianOptimisationState(
                observed_points=torch.zeros((0, 2), dtype=self.bounds_2d.dtype),
                observed_objective_scores=torch.zeros((0,), dtype=self.bounds_2d.dtype),
                observed_noise_std=torch.zeros((0,), dtype=self.bounds_2d.dtype),
                gp=None,
                bounds=self.bounds_2d,
                batch_index=batch_index,
            )

        self.assertTrue(
            torch.allclose(
                strategy(state(0), 2),
                torch.tensor([[0.25, 0.75]], dtype=self.bounds_2d.dtype),
            )
        )
        self.assertEqual(strategy(state(1), 2).shape, (0, 2))
        self.assertTrue(
            torch.allclose(
                strategy(state(3), 2),
                torch.tensor([[0.75, 0.25]], dtype=self.bounds_2d.dtype),
            )
        )

    def test_mhcs_strategy_finds_central_uncovered_region(self):
        strategy = MhcsExplorationStrategy(
            num_points=1,
            pair_batch_size=32,
            interior_tol=1e-9,
            min_normalised_distance=0.0,
        )
        state = NuboBayesianOptimisationState(
            observed_points=torch.tensor(
                [
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                ],
                dtype=self.bounds_2d.dtype,
            ),
            observed_objective_scores=torch.zeros((4,), dtype=self.bounds_2d.dtype),
            observed_noise_std=torch.full((4,), 0.01, dtype=self.bounds_2d.dtype),
            gp=None,
            bounds=self.bounds_2d,
            batch_index=0,
        )

        points = strategy(state, 1)
        self.assertEqual(points.shape, (1, 2))
        self.assertTrue(torch.allclose(points[0], torch.tensor([0.5, 0.5]), atol=1e-6))


@unittest.skipUnless(
    _OPTIMISATION_DEPS_AVAILABLE,
    "Optional Bayesian optimisation dependencies are not installed",
)
class BayesianOptimisationBackendTest(unittest.TestCase):
    def test_backend_returns_seed_points_then_bo_batch_with_exploration(self):
        backend = NuboBatchBayesianOptimisationBackend(
            bounds=[[-1.0], [1.0]],
            batch_size=3,
            initial_points=[[-1.0], [1.0]],
            random_seed=0,
            fit_steps=2,
            acquisition_num_starts=1,
            surrogate_num_starts=1,
            batch_mc_samples=8,
            batch_acq_steps=2,
            max_batches=1,
            exploration_strategy=ScheduledExplorationStrategy(
                every_batches=1,
                strategy=ExplicitBatchExplorationStrategy({0: [[0.25]]}),
            ),
        )

        seed_points = backend.suggest(3)
        self.assertEqual(seed_points, [(-1.0,), (1.0,)])

        backend.observe(
            (
                OptimiserObservation(point=(-1.0,), objective=1.0, noise_std=0.01),
                OptimiserObservation(point=(1.0,), objective=1.0, noise_std=0.01),
            )
        )

        next_batch = backend.suggest(3)
        self.assertGreaterEqual(len(next_batch), 1)
        self.assertLessEqual(len(next_batch), 3)
        self.assertEqual(next_batch[0], (0.25,))
        for point in next_batch:
            self.assertGreaterEqual(point[0], -1.0)
            self.assertLessEqual(point[0], 1.0)

    def test_backend_supports_local_exploration_strategy(self):
        backend = NuboBatchBayesianOptimisationBackend(
            bounds=[[-1.0], [1.0]],
            batch_size=2,
            initial_points=[[-1.0], [0.0], [1.0]],
            random_seed=0,
            fit_steps=2,
            acquisition_num_starts=1,
            surrogate_num_starts=1,
            batch_mc_samples=8,
            batch_acq_steps=2,
            max_batches=1,
            exploration_strategy=LocalLengthscaleExplorationStrategy(
                num_points=1,
                axis_points=5,
                pair_points=3,
                span_in_lengthscales=1.0,
                min_normalised_distance=0.01,
                surrogate_num_starts=1,
            ),
        )

        _ = backend.suggest(3)
        backend.observe(
            (
                OptimiserObservation(point=(-1.0,), objective=1.0, noise_std=0.01),
                OptimiserObservation(point=(0.0,), objective=0.0, noise_std=0.01),
                OptimiserObservation(point=(1.0,), objective=1.0, noise_std=0.01),
            )
        )

        next_batch = backend.suggest(2)
        self.assertGreaterEqual(len(next_batch), 1)
        for point in next_batch:
            self.assertGreaterEqual(point[0], -1.0)
            self.assertLessEqual(point[0], 1.0)


if _OPTIMISATION_DEPS_AVAILABLE:

    class OneDimQuadraticFragment(ExpFragment):
        def build_fragment(self):
            self.setattr_param("x", FloatParam, "x", default=0.0)
            self.setattr_result("cost", FloatChannel)

        def run_once(self):
            self.cost.push((self.x.get() - 0.3) ** 2)


    @unittest.skipUnless(
        _OPTIMISATION_DEPS_AVAILABLE,
        "Optional Bayesian optimisation dependencies are not installed",
    )
    class BayesianOptimisationRuntimeTest(HasEnvironmentCase):
        def test_host_runtime_runs_nubo_bayesian_optimisation_batches(self):
            fragment = self.create(OneDimQuadraticFragment, [])
            backend = NuboBatchBayesianOptimisationBackend(
                bounds=[[-1.0], [1.0]],
                batch_size=2,
                initial_points=[[-1.0], [1.0]],
                random_seed=0,
                fit_steps=2,
                acquisition_num_starts=1,
                surrogate_num_starts=1,
                batch_mc_samples=8,
                batch_acq_steps=2,
                max_batches=1,
            )
            request = ScanRequest(
                axes=(fragment.x,),
                point_source=AskTellOptimiserPointSource(
                    backend,
                    extract_scalar_channel_objective("channel_0"),
                ),
                execution_policy=ExecutionPolicy(max_points_per_batch=2),
            )

            result = HostScanSession(fragment, fragment, request).run()

            prefix = result.site_prefix
            self.assertEqual(self.dataset_db.get(prefix + "state.num_points"), 4)
            self.assertEqual(len(self.dataset_db.get(prefix + "points.channel_0")), 4)
            self.assertEqual(backend.describe()["kind"], "nubo_bayesian_optimisation")


if __name__ == "__main__":
    unittest.main()
