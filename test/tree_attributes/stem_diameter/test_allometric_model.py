"""Tests for pointtree.tree_attributes.stem_diameter.StemDiameterAllometricModel."""

import numpy as np
import pytest

from pointtree.tree_attributes.stem_diameter import StemDiameterAllometricModel


def generate_power_law_data(scale: float, exponent: float, noise_std: float = 0.0, seed: int = 0):
    """
    Generates synthetic tree height, crown width, and stem diameter data that follows the power-law relationship
    :code:`D = scale * (H * CD) ** exponent`, optionally perturbed by log-normal noise.

    Args:
        scale: Scale parameter :math:`a` of the power-law model used to generate the stem diameters.
        exponent: Exponent :math:`b` of the power-law model used to generate the stem diameters.
        noise_std: Standard deviation of the log-normal noise added to the stem diameters. If :code:`0.0`, the
            stem diameters follow the power-law relationship exactly.
        seed: Seed for the random number generator.

    Returns:
        :Tuple of three elements:
            - Height of each tree.
            - Crown width of each tree.
            - Stem diameter of each tree.
    """

    rng = np.random.default_rng(seed)
    tree_heights = rng.uniform(5.0, 40.0, 2000)
    crown_widths = rng.uniform(1.0, 15.0, 2000)
    noise = rng.normal(0.0, noise_std, len(tree_heights)) if noise_std > 0.0 else 0.0
    stem_diameters = scale * (tree_heights * crown_widths) ** exponent * np.exp(noise)
    return tree_heights, crown_widths, stem_diameters


class TestStemDiameterAllometricModel:
    """Tests for pointtree.tree_attributes.stem_diameter.StemDiameterAllometricModel."""

    def test_not_fitted_by_default(self):
        model = StemDiameterAllometricModel()

        assert model.scale is None
        assert model.exponent is None
        assert model.correction_factor is None
        assert model.log_residual_variance is None

    def test_predict_before_fit_raises(self):
        model = StemDiameterAllometricModel()

        with pytest.raises(RuntimeError):
            model.predict(np.array([10.0]), np.array([2.0]))

    def test_fit(self):
        scale = 0.6
        exponent = 0.8
        tree_heights, crown_widths, stem_diameters = generate_power_law_data(scale, exponent)

        model = StemDiameterAllometricModel().fit(tree_heights, crown_widths, stem_diameters)

        assert model.scale == pytest.approx(scale, abs=4)
        assert model.exponent == pytest.approx(exponent, abs=4)
        assert model.correction_factor == pytest.approx(1.0, abs=4)
        assert model.log_residual_variance == pytest.approx(0.0, abs=4)

        predictions = model.predict(tree_heights, crown_widths)

        np.testing.assert_array_almost_equal(predictions, stem_diameters, decimal=2)

    def test_predict_apply_correction_factor(self):
        tree_heights, crown_widths, stem_diameters = generate_power_law_data(0.6, 0.8, noise_std=0.3)

        model = StemDiameterAllometricModel().fit(tree_heights, crown_widths, stem_diameters)

        assert model.correction_factor is not None
        assert model.correction_factor > 1.0

        prediction_with_cf = model.predict(tree_heights, crown_widths, apply_correction_factor=True)
        prediction_without_cf = model.predict(tree_heights, crown_widths, apply_correction_factor=False)

        np.testing.assert_array_almost_equal(prediction_with_cf, prediction_without_cf * model.correction_factor)

    def test_num_bins_is_configurable(self):
        scale = 0.6
        exponent = 0.8
        tree_heights, crown_widths, stem_diameters = generate_power_law_data(scale, exponent)

        model = StemDiameterAllometricModel(num_bins=10).fit(tree_heights, crown_widths, stem_diameters)

        assert model.scale == pytest.approx(scale, abs=4)
        assert model.exponent == pytest.approx(exponent, rel=4)

    def test_non_positive_num_bins(self):
        tree_heights, crown_widths, stem_diameters = generate_power_law_data(0.6, 0.8)

        with pytest.raises(ValueError):
            StemDiameterAllometricModel(num_bins=0).fit(tree_heights, crown_widths, stem_diameters)

    def test_mismatched_input_lengths(self):
        with pytest.raises(ValueError):
            StemDiameterAllometricModel().fit(
                np.array([10.0, 20.0, 30.0]), np.array([2.0, 3.0]), np.array([0.2, 0.3, 0.4])
            )

    def test_empty_input(self):
        empty = np.empty(0, dtype=np.float64)
        with pytest.raises(ValueError):
            StemDiameterAllometricModel().fit(empty, empty, empty)

    def test_non_positive_input_values(self):
        tree_heights, crown_widths, stem_diameters = generate_power_law_data(0.6, 0.8)
        stem_diameters = stem_diameters.copy()
        stem_diameters[0] = 0.0

        with pytest.raises(ValueError):
            StemDiameterAllometricModel().fit(tree_heights, crown_widths, stem_diameters)

    def test_too_few_non_empty_bins(self):
        with pytest.raises(ValueError):
            StemDiameterAllometricModel(num_bins=50).fit(
                np.array([10.0, 20.0]), np.array([2.0, 3.0]), np.array([0.2, 0.3])
            )
