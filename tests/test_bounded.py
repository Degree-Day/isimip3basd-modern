import numpy as np

from isimip3basd_modern.bounded import _adjust_window


def test_bounded_humidity_matches_canonical_isimip_regression_vector():
    observed = np.array([5, 15, 25, 40, 55, 70, 80, 90, 95, 99.0])
    historical = np.array([8, 18, 30, 48, 65, 82, 95, 101, 104, 110.0])
    simulation = np.array(
        [6, 20, 35, 50, 68, 85, 98, 103, 108, 115, 72, 100.5]
    )
    expected = np.array(
        [
            3.75,
            16.266575854700857,
            29.677192841880345,
            42.63793227307152,
            58.05585632370443,
            72.66235205777792,
            87.96663231062661,
            93.42852212792297,
            98.6217086618368,
            99.0000393032093,
            61.489093827186174,
            91.26262116002363,
        ]
    )

    actual = _adjust_window(
        observed,
        historical,
        simulation,
        quantiles=5,
        seed=0,
    )

    np.testing.assert_array_equal(actual, expected)
