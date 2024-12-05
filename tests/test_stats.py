"""Tests for VectoRose stats.

This module contains the automated tests for the :mod:`vectorose.stats`
module. The statistical methods are based on definitions and descriptions
provided by Fisher, Lewis and Embleton. [#fisher-lewis-embleton]_

As these tests rely on statistics which are very difficult to compute by
hand, we often will not test for specific numerical results. Instead, we
will be assessing whether the computed results follow expected trends and
patterns. In some cases, as the data are derived from a defined
distribution which takes certain parameters, we will be working to
determine whether estimates are indeed close (but not necessarily equal) to
the original parameters. In other cases, we will be checking to see whether
hypotheses are rejected or not, without necessarily checking to see if the
value of the test statistic is equal to some precise pre-determined value.
The individual tests will detail these subtleties in their documentation.

References
----------
.. [#fisher-lewis-embleton] Fisher, N. I., Lewis, T., & Embleton, B. J.
       J. (1993). Statistical analysis of spherical data ([New ed.], 1.
       paperback ed). Cambridge Univ. Press.

"""
import functools

import numpy as np
import scipy as sp
import vectorose as vr
import vectorose.mock_data

RANDOM_SEED = 20241205


# Test the resultant vector computation
def test_compute_resultant_vector_aligned():
    """Test to compute the resultant vector for aligned data.

    Unit test for :func:`stats.compute_resultant_vector`. Compute the
    resultant vector for perfectly aligned data. In this case, the
    resultant vector should have the same orientation as the constituent
    vector, but be ``n`` times longer for ``n`` vectors.
    """

    unit_vector = np.array([0, 1, 0])

    my_collection_of_vectors = np.tile(unit_vector, (10, 1))

    resultant_vector = vr.stats.compute_resultant_vector(
        my_collection_of_vectors, compute_mean_resultant=False
    )

    # Check the magnitude
    resultant_length = np.linalg.norm(resultant_vector)

    # The resultant length should be 10
    assert resultant_length == 10

    # Now, find the direction by taking the unit vector
    resultant_direction = resultant_vector / resultant_length

    assert np.all(resultant_direction == unit_vector)


def test_compute_mean_resultant_vector_aligned():
    """Test to compute the resultant vector for aligned data.

    Unit test for :func:`stats.compute_resultant_vector`. Compute the
    resultant vector for perfectly aligned data. In this case, the mean
    resultant vector should have the same orientation as the constituent
    vector and should have length 1.
    """

    unit_vector = np.array([0, 1, 0])

    my_collection_of_vectors = np.tile(unit_vector, (10, 1))

    resultant_vector = vr.stats.compute_resultant_vector(
        my_collection_of_vectors, compute_mean_resultant=True
    )

    # Check the magnitude
    resultant_length = np.linalg.norm(resultant_vector)

    # The resultant length should be 10
    assert resultant_length == 1

    # Now, find the direction by taking the unit vector
    resultant_direction = resultant_vector / resultant_length

    assert np.all(resultant_direction == unit_vector)


def test_compute_resultant_vector_opposite():
    """Test to compute the resultant vector for opposite data.

    Unit test for :func:`stats.compute_resultant_vector`. Compute the
    resultant vector for perfectly opposite-aligned data. In this case, the
    resultant vector should have zero length, as all vectors should cancel
    each other out.
    """

    my_collection_of_vectors = np.array(
        [
            [0, 1, 0],
            [0, -1, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [np.sqrt(1 / 2), -np.sqrt(1 / 2), 0],
            [-np.sqrt(1 / 2), np.sqrt(1 / 2), 0],
        ]
    )

    resultant_vector = vr.stats.compute_resultant_vector(
        my_collection_of_vectors, compute_mean_resultant=False
    )

    # Check the magnitude
    resultant_length = np.linalg.norm(resultant_vector)

    # The resultant length should be 10
    assert resultant_length == 0


def test_compute_mean_resultant_vector_opposite():
    """Test to compute the resultant vector for opposite data.

    Unit test for :func:`stats.compute_resultant_vector`. Compute the
    resultant vector for perfectly opposite data. In this case, the mean
    resultant vector should have length 0 as all the data cancel each other
    out.
    """

    my_collection_of_vectors = np.array(
        [
            [0, 1, 0],
            [0, -1, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [np.sqrt(1 / 2), -np.sqrt(1 / 2), 0],
            [-np.sqrt(1 / 2), np.sqrt(1 / 2), 0],
        ]
    )

    resultant_vector = vr.stats.compute_resultant_vector(
        my_collection_of_vectors, compute_mean_resultant=True
    )

    # Check the magnitude
    resultant_length = np.linalg.norm(resultant_vector)

    # The resultant length should be 10
    assert resultant_length == 0


# Test the function for finding Woodcock's parameters
def test_woodcock_parameters_uniform():
    """Test for Woodcock's strength parameter in the uniform distribution.

    Unit test for :func:`stats.compute_orientation_matrix_parameters` which
    implements Woodcock's shape and strength parameters.

    In this test, we check that it indeed produces a low, near-zero value
    for the strength parameter when uniform data is provided.
    """

    # Generate vector data
    vectors = sp.stats.uniform_direction(dim=3, seed=RANDOM_SEED).rvs(size=1000000)

    # Compute the orientation matrix eigenvectors and eigenvalues
    orientation_matrix_eigs = vr.stats.compute_orientation_matrix_eigs(vectors)

    # Compute the shape and strength parameters
    woodcock_parameters = vr.stats.compute_orientation_matrix_parameters(
        orientation_matrix_eigs.eigenvalues
    )

    # Check the strength parameter
    assert woodcock_parameters.strength_parameter < 0.5


def test_woodcock_parameters_vmf():
    """Test for Woodcock's parameters in the von Mises-Fisher distribution.

    Unit test for :func:`stats.compute_orientation_matrix_parameters` which
    implements Woodcock's shape and strength parameters.

    In this test, we check that this function produces a high strength
    parameter value and a shape parameter value above 1 for concentrated
    data generated by a von Mises-Fisher distribution.
    """

    # Generate vector data
    vectors = sp.stats.vonmises_fisher(mu=[0, 0, 1], kappa=100, seed=RANDOM_SEED).rvs(
        size=1000000
    )

    # Compute the orientation matrix eigenvectors and eigenvalues
    orientation_matrix_eigs = vr.stats.compute_orientation_matrix_eigs(vectors)

    # Compute the shape and strength parameters
    woodcock_parameters = vr.stats.compute_orientation_matrix_parameters(
        orientation_matrix_eigs.eigenvalues
    )

    # Check the strength parameter
    assert woodcock_parameters.strength_parameter > 0.5

    # Check the shape parameter
    assert woodcock_parameters.shape_parameter > 1


def test_woodcock_parameters_watson():
    """Test for Woodcock's parameters in the Watson distribution.

    Unit test for :func:`stats.compute_orientation_matrix_parameters` which
    implements Woodcock's shape and strength parameters.

    In this test, we check that this function produces a high strength
    parameter value and a shape parameter value below 1 for data in a
    compact girdle generated using a Watson distribution.
    """

    # Generate vector data
    vectors = vr.mock_data.generate_watson_distribution(
        mean_direction=np.array([0, 0, 1]), kappa=-100, n=1000000, seed=RANDOM_SEED
    )

    # Compute the orientation matrix eigenvectors and eigenvalues
    orientation_matrix_eigs = vr.stats.compute_orientation_matrix_eigs(vectors)

    # Compute the shape and strength parameters
    woodcock_parameters = vr.stats.compute_orientation_matrix_parameters(
        orientation_matrix_eigs.eigenvalues
    )

    # Check the strength parameter
    assert woodcock_parameters.strength_parameter > 0.5

    # Check the shape parameter
    assert woodcock_parameters.shape_parameter < 1


# Test the function for hypothesis testing of uniform vs. unimodal
def test_uniform_unimodal_uniform():
    """Test for uniformity vs. unimodality in uniform data.

    Unit test for :func:`stats.uniform_vs_unimodal_test` which performs
    hypothesis testing to determine whether a distribution in uniform over
    the surface of the sphere (null hypothesis) or unimodal (alternate
    hypothesis).

    This test verifies that for uniform data, the null hypothesis is not
    rejected and the p-value is above 0.05.
    """

    # Generate vector data
    vectors = sp.stats.uniform_direction(dim=3, seed=RANDOM_SEED).rvs(size=1000000)

    # Perform the hypothesis testing
    result = vr.stats.uniform_vs_unimodal_test(vectors, significance_level=0.05)

    # Check the p-value
    assert result.p_value > 0.05

    # Check that we can't reject the null
    assert not result.can_reject_null_hypothesis


def test_uniform_unimodal_unimodal():
    """Test for uniformity vs. unimodality in unimodal data.

    Unit test for :func:`stats.uniform_vs_unimodal_test` which performs
    hypothesis testing to determine whether a distribution in uniform over
    the surface of the sphere (null hypothesis) or unimodal (alternate
    hypothesis).

    This test verifies that for unimodal data, the null hypothesis is
    indeed rejected and the p-value is below 0.05.
    """

    # Generate vector data
    vectors = sp.stats.vonmises_fisher(mu=[0, 0, 1], kappa=100, seed=RANDOM_SEED).rvs(
        size=1000000
    )

    # Perform the hypothesis testing
    result = vr.stats.uniform_vs_unimodal_test(vectors, significance_level=0.05)

    # Check the p-value
    assert result.p_value < 0.05

    # Check that we can't reject the null
    assert result.can_reject_null_hypothesis


# Test the Median Direction
def test_median_direction():
    """Test for computing the spherical median direction.

    Unit test for :func:`stats.compute_median_direction`. The spherical
    median is defined as the vector that minimises the sum of arc lengths
    to all other vectors. We can't directly test this, but we can get a
    rough idea of the result by generating a number of vectors uniformly
    on the surface of the sphere and testing to ensure that the sum of
    arc lengths is indeed lower in the median than in all the other
    vectors.
    """

    # Generate vector data
    vectors = sp.stats.vonmises_fisher(mu=[0, 0, 1], kappa=100, seed=RANDOM_SEED).rvs(
        size=1000
    )

    # Compute the spherical median
    spherical_median = vr.stats.compute_median_direction(vectors)

    # Generate many uniformly-distributed vectors
    uniform_vectors = sp.stats.uniform_direction(dim=3, seed=RANDOM_SEED).rvs(
        size=100000
    )

    # Compute the sum of arc lengths for each vector against the distribution
    compute_sum_of_arc_lengths_to_vectors = functools.partial(
        vr.stats._compute_sum_of_arc_lengths, vectors=vectors
    )

    uniform_sum_of_arc_lengths = np.apply_along_axis(
        compute_sum_of_arc_lengths_to_vectors, axis=1, arr=uniform_vectors
    )

    # And now compute it for the spherical median
    spherical_median_sum_of_arc_lengths = compute_sum_of_arc_lengths_to_vectors(
        spherical_median
    )

    assert spherical_median_sum_of_arc_lengths < uniform_sum_of_arc_lengths.min()


# Test parameter estimation for the von Mises-Fisher distribution
def test_mean_direction_vmf():
    """Test to compute the mean of a von Mises-Fisher distribution.

    Unit test for :func:`stats.compute_mean_direction` when the provided
    data are drawn from a von Mises-Fisher distribution.
    """

    # Define the true mean direction
    mu = [np.sqrt(1 / 2), -np.sqrt(1 / 2), 0]
    kappa = 20

    # Generate vector data
    vectors = sp.stats.vonmises_fisher(mu=mu, kappa=kappa, seed=RANDOM_SEED).rvs(
        size=1000000
    )

    # Estimate the mean direction
    estimated_mu = vr.stats.compute_mean_unit_direction(vectors)

    # Check if they are close
    assert np.all(np.isclose(estimated_mu, mu, atol=1e-3, rtol=1e-2))


def test_concentration_parameter_estimate():
    """Test to compute the VMF concentration parameter.

    Unit test for :func:`stats.estimate_concentration_parameter` when the
    provided data are drawn from a von Mises-Fisher distribution.
    """

    # Define the true mean direction
    mu = [np.sqrt(1 / 2), -np.sqrt(1 / 2), 0]
    kappa = 20

    # Generate vector data
    vectors = sp.stats.vonmises_fisher(mu=mu, kappa=kappa, seed=RANDOM_SEED).rvs(
        size=1000000
    )

    # Estimate the concentration parameter
    estimated_kappa = vr.stats.estimate_concentration_parameter(vectors)

    # Check if they are close
    assert np.isclose(estimated_kappa, kappa, atol=1e-3, rtol=1e-2)


def test_vmf_parameter_estimation():
    """Test for the VMF parameter estimation.

    Unit test for :func:`stats.fit_fisher_vonmises_distribution` which
    wraps the ``fit`` method of :obj:`scipy.stats.vonmises_fisher`.
    """

    # Define the true mean direction
    mu = [np.sqrt(1 / 2), -np.sqrt(1 / 2), 0]
    kappa = 20

    # Generate vector data
    vectors = sp.stats.vonmises_fisher(mu=mu, kappa=kappa, seed=RANDOM_SEED).rvs(
        size=1000000
    )

    # Estimate the parameters
    parameter_estimates = vr.stats.fit_fisher_vonmises_distribution(vectors)

    # Compare both
    assert np.all(np.isclose(parameter_estimates.mu, mu, atol=1e-3, rtol=1e-2))
    assert np.isclose(parameter_estimates.kappa, kappa, atol=1e-3, rtol=1e-2)


# Test the magnitude-orientation correlation
def test_magnitude_orientation_correlation_independent():
    """Test the magnitude and orientation correlation for independent data.

    Unit test for :func:`stats.compute_magnitude_orientation_correlation`
    where the data have been engineered so that the magnitudes and
    orientations are completely independent.

    In this test, as the two distributions are constructed separately, the
    null hypothesis that the data are not correlated should not be rejected
    and the p-value computed should be above 0.05.

    Note that we don't check the actual value of the correlation
    coefficient and we don't check the test statistic. We would have to
    compute these values by hand (which would probably give the exact same
    results). Instead, we rely on the hypothesis result and the p-value.
    """

    n = 1000000

    # Generate vectors
    vectors = sp.stats.vonmises_fisher(mu=[0, 0, 1], kappa=100, seed=RANDOM_SEED).rvs(
        size=n
    )

    magnitudes = np.random.default_rng(RANDOM_SEED).uniform(1e-5, 1, size=n)

    non_unit_vectors = vectors * magnitudes[:, None]

    (
        correlation_coefficient,
        result,
    ) = vr.stats.compute_magnitude_orientation_correlation(
        non_unit_vectors, significance_level=0.05
    )

    assert result.p_value > 0.05
    assert not result.can_reject_null_hypothesis


def test_magnitude_orientation_correlation_dependent():
    """Test the magnitude and orientation correlation for dependent data.

    Unit test for :func:`stats.compute_magnitude_orientation_correlation`
    where the data have been engineered so that the magnitudes and
    orientations are not independent. Different magnitudes are set in
    different regions to ensure that there is a dependent relationship.

    In this test, as the two distributions are constructed together, the
    null hypothesis that the data are not correlated should be rejected
    and the p-value computed should be below 0.05.

    Note that we don't check the actual value of the correlation
    coefficient and we don't check the test statistic. We would have to
    compute these values by hand (which would probably give the exact same
    results). Instead, we rely on the hypothesis result and the p-value.
    """

    n = 1000000

    # Generate vectors
    vectors = vr.mock_data.create_von_mises_fisher_vectors_multiple_directions(
        phis=[45, 60, 10],
        thetas=[0, 120, 160],
        kappas=[10, 50, 5],
        numbers_of_vectors=n,
        magnitudes=[0.4, 0.7, 0.2],
        magnitude_stds=[0.05, 0.1, 0.01],
        use_degrees=True,
        seeds=[RANDOM_SEED, RANDOM_SEED, RANDOM_SEED],
    )

    (
        correlation_coefficient,
        result,
    ) = vr.stats.compute_magnitude_orientation_correlation(
        vectors, significance_level=0.05
    )

    assert result.p_value < 0.05
    assert result.can_reject_null_hypothesis
