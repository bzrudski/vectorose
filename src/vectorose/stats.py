"""Statistical analyses.

Statistical tests, analyses and routines for analysing the directional
data used to construct the VectoRose plots.

Warnings
--------
For many of the statistical operations defined here, a set of *unit
vectors* is required in order for interpretation to be possible. To produce
a set of unit vectors, the function :func:`.util.normalise_vectors` can be
called.

The vectors passed to these functions should not contain spatial location
coordinates.

Notes
-----
These statistical tests are largely derived from the work by Fisher, Lewis
and Embleton. [#fisher-lewis-embleton]_

References
----------
.. [#fisher-lewis-embleton] Fisher, N. I., Lewis, T., & Embleton, B. J.
       J. (1993). Statistical analysis of spherical data ([New ed.], 1.
       paperback ed). Cambridge Univ. Press.

.. [#woodcock-1977] Woodcock, N. H. (1977). Specification of fabric shapes
       using an eigenvalue method. Geological Society of America Bulletin,
       88(9), 1231.
       https://doi.org/10.1130/0016-7606(1977)88<1231:SOFSUA>2.0.CO;2

"""
import dataclasses
import functools
from typing import NamedTuple, Optional, Tuple

from scipy.optimize import NonlinearConstraint, minimize, fsolve
from scipy.stats import chi2, vonmises_fisher

from . import util

import numpy as np


# Define result holder
@dataclasses.dataclass
class HypothesisResult:
    """Results for hypothesis testing."""

    can_reject_null_hypothesis: bool
    """Indicate whether the null hypothesis can be rejected."""

    p_value: float
    """Computed p-value for the test."""

    test_significance: float
    """Significance level used for the test."""


def compute_resultant_vector(
    vector_field: np.ndarray,
    compute_mean_resultant: bool = True,
) -> np.ndarray:
    """Compute the resultant vector for a set of orientations.

    Compute the resultant vector from a set of orientations or direction.
    This vector is computed as the sum of all constituent vectors.

    Parameters
    ----------
    vector_field
        The vector field to consider, represented as either an array of
        shape ``(n, d)`` or an ``n+1``-dimensional array containing the
        components at their spatial locations, with the components present
        along the *last* axis.
    compute_mean_resultant
        Indicate whether the mean resultant should be returned instead of
        the non-normalised resultant vector.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(d,)`` containing the resultant vector. If
        `compute_mean_resultant` is `True`, then this is the mean resultant
        vector.

    Notes
    -----
    This implementation is based on the description in chapter 3 of Fisher,
    Lewis and Embleton's book [#fisher-lewis-embleton]_ on statistics on
    the sphere.

    """

    # Check the shape of the input, reshape if necessary.
    d = vector_field.ndim

    if d > 2:
        vector_dimension = vector_field.shape[-1]
        stacked_vectors = vector_field.reshape(-1, vector_dimension)
    else:
        stacked_vectors = vector_field.copy()

    # Now, we need to sum all the components
    resultant_vector = stacked_vectors.sum(axis=0).astype(float)

    if compute_mean_resultant:
        n = len(stacked_vectors)
        resultant_vector /= n

    return resultant_vector


def compute_orientation_matrix(
    vectors: np.ndarray
) -> np.ndarray:
    """Compute the orientation matrix for a set of vectors.

    Compute the orientation matrix for a set of vectors, as described in
    Fisher, Lewis and Embleton. [#fisher-lewis-embleton]_ This ``d * d``
    matrix contains the sum of the pairwise products of the vector
    components.

    Parameters
    ----------
    vectors
        Array of shape ``(n, d)`` where ``n`` is the number of vectors and
        ``d`` is the number of dimensions.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(d, d)`` corresponding to the orientation matrix.

    """

    # Compute the orientation matrix using the inner product
    transposed_vectors = vectors.T
    orientation_matrix = np.inner(transposed_vectors, transposed_vectors)

    return orientation_matrix


def compute_orientation_matrix_eigs(
    vector_field: np.ndarray,
) -> NamedTuple:
    """Compute the eigenvectors and eigenvalues of the orientation matrix.

    Compute the eigen-decomposition of the orientation matrix. This
    function computes the matrix and then performs eigenvector calculation.

    Parameters
    ----------
    vector_field
        The vector field to consider, represented as either an array of
        shape ``(n, d)`` or an ``n+1``-dimensional array containing the
        components at their spatial locations, with the components present
        along the *last* axis.

    Returns
    -------
    eigenvectors : numpy.ndarray
        Eigenvectors of the orientation matrix.
    eigenvalues : numpy.ndarray
         Eigenvalues of the orientation matrix.

    Notes
    -----
    Equivalent to calling :func:`compute_orientation_matrix` and then using
    NumPy to compute the eigenvectors and eigenvalues.
    """

    orientation_matrix = compute_orientation_matrix(vector_field)

    return np.linalg.eig(orientation_matrix)


class OrientationMatrixParameters(NamedTuple):
    """Orientation matrix parameters.

    These parameters were first described by Woodcock. [#woodcock-1977]_
    """

    shape_parameter: float
    """Shape parameter, also known as gamma."""

    strength_parameter: float
    """Strength parameter, also known as zeta."""


def compute_orientation_matrix_parameters(eigs: np.ndarray) -> OrientationMatrixParameters:
    """Compute Woodcock's orientation matrix parameters.

    Compute the shape and strength parameters based on the orientation
    matrix, using the process first described by Woodcock [#woodcock-1977]_
    and using the notation presented by Fisher, Lewis and
    Embleton. [#fisher-lewis-embleton]_

    Parameters
    ----------
    eigs
        The eigenvalues of the orientation matrix.

    Returns
    -------
    OrientationMatrixParameters
        The distribution parameters computed from the orientation matrix
        eigenvalues.

    Notes
    -----
    See section 3.4 of Fisher, Lewis and Embleton [#fisher-lewis-embleton]_
    for computational and notational details. For the original description,
    see Woodcock. [#woodcock-1977]
    """

    # Sort the eigenvalues
    sorted_eigs = np.sort(eigs)

    # Get the eigenvalues
    t1, t2, t3 = sorted_eigs

    # Compute the shape parameter
    gamma = np.log(t3 / t2) / np.log(t2 / t1)

    # Compute the strength parameter
    zeta = np.log(t3 / t1)

    # Return the result
    return OrientationMatrixParameters(gamma, zeta)


def uniform_vs_unimodal_test(
    vector_field: np.ndarray,
    significance_level: float = 0.05,
) -> HypothesisResult:
    """Uniformity vs. unimodality test.

    Apply a test to determine if a distribution is uniform or unimodal, as
    described in section 5.3.1(i) of Fisher, Lewish and
    Embleton. [#fisher-lewis-embleton]_

    Parameters
    ----------
    vector_field
        The vector field to consider, represented as either an array of
        shape ``(n, d)`` or an ``n+1``-dimensional array containing the
        components at their spatial locations, with the components present
        along the *last* axis.
    significance_level
        Type I error value for the statistical test, default 0.05.

    Returns
    -------
    HypothesisResult
        Results of the hypothesis testing, indicating whether the null
        hypothesis of uniformity can be rejected, as well as the computed
        p-value.

    Notes
    -----
    In this function, the null hypothesis considers the orientations to be
    uniformly distributed on the surface of a sphere. The alternative
    hypothesis states that the data are not uniform, and are instead
    unimodal.

    This implementation assumes the large sample size scenario. The
    resultant length ``R`` is computed, and then the test
    statistic ``3R^2 / n`` is calculated and compared with a chi-squared
    variable with 3 degrees of freedom. If the test statistic is greater
    than the chi-squared value, then we can reject the null hypothesis in
    favour of the alternative hypothesis.

    References
    ----------
    See [#fisher-lewis-embleton]_, section 5.3.1(i).
    """

    # Check if the vectors
    vector_field = util.flatten_vector_field(vector_field)

    # Compute the resultant vector
    resultant_vector = compute_resultant_vector(
        vector_field=vector_field, compute_mean_resultant=False
    )

    # Find the squared magnitude of the resultant vector
    squared_resultant = np.sum(resultant_vector * resultant_vector)

    # Compute the test statistic
    test_statistic = 3 * squared_resultant / len(vector_field)

    # Find the chi-squared value to compare against
    # significant_value = chi2.ppf(1 - significance_level, df=3)

    # Determine if we can reject the null hypothesis
    # reject_null = test_statistic > significant_value

    # Compute the probability of type I error, i.e. the probability that
    # the distribution is indeed uniform but that we reject it.
    p_value = chi2.sf(test_statistic, df=3)

    # We reject the null if the probability of type I error is less than
    # the significance cutoff.
    can_reject_null = p_value < significance_level

    # Construct the results
    test_result = HypothesisResult(
        can_reject_null_hypothesis=can_reject_null,
        p_value=p_value,
        test_significance=significance_level,
    )

    # Return the result
    return test_result


def _compute_sum_of_arc_lengths(new_vector: np.ndarray, vectors: np.ndarray) -> float:
    """Compute the sum of arc lengths from vectors to a specified vector.

    See section 5.3.1(ii) in [#fisher-lewis-embleton]_. This function is
    used to estimate the spherical median of a sample of vectors.

    Parameters
    ----------
    new_vector
        The vector under consideration.
    vectors
        Cartesian components of a set of vectors.

    Returns
    -------
    float
        The sum of arc lengths from all vectors to the specified vector.
    """

    arc_lengths = util.compute_arc_lengths(new_vector, vectors)

    sum_of_arc_lengths = np.sum(arc_lengths)

    return sum_of_arc_lengths


def compute_median_direction(vector_field: np.ndarray) -> np.ndarray:
    """Compute the median direction for a unimodal distribution.

    Using the method described by Fisher, Lewis and
    Embleton [#fisher-lewis-embleton]_, compute the median direction for a
    unimodal directional distribution.

    Parameters
    ----------
    vector_field
        The vector field to consider, represented as either an array of
        shape ``(n, d)`` or an ``n+1``-dimensional array containing the
        components at their spatial locations, with the components present
        along the *last* axis.

    Returns
    -------
    numpy.ndarray
        Cartesian coordinates for the estimate of the median direction.

    Warnings
    --------
    The vectors must be normalised to unit length before computing these
    statistics.
    """

    vector_field = util.flatten_vector_field(vector_field)

    # Define the function to minimise
    sum_of_arc_lengths_function = functools.partial(
        _compute_sum_of_arc_lengths, vectors=vector_field
    )

    # Compute the mean as an initial guess
    mean_vector = compute_resultant_vector(vector_field, compute_mean_resultant=True)

    mean_vector /= np.linalg.norm(mean_vector)

    # Perform the minimisation
    res = minimize(
        sum_of_arc_lengths_function,
        mean_vector,
        bounds=[(-1, 1), (-1, 1), (-1, 1)],
        constraints=[NonlinearConstraint(fun=np.linalg.norm, lb=1, ub=1)],
    )

    # Get the values of x, y, z
    median_vector_cartesian = np.array(res.x)

    return median_vector_cartesian


def compute_elliptical_confidence_cone_points(
    w_matrix: np.ndarray,
    constant: float,
    h_matrix: np.ndarray = np.eye(3),
    number_of_points: int = 36,
) -> np.ndarray:
    """Compute ellipse points on the surface of a sphere.

    Following the procedure described in section 3.2.5 of Fisher, Lewis and
    Embleton, [#fisher-lewis-embleton]_ compute the coordinates of an
    ellipse on the surface of a unit sphere.

    Parameters
    ----------
    w_matrix
        Matrix whose eigenvectors and eigenvalues define the ellipse.
    constant
        Value on the right hand side of the ellipse equation.
    h_matrix
        Frame matrix used to compute the ellipse, by default the identity
        matrix.
    number_of_points
        Number of points to compute.

    Returns
    -------
    numpy.ndarray
        Array containing `number_of_points + 1` rows of 3D cartesian points
        which lie on the computed ellipse. The extra point is added to
        ensure that the ellipse is complete.

    """

    # First, take the eigenvectors and eigenvalues of the w matrix
    eigen_decomposition = np.linalg.eig(w_matrix)
    eigenvectors = eigen_decomposition.eigenvectors
    eigenvalues = eigen_decomposition.eigenvalues

    # Sort the eigenvalues
    eigenvalue_order = np.flip(np.argsort(eigenvalues))

    # Calculate the g_values
    g_1 = np.sqrt(constant / eigenvalues[eigenvalue_order[0]])
    g_2 = np.sqrt(constant / eigenvalues[eigenvalue_order[1]])

    # Get the constants from the eigenvectors, using the larger value.
    e_1 = eigenvectors[:, eigenvalue_order[0]]
    a = e_1[0]
    b = e_1[1]

    # Define the semivertical angles
    # b_1 = np.arcsin(g_1)
    # b_2 = np.arcsin(g_2)

    # Now, define the angles
    angles = np.linspace(
        start=0, stop=2 * np.pi, num=number_of_points + 1, endpoint=True
    )

    # Now, compute the ellipse points
    v_1 = g_1 * np.cos(angles)
    v_2 = g_2 * np.sin(angles)

    xs = a * v_1 - b * v_2
    ys = b * v_1 + a * v_2

    zs = np.sqrt(1 - xs * xs - ys * ys)

    coordinates = np.stack([xs, ys, zs], axis=-1)

    ellipse_points = (h_matrix @ coordinates.T).T

    return ellipse_points


def compute_confidence_cone_for_median(
    vector_field: np.ndarray,
    median_direction: Optional[np.ndarray] = None,
    significance_level: float = 0.05,
    number_of_points: int = 36,
) -> np.ndarray:
    """Compute elliptical confidence cone for the median orientation.

    Compute the matrix defining the elliptical confidence cone, as
    described by Fisher, Lewis and Embleton. [#fisher-lewis-embleton]_

    Parameters
    ----------
    vector_field
        The vector field to consider, represented as either an array of
        shape ``(n, d)`` or an ``n+1``-dimensional array containing the
        components at their spatial locations, with the components present
        along the *last* axis.
    median_direction
        The median direction vector in a NumPy array. If `None`, then the
        median direction is computed in this function.
    significance_level
        The acceptable type I error used to define the confidence cone.
        Based on repeated sampling, the true population median should fall
        within the computed ellipse `(1 - significance_level) * 100`
        percent of the time.
    number_of_points
        Number of ellipse points to compute for the confidence cone.

    Returns
    -------
    numpy.ndarray
        Points on the elliptical confidence cone.

    Warnings
    --------
    Requires a large sample size.
    """

    # Flatten the vector field, if need be.
    vector_field = util.flatten_vector_field(vector_field)

    # Compute the median vector, if necessary
    if median_direction is None:
        median_direction = compute_median_direction(vector_field)

    # Rotate to have the pole at the median direction
    rotated_vectors = util.rotate_vectors(
        vectors=vector_field, new_pole=median_direction
    )

    # Convert the rotated vectors to spherical coordinates
    vectors_spherical_coordinates = util.compute_vector_orientation_angles(rotated_vectors)

    # Convert the angles to match the definition in the book
    vectors_spherical_coordinates = util.convert_to_math_spherical_coordinates(
        vectors_spherical_coordinates
    )

    phi = vectors_spherical_coordinates[:, util.AngularIndex.PHI]
    theta = vectors_spherical_coordinates[:, util.AngularIndex.THETA]

    # Compute the C matrix entries
    n = len(vectors_spherical_coordinates)

    c_11 = np.sum(1 / np.tan(theta) * (1 - np.cos(2 * phi)) / 2) / n
    c_22 = np.sum(1 / np.tan(theta) * (1 + np.cos(2 * phi)) / 2) / n
    c_12 = -np.sum(1 / np.tan(theta) * np.sin(2 * phi) / 2) / n

    c_mat = np.array([[c_11, c_12], [c_12, c_22]])

    # Compute the sigma matrix entries s_ij
    s_11 = 1 + np.sum(np.cos(2 * phi)) / n
    s_22 = 1 - np.sum(np.cos(2 * phi)) / n
    s_12 = np.sum(np.sin(2 * phi)) / n

    s_mat = np.array([[s_11, s_12], [s_12, s_22]]) / 2

    # Compute the W matrix
    w_mat = c_mat @ np.linalg.inv(s_mat) @ c_mat

    # Find the constant value for the ellipse using the confidence level
    constant = -2 * np.log(significance_level) / n

    # And now for the ellipse points - use I since rotated data... I think,
    # although maybe we need to pass in the rotation matrix...
    ellipse_points = compute_elliptical_confidence_cone_points(
        w_matrix=w_mat, constant=constant, number_of_points=number_of_points
    )

    # TODO: Rotate the ellipse points back to the median

    return ellipse_points


def _kappa_equation(k: float, mean_resultant_length: float) -> float:
    """Equation which is satisfied by the concentration parameter.

    See Fisher, Lewis and Embleton, [#fisher-lewis-embleton]_ section
    5.3.2(iv).

    Parameters
    ----------
    k
        The concentration parameter of the Fisher-von Mises distribution.
    mean_resultant_length
        Mean resultant length of the sampled vectors.

    Returns
    -------
    float
        Value of the equation. Should be zero.
    """
    return 1 / np.tanh(k) - 1 / k - mean_resultant_length


def compute_mean_unit_direction(
    vector_field: np.ndarray,
    mean_resultant_vector: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compute the mean direction as a unit vector.

    Parameters
    ----------
    vector_field
        The vector field to consider, represented as either an array of
        shape ``(n, d)`` or an ``n+1``-dimensional array containing the
        components at their spatial locations, with the components present
        along the *last* axis.
    mean_resultant_vector
        Optional mean resultant vector. If provided, this vector is
        normalised. Otherwise, this vector is computed.

    Returns
    -------
    numpy.ndarray
        Unit vector containing the cartesian coordinates of the mean
        direction.

    Warnings
    --------
    Per Fisher, Lewis and Embleton, [#fisher-lewis-embleton]_ the sample
    mean corresponds to the maximum likelihood estimate. This may not hold
    for other distributions.
    """

    # As usual, flatten the vector field
    vector_field = util.flatten_vector_field(vector_field)

    # Compute the mean resultant vector
    if mean_resultant_vector is None:
        mean_resultant_vector = compute_resultant_vector(
            vector_field=vector_field, compute_mean_resultant=True
        )

    # Normalise the mean resultant
    mean_unit_vector = mean_resultant_vector / np.linalg.norm(mean_resultant_vector)

    # Return the unit mean vector
    return mean_unit_vector


def estimate_concentration_parameter(
    vector_field: np.ndarray,
    mean_resultant_vector: Optional[np.ndarray] = None,
    initial_guess: float = 0.5,
) -> float:
    """Estimate the concentration parameter.

    Using the maximum likelihood estimator presented in section 5.3.2(iv)
    of Fisher, Lewis and Embleton, [#fisher-lewis-embleton]_ estimate the
    concentration parameter of the provided vector field, assuming that the
    orientations follow a Fisher-von Mises distribution.

    Parameters
    ----------
    vector_field
        The vector field to consider, represented as either an array of
        shape ``(n, d)`` or an ``n+1``-dimensional array containing the
        components at their spatial locations, with the components present
        along the *last* axis.
    mean_resultant_vector
        The mean resultant vector, in cartesian coordinates. If not
        provided, it will be computed in this function.
    initial_guess
        Initial guess for the concentration parameter.

    Returns
    -------
    float
        The maximum likelihood estimate of the concentration parameter.

    Warnings
    --------
    The orientations provided are assumed to be distributed following a
    Fisher-von Mises distribution. The result is meaningless if the data
    are obtained from a different underlying distribution.

    This estimator is biased. See Fisher, Lewis and Embleton for
    alternative unbiased estimators.

    Notes
    -----
    As described by Fisher, Lewis and Embleton, [#fisher-lewis-embleton]_
    the maximum likelihood estimate of the Fisher-von Mises concentration
    parameter :math:`\\kappa` is obtained by solving:

    .. math::

        \\coth(\\kappa) - 1/\\kappa = R / n

    where :math:`\\coth` is the hyperbolic cotangent, :math:`R` is the
    resultant length and :math:`n` is the number of vectors.
    """

    # Flatten the vector field
    vector_field = util.flatten_vector_field(vector_field)

    # Compute the mean resultant length
    if mean_resultant_vector is None:
        mean_resultant_vector = compute_resultant_vector(
            vector_field, compute_mean_resultant=True
        )
    mean_resultant_length = np.linalg.norm(mean_resultant_vector)

    # Solve for the concentration parameter
    estimator = functools.partial(
        _kappa_equation, mean_resultant_length=mean_resultant_length
    )

    kappa_estimate = fsolve(estimator, np.array(initial_guess))

    return kappa_estimate[0]


def compute_confidence_cone_radius(
    vector_field: np.ndarray,
    kappa_estimate: Optional[float] = None,
    confidence_level: float = 0.01,
    use_degrees: bool = False,
) -> float:
    """Compute confidence cone radius for mean direction estimate.

    Determine the confidence cone radius around the estimated mean
    direction for a specified significance level for a Fisher-von Mises
    distribution. See the description in section 5.3.2 (iv) of Fisher,
    Lewis and Embleton. [#fisher-lewis-embleton]_

    Parameters
    ----------
    vector_field
        The vector field to consider, represented as either an array of
        shape ``(n, d)`` or an ``n+1``-dimensional array containing the
        components at their spatial locations, with the components present
        along the *last* axis.
    kappa_estimate
        Optional estimate of kappa. If not provided, then an estimate is
        computed to determine which estimation approach to use.
    confidence_level
        Desired confidence level for the mean direction estimate.
    use_degrees
        Indicate whether the angular radius should be converted to degrees.

    Returns
    -------
    float
        Arc length along the sphere of the confidence cone for the
        specified significance level.

    Warnings
    --------
    This function is only valid on orientations obtained by processes with
    an underlying Fisher-von Mises distribution. The results cannot be
    interpreted for data generated by other processes.
    """

    # As usual, start by flattening the vector field, if necessary.
    vector_field = util.flatten_vector_field(vector_field)

    # Check if we need to estimate the concentration parameter
    kappa_estimate = kappa_estimate or estimate_concentration_parameter(vector_field)

    # Compute the resultant length
    resultant_vector = compute_resultant_vector(
        vector_field, compute_mean_resultant=False
    )
    resultant_length = np.linalg.norm(resultant_vector)

    # Define some shorter variable names
    n = len(vector_field)
    r = resultant_length
    a = confidence_level
    k = kappa_estimate

    # Find the angular arc length of the confidence cone.
    # Behaviour differs if the value of kappa_estimate >= 5
    if k >= 5:
        theta_alpha = np.arccos(1 - ((n - r) / r) * ((1 / a) ** (1 / (n - 1)) - 1))
    else:
        theta_alpha = np.arccos(1 + np.log(a) / (k * r))

    # Return the arc length of the confidence cone radius
    if use_degrees:
        theta_alpha = np.degrees(theta_alpha)

    return theta_alpha


# TODO: Add function to compute the vertex points for the mean confidence cone


@dataclasses.dataclass
class FisherVonMisesParameters:
    """Parameters for a Fisher-von Mises distribution."""

    mu: np.ndarray
    """Mean direction in cartesian coordinates, of shape ``(3, )``."""

    kappa: float
    """Concentration parameter."""


def fit_fisher_vonmises_distribution(
    vector_field: np.ndarray,
) -> FisherVonMisesParameters:
    """Fit a Fisher-von Mises spherical distribution to a vector field.

    Parameters
    ----------
    vector_field
        The vector field to consider, represented as either an array of
        shape ``(n, d)`` or an ``n+1``-dimensional array containing the
        components at their spatial locations, with the components present
        along the *last* axis.

    Returns
    -------
    FisherVonMisesParameters
        Parameters necessary to construct a Fisher-von Mises distribution
        that fits the vector field.

    Warnings
    --------
    The vectors must be normalised to unit length before computing these
    statistics.

    See Also
    --------
    scipy.stats.vonmises_fisher.fit : Function used to perform the fitting.
    """

    # Flatten the vector field
    vector_field = util.flatten_vector_field(vector_field)

    # Perform the fitting
    mu, kappa = vonmises_fisher.fit(vector_field)

    # Build the parameters
    parameters = FisherVonMisesParameters(mu=mu, kappa=kappa)

    # And return
    return parameters


def compute_magnitude_orientation_correlation(
    vectors: np.ndarray, significance_level: float = 0.05
) -> Tuple[float, HypothesisResult]:
    """Compute the correlation between the magnitude and orientation.

    Following the procedure outlined in section 8.2.4 in Fisher, Lewis and
    Embleton, [#fisher-lewis-embleton]_ compute the correlation between the
    magnitude and orientation of a set of **non-unit** vectors.

    Parameters
    ----------
    vectors
        Array of shape ``(n, 3)`` containing the vectors to analyse. These
        should **not** be unit vectors.
    significance_level
        The test significance to compare the computed p-value against.

    Returns
    -------
    correlation_coefficient : float
        Biased estimate of the correlation coefficient.
    hypothesis_result : HypothesisResult
        Result of the hypothesis test to determine if the magnitude and
        orientation are correlated.

    Warnings
    --------
    This implementation assumes that a **large sample** is used (i.e.,
    ``n`` > 25).

    Notes
    -----
    The correlation coefficient is computed using the deviations from the
    mean of each variable. The jackknife approach has not yet been
    implemented.

    In this statistical test, the **null hypothesis** is that the magnitude
    and orientation are **not** correlated. If the test statistics is below
    the chi-squared value at the desired significance level, we reject this
    null hypothesis.

    The current implementation modifies the description in Fisher, Lewis
    and Embleton [#fisher-lewis-embleton]_ by performing array operations.
    """

    # Normalise vectors and compute magnitudes to separate the variables
    unit_vectors, magnitudes = util.normalise_vectors(vectors)

    # Make the magnitudes also a column
    magnitudes = np.expand_dims(magnitudes, -1)

    # The unit_vectors occupy the place of X, and the magnitudes that of Y
    x_bar = np.mean(unit_vectors, axis=0)
    y_bar = np.mean(magnitudes)

    # Perform the subtractions
    x_sub = unit_vectors - x_bar
    y_sub = magnitudes - y_bar

    # And now for the multiplications
    s_11 = x_sub.T @ x_sub
    s_12 = x_sub.T @ y_sub
    s_22 = y_sub.T @ y_sub

    # Now, build the matrix from which the correlations are computed
    s_matrix = np.linalg.inv(s_11) @ s_12 @ np.linalg.inv(s_22) @ s_12.T

    # Now, we must sum the diagonals.
    n, p = magnitudes.shape
    q = np.min([p, 3])

    rho_hat: float = s_matrix.diagonal().sum() / q

    # Now, for the hypothesis testing
    df = 3 * p
    test_statistic = q * n * rho_hat
    p_value = chi2.sf(test_statistic, df=df)

    can_reject_null = p_value < significance_level

    # Combine everything into the results
    test_result = HypothesisResult(
        can_reject_null,
        p_value,
        significance_level
    )

    # And return everything
    return rho_hat, test_result
