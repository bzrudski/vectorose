# Copyright (c) 2023-, Benjamin Rudski, Joseph Deering
#
# This code is licensed under the MIT License. See the `LICENSE` file for
# more details about copying.

"""Polar data histogram calculations.

Compute 1D histograms for phi and theta angles separately.

Warnings
--------
Currently, this polar analysis is purely based on orientation and cannot
be done in conjunction with studies of magnitude.
"""
from typing import Tuple

import numpy as np

from . import util
from .util import AngularIndex


def create_binned_orientations(
    vector_orientations: np.ndarray,
    half_number_of_bins: int = 16,
    use_degrees: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin the vector orientation data.

    Construct an array containing the histogram data obtained by binning
    the orientation data in two angular dimensions. The same number of bins
    are created in the phi and theta axes, which is twice the number of
    bins passed in the ``half_number_of_bins`` parameter.

    Parameters
    ----------
    vector_orientations
        Array of shape ``(n, 2)`` where the columns correspond to the phi
        and theta angles, respectively.
    half_number_of_bins
        The half-number of bins. This represents the number of bins that
        should be produced in the 180 degree (pi radian) range for each set
        of angles.
    use_degrees
        Indicate whether the angles are provided in degrees.

    Returns
    -------
    binned_data: numpy.ndarray
        Array of shape ``(half_number_of_bins, number_of_bins)``
        containing the counts in each bin in ``(phi, theta)``. This array
        should **not** be considered a histogram. The rows correspond with
        phi angular bins, while the columns correspond to theta angular
        bins.
    phi_bins: numpy.ndarray
        The bounds of the phi histogram bins in an array of length
        ``half_number_of_bins + 1``.
    theta_bins: numpy.ndarray
        The bounds of the theta histogram bins in an array of length
        ``2 * half_number_of_bins + 1``.

    Warnings
    --------
    The input angles must be in the range ``0 <= phi < pi`` and ``0 <=
    theta < 2*pi`` in radians, or ``0 <= phi < 180`` and ``0 <= theta <
    360`` in degrees.
    """

    # Get the number of vectors
    # number_of_vectors = len(vector_orientations)

    # Extract the angles
    phi = vector_orientations[:, util.AngularIndex.PHI]
    theta = vector_orientations[:, util.AngularIndex.THETA]

    number_of_bins = 2 * half_number_of_bins

    minimum_angle = 0

    if use_degrees:
        maximum_phi_angle = 180
        maximum_theta_angle = 360
    else:
        maximum_phi_angle = np.pi
        maximum_theta_angle = 2 * np.pi

    phi_histogram_bins = np.histogram_bin_edges(
        phi, bins=half_number_of_bins, range=(minimum_angle, maximum_phi_angle)
    )
    theta_histogram_bins = np.histogram_bin_edges(
        theta, bins=number_of_bins, range=(minimum_angle, maximum_theta_angle)
    )

    angular_histogram_2d, theta_histogram_bins, phi_histogram_bins = np.histogram2d(
        phi, theta, (phi_histogram_bins, theta_histogram_bins), density=False
    )

    # Create a tuple that contains the phi and theta histogram boundaries.
    return angular_histogram_2d, phi_histogram_bins, theta_histogram_bins


def create_angular_binning_from_vectors(
    vectors: np.ndarray,
    half_number_of_bins: int = 18,
    use_degrees: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the complete binning procedure on a list of vectors.

    Construct the data for angular histograms from a 2D array of vectors.

    Parameters
    ----------
    vectors
        Array of shape ``(n, 6)`` or ``(n, 3)`` containing ``n`` vectors
        whose orientations will be analysed. If the vector array contains 6
        columns, **the last three are assumed to be the vector
        components**.
    half_number_of_bins
        Number of bins to construct for the phi angle of inclination. The
        number of theta bins will be twice this value, as theta covers
        twice the angular range of phi.
    use_degrees
        Indicate whether the angles should be computed in degrees.

    Returns
    -------
    binned_data: numpy.ndarray
        Array of shape ``(half_number_of_bins, number_of_bins)``
        containing the counts in each bin in ``(phi, theta)``. This array
        should **not** be considered a histogram. The rows correspond with
        phi angular bins, while the columns correspond to theta angular
        bins.
    phi_bins: numpy.ndarray
        The bounds of the phi histogram bins in an array of length
        ``half_number_of_bins + 1``.
    theta_bins: numpy.ndarray
        The bounds of the theta histogram bins in an array of length
        ``(2 * half_number_of_bins + 1)``.

    See Also
    --------
    compute_vector_orientation_angles:
        Second step, computes the ``phi`` and ``theta`` angles for all
        vectors provided.
    create_binned_orientation:
        Final step, performs the binning based on the computed orientations
        and magnitudes to produce polar histograms.

    """

    # First, check the size of the vector array. Only keep the
    # components. We can discard the coordinates.
    if vectors.shape[1] > 3:
        vectors = vectors[:, 3:6]

    # Compute the angles
    vector_angles = util.compute_vector_orientation_angles(
        vectors=vectors, use_degrees=use_degrees
    )

    # Bin the data into the 2D histogram
    binned_data, phi_bins, theta_bins = create_binned_orientations(
        vector_orientations=vector_angles,
        half_number_of_bins=half_number_of_bins,
        use_degrees=use_degrees,
    )

    return binned_data, phi_bins, theta_bins


def produce_phi_theta_1d_histogram_data(
    binned_data: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the separate polar phi and theta histogram arrays.

    Compute the 1D histograms for the angles phi and theta based on the
    provided binning.

    Parameters
    ----------
    binned_data
        NumPy array containing the binned :math:`\\phi,\\theta` histogram.
        This array should have shape ``(n, n, 3)`` where ``n`` is the
        number of histogram bins. See :func:`.create_binned_orientation`
        for a detailed explanation of the format.
    Returns
    -------
    phi_values : numpy.ndarray
        NumPy array of shape ``(n,)`` containing the marginal phi
        histogram, going from 0 to 180 degrees (or 0 to pi radians).
    theta_values : numpy.ndarray
        NumPy array of shape ``(2n,)`` containing the marginal theta
        histogram, going from 0 to 360 degrees (or 0 to 2 * pi radians).

    See Also
    --------
    .create_binned_orientation:
        Create the 2D binning data to pass to this function.

    Notes
    -----
    The phi histogram is obtained by summing the binned values along the
    theta axis, while the theta histogram is obtained by summing the binned
    values along the phi axis.
    """

    # Sum along an axis to compute the single-angle distributions
    phi_histogram = np.sum(
        binned_data, axis=AngularIndex.THETA
    )
    theta_histogram = np.sum(
        binned_data, axis=AngularIndex.PHI
    )

    # return one_dimensional_histograms
    return phi_histogram, theta_histogram


def prepare_two_dimensional_histogram(binned_data: np.ndarray) -> np.ndarray:
    """Prepare data for plotting on a UV sphere.

    Parameters
    ----------
    binned_data
        Binned 2D data of shape ``(n, n)``. Note that only a
        single sheet is passed to this function. The input may be
        magnitude-weighted or count weighted.

    Returns
    -------
    numpy.ndarray
        Adjusted histogram data of shape ``(n, 2n)``.

    See Also
    --------
    .create_binned_orientation:
        Create the 2D histogram to pass to this function.

    Notes
    -----
    This function takes 2D face data of shape ``(n, n)`` and
    returns the data to plot on a sphere, having shape
    ``(n, 2n)``, where ``n`` corresponds to the half-number of bins. In
    the final array that is returned, the ``n`` rows correspond to the
    ``n`` histogram bins in ``phi`` going from ``0`` to ``pi / 2``
    while the ``2n`` columns represent the bins in ``theta``, going from
    ``-pi`` to ``pi`` radians. To prepare for wrapping around the sphere,
    the first ``n`` columns are row-inverted with respect to the last ``n``
    columns. This is done to correspond to the "negative" values of ``phi``
    (which do not actually exist, but are used as a convenience to plot the
    full sphere).
    """

    # This first array corresponds to the mirrored angles on the
    # back of the sphere
    sphere_intensity_data_first_half: np.ndarray = binned_data

    # This second array corresponds to the values on the
    # front half of the sphere
    sphere_intensity_data_second_half: np.ndarray = binned_data.copy()

    # Combine the two arrays together
    sphere_intensity_data_first_half: np.ndarray = np.flip(
        sphere_intensity_data_first_half, axis=0
    )
    sphere_intensity_data = np.concatenate(
        [sphere_intensity_data_first_half, sphere_intensity_data_second_half], axis=-1
    )

    return sphere_intensity_data
