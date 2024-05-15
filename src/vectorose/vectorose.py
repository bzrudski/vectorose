# Copyright (c) 2023-, Benjamin Rudski, Joseph Deering
#
# This code is licensed under the MIT License. See the `LICENSE` file for
# more details about copying.

"""Angle and histogram calculations

This module provides the calculations for the :math:`\\phi` and
:math:`\\theta` angles from a vector field, as well as the functions for
binning the orientations.

"""

import enum
from typing import List, Tuple

import numpy as np

from . import util


class MagnitudeType(enum.IntEnum):
    """
    Type of magnitude.

    Type of magnitude considered when constructing the histograms.

    Members
    -------
    THREE_DIMENSIONAL
        Euclidean magnitude in 3D space. Index 0 in all arrays.

    IN_PLANE
        Magnitude of the :math:`x,y`-projection of the vector. Index 1 in
        all arrays.

    COUNT
        Simple count-based approach, where every vector has a weight of 1.
    """

    THREE_DIMENSIONAL = 0
    IN_PLANE = 1
    COUNT = 2


def compute_vector_magnitudes(vectors: np.ndarray) -> np.ndarray:
    """
    Compute vector magnitudes.

    Compute vector magnitudes in 3D, as well as the component of the
    magnitude in the :math:`(x,y)`-plane. See **Notes** for the equations
    used. See :class:`MagnitudeType` for the ordering of the different
    magnitudes in the output array.

    Parameters
    ----------
    vectors
        2D NumPy array containing 3 columns, corresponding to the x, y and
        z **components** of the vectors, and ``n`` rows, one for each
        vector. **Note:** We only require the vector *components*, not the
        *coordinates* in space.

    Returns
    -------
    numpy.ndarray
        2D NumPy array containing three columns and ``n`` rows. The first
        column corresponds to the 3D vector magnitude. The second
        column corresponds to the ``(x,y)`` in-plane magnitude. The third
        column is 1 for all rows, corresponding to the count for each.

    Notes
    -----
    The vector magnitudes are computed using the following equations:

    .. math::

        \\| v \\| &= \\sqrt{{v_x}^2 + {v_y}^2 + {v_z} ^ 2}

        \\| v \\|_{xy} &= \\sqrt{{v_x}^2 + {v_y}^2}

    """

    n = len(vectors)
    x: np.ndarray = vectors[:, 0]
    y: np.ndarray = vectors[:, 1]
    z: np.ndarray = vectors[:, 2]

    three_dimensional_magnitude = np.sqrt(x**2 + y**2 + z**2)
    in_plane_magnitude = np.sqrt(x**2 + y**2)

    magnitudes_array = np.zeros((n, 3))
    magnitudes_array[:, MagnitudeType.IN_PLANE] = in_plane_magnitude
    magnitudes_array[:, MagnitudeType.THREE_DIMENSIONAL] = three_dimensional_magnitude
    magnitudes_array[:, MagnitudeType.COUNT] = 1

    return magnitudes_array


def create_binned_orientation(
    vector_orientations: np.ndarray,
    vector_magnitudes: np.ndarray,
    half_number_of_bins: int = 16,
    use_degrees: bool = True,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Bin the vector orientation data.

    Construct an array containing the histogram data obtained by binning
    the orientation data in two angular dimensions. The same number of bins
    are created in the :math:`\\phi` and :math:`\\theta` axes, which is
    twice the number of bins passed in the ``half_number_of_bins``
    parameter.

    Parameters
    ----------
    vector_orientations
        2D NumPy array containing ``n`` rows,
        one for each vector, and 3 columns, corresponding to the
        angles :math:`\\phi,\\theta`.

    vector_magnitudes
        2D NumPy array containing ``n`` rows, one
        for each vector, and 2 columns, corresponding to the 3D and
        in-plane magnitudes, respectively.

    half_number_of_bins
        The half-number of bins. This represents
        the number of bins that should be produced in the 180\u00b0
        (:math:`\\pi` rad) range for each set of angles.

    use_degrees
        Indicate whether the angles are provided in
        degrees. If ``True``, angles are interpreted as degrees,
        otherwise the angles are interpreted as radians.

    Returns
    -------
    binned_data: numpy.ndarray
        2D histogram of :math:`\\phi,\\theta`. This histogram is
        a three-sheet :class:`numpy.ndarray`, with dimensions
        ``(half_number_of_bins, half_number_of_bins, 3)``. Axis
        zero corresponds to :math:`\\phi` and axis one corresponds
        to :math:`\\theta`. The last axis is used for indexing the
        histogram by the magnitude type (see :class:`MagnitudeType`).

    bins: list[numpy.ndarray]
        The bounds of the histogram bins. This array is of shape
        ``(2, half_half_number_of_bins + 1)``, where the first row
        represents the bins for :math:`\\phi` and the second represents
        the bins for :math:`\\theta` (see :class:`AngularIndex`).

    Warnings
    --------
    The input angles must be in the range :math:`0 \\leq \\phi < \\pi`
    and :math:`0 \\leq \\theta < 2\\pi` in radians, or
    :math:`0 \\leq \\phi < 180` and :math:`0 \\leq \\theta < 360` in
    degrees. There **cannot** be any other orientations included, as
    these will be overwritten.

    Notes
    -----
    Histogram bins are computed using :func:`numpy.histogram_bin_edges`
    and each vector is assigned to a bin using :func:`numpy.digitize`.

    As mentioned above, the input vectors may only occupy half the range of
    angular values for :math:`\\phi` and :math:`\\theta`, respectively. As
    the current package deals with orientations without considering
    direction, a given "vector" can be represented as having two
    corresponding locations on the surface of a sphere. When constructing
    the histogram, we must perform a mirroring step to include each point
    both in the origin angular location, as well as the angular bin
    on the opposite side of the sphere. This corresponding location is
    obtained by subtracting the half-number of bins from the computed bin.
    """

    # Get the number of vectors
    number_of_vectors = len(vector_orientations)

    # Extract the angles
    phi = vector_orientations[:, util.AngularIndex.PHI]
    theta = vector_orientations[:, util.AngularIndex.THETA]

    number_of_bins = 2 * half_number_of_bins

    if use_degrees:
        minimum_angle = 0
        maximum_phi_angle = 180
        maximum_theta_angle = 360
    else:
        minimum_angle = 0
        maximum_phi_angle = np.pi
        maximum_theta_angle = 2 * np.pi

    phi_histogram_bins = np.histogram_bin_edges(
        phi, bins=half_number_of_bins, range=(minimum_angle, maximum_phi_angle)
    )
    theta_histogram_bins = np.histogram_bin_edges(
        theta, bins=number_of_bins, range=(minimum_angle, maximum_theta_angle)
    )

    # Digitize returns indices which are off-by-one
    phi_bin_indices = np.digitize(phi, phi_histogram_bins) - 1
    theta_bin_indices = np.digitize(theta, theta_histogram_bins) - 1

    # Now, to prepare the histogram array:
    angular_histogram_2d = np.zeros(
        (half_number_of_bins, number_of_bins, len(MagnitudeType))
    )

    # Now, for the iterations. We set all the magnitudes simultaneously.
    for i in range(number_of_vectors):
        phi_bin = phi_bin_indices[i]
        theta_bin = theta_bin_indices[i]
        angular_histogram_2d[phi_bin, theta_bin] += vector_magnitudes[i]

    # Create an array that contains both the phi and theta
    # histogram boundaries.
    bin_boundaries = [phi_histogram_bins, theta_histogram_bins]

    return angular_histogram_2d, bin_boundaries


def create_angular_binning_from_vectors(
    vectors: np.ndarray,
    half_number_of_bins: int = 18,
    use_degrees: bool = True,
    consider_axial_data: bool = True,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Run the complete binning procedure on a list of vectors.

    Construct a 2D angular histogram from a 2D array of vectors with either
    3 columns or 6 columns. If the vector field has 6 columns, the first
    three will be considered the location coordinates and the final three
    will be considered the vector components, in the order ``x,y,z``.

    Parameters
    ----------
    vectors
        ``n`` by 6 or ``n`` by 3 array of vectors whose
        orientations will be analysed. If the vector array contains 6
        columns, **the last three are assumed to be the components**.

    half_number_of_bins
        number of bins in :math:`\\phi,\\theta`
        in half the angular range. The number of bins will be the same
        in both angular directions.

    use_degrees
        Indicates whether the angles should be computed
        in degrees. If ``True``, all angles will be stored in degrees
        (default). Otherwise, all angles will be stored in radians.

    consider_axial_data
        Indicates whether the vectors should be considered as axial data.
        If `True`, vectors with a negative Z-component are inverted, and
        the data are copied to create a symmetric pair.

    Returns
    -------
    binned_data: numpy.ndarray
        2D histogram of :math:`\\phi,\\theta`. This histogram is
        a three-sheet :class:`numpy.ndarray`, with dimensions
        ``(half_number_of_bins, half_number_of_bins, 3)``. Axis
        zero corresponds to :math:`\\phi` and axis one corresponds
        to :math:`\\theta`. The last axis is used for indexing the
        histogram by the magnitude type (see :class:`MagnitudeType`).

    bins: list[numpy.ndarray]
        The bounds of the histogram bins. This array is of shape
        ``(2, half_half_number_of_bins + 1)``, where the first row
        represents the bins for :math:`\\phi` and the second represents
        the bins for :math:`\\theta` (see :class:`AngularIndex`).

    See Also
    --------
    remove_zero_vectors:
        First step in this function, removes vectors of zero magnitude.

    compute_vector_orientation_angles:
        Second step, computes the :math:`\\phi` and :math:`\\theta` angles
        for all vectors provided.

    compute_vector_magnitudes:
        Next step, computes the 3D and in-plane magnitudes for the vectors.

    create_binned_orientation:
        Final step, performs the binning based on the computed orientations
        and magnitudes to produce a 2D angular histogram.

    convert_vectors_to_axes:
        Convert vectors to orientation axial data.

    create_symmetric_vectors_from_axes:
        Convert axial data into symmetric vector data.

    """

    # First, check the size of the vector array. Only keep the
    # components. We can discard the coordinates.
    if vectors.shape[1] > 3:
        vectors = vectors[:, 3:6]

    # Remove the zero-magnitude vectors
    non_zero_vectors = util.remove_zero_vectors(vectors)

    # If axial data, perform the mirroring
    if consider_axial_data:
        non_zero_vectors = util.convert_vectors_to_axes(non_zero_vectors)
        non_zero_vectors = util.create_symmetric_vectors_from_axes(non_zero_vectors)

    # Compute the angles
    vector_angles = util.compute_vector_orientation_angles(
        vectors=non_zero_vectors, use_degrees=use_degrees
    )

    # If weighing by magnitude, compute the magnitudes.
    # if weight_by_magnitude:
    vector_magnitudes = compute_vector_magnitudes(non_zero_vectors)
    # else:
    #     vector_magnitudes = None

    # Bin the data into the 2D histogram
    binned_data, bins = create_binned_orientation(
        vector_orientations=vector_angles,
        vector_magnitudes=vector_magnitudes,
        half_number_of_bins=half_number_of_bins,
        use_degrees=use_degrees,
    )

    return binned_data, bins
