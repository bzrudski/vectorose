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
from typing import Any, List, Sequence, Tuple, Union

import numpy as np


class AngularIndex(enum.IntEnum):
    """Angular Index

    Stores the index of the different angles to avoid ambiguity in code.

    Members
    -------
    PHI
        Angle phi (:math:`\\phi`), representing the angle with respect to
        the positive :math:`y`-axis. Index 0 in all arrays.

    THETA
        Angle theta (:math:`\\theta`), representing the angle of incline
        with respect to the positive :math:`z`-axis. Index 1 in all arrays.
    """

    PHI = 0
    THETA = 1


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


def remove_zero_vectors(vectors: np.ndarray) -> np.ndarray:
    """Prune zero-vectors.

    Remove vectors of zero magnitude from the list of vectors.

    Parameters
    ----------
    vectors
        ``n`` by 6 or ``n`` by 3 array of vectors. If the array has 6
        columns, *the last 3 are assumed to be the vector components*.

    Return
    ------
    numpy.ndarray:
        List of vectors with the same number of columns as the
        original without any vectors of zero magnitude.
    """

    # Determine which columns contain the vector components
    _, number_of_columns = vectors.shape

    if number_of_columns == 6:
        vector_columns = np.arange(3, 6)
    else:
        vector_columns = np.arange(3)

    # Only take the vectors that do not have zero in all components.
    non_zero_vectors = vectors[~np.all(vectors[:, vector_columns] == 0, axis=1)]

    return non_zero_vectors


def normalise_vectors(vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Normalise an array of vectors.

    Rescale a series of vectors to ensure that all have unit length. All
    zero-vectors should be removed before using this function.

    Parameters
    ----------
    vectors
        ``n`` by 6 or ``n`` by 3 array of vectors. If the array has 6
        columns, *the last 3 are assumed to be the vector components*.
        This array must contain **no zero-vectors**.

    Returns
    -------
    normalised_vectors : numpy.ndarray
        Array of the same shape as `vectors`, but with all vector
        components rescaled to ensure that the vectors have unit length.
    magnitudes : numpy.ndarray
        Array of shape ``(n,)`` containing the magnitud of each vector.

    Notes
    -----
    This function does not modify the original array. A new array is
    created and returned.

    The 3D magnitude is used to perform the normalisation. This magnitude
    is computed as

    .. math::

        \\|\\vec{v}\\| = \\sqrt{v_x^2 + v_y^2 + v_z^2}

    where :math:`v_i` refers to the component of :math:`\\vec{v}` along
    the *i*-th axis.
    """

    # Compute the vector magnitudes
    vector_components = vectors[:, -3:]
    vector_magnitudes = np.sqrt(np.sum(vector_components * vector_components, axis=-1))

    # Divide by the magnitudes
    normalised_components = vector_components / vector_magnitudes[:, None]

    # Create a new array with the modified components if necessary
    if normalised_components.shape != vectors.shape:
        normalised_vectors = normalised_components.copy()
        normalised_vectors[:, -3:] = normalised_components
    else:
        normalised_vectors = normalised_components

    return normalised_vectors, vector_magnitudes


def convert_vectors_to_axes(vectors: np.ndarray) -> np.ndarray:
    """Convert vectors to axes.

    Reflect all vectors so that they are oriented in the four octants that
    have positive z-values. These correspond to the axes conventionally
    used in directional statistics (see the book by Fisher, Lewis and
    Embleton [#fisher-lewis-embleton]_).

    Parameters
    ----------
    vectors
        NumPy array of shape ``(n, 3)`` or ``(n, 6)`` containing the
        vectors.

    Returns
    -------
    numpy.ndarray
        NumPy array of the same shape as the original, but with all vectors
        oriented towards a non-negative Z value.

    References
    ----------
    .. [#fisher-lewis-embleton] Fisher, N. I., Lewis, T., & Embleton, B. J.
       J. (1993). Statistical analysis of spherical data ([New ed.], 1.
       paperback ed). Cambridge Univ. Press.
    """

    # Get the vector components
    vector_components = vectors[:, -3:]

    # Invert the vectors with z component below zero
    indices_to_flip = vector_components[:, -1] < 0
    vector_components[indices_to_flip] = -vector_components[indices_to_flip]

    # Assign the axes to the positions, if necessary.
    axes = vectors.copy()
    axes[:, -3:] = vector_components

    return axes


def create_symmetric_vectors_from_axes(axes: np.ndarray) -> np.ndarray:
    """Create a set of symmetric vectors from axes.

    Duplicate a collection of axes to produce vectors pointing in both
    directions corresponding to each orientation.

    Parameters
    ----------
    axes
        NumPy array of shape ``(n, 3)`` containing the axes. All entries in
        this array should have a positive Z-value.

    Returns
    -------
    numpy.ndarray
        NumPy array of shape ``(2n, 3)`` containing the vectors along each
        direction. The inverted vectors appear in the same order as the
        axes **after the non-inverted vectors**.

    Warnings
    --------
    The inverted vectors, having negative z-values, are appended after the
    non-inverted vectors. Corresponding vectors are **not** interleaved.
    """

    upward_vectors = axes.copy()
    downward_vectors = -upward_vectors

    vectors = np.concatenate([upward_vectors, downward_vectors], axis=0)

    return vectors


def convert_spherical_to_cartesian_coordinates(
    angular_coordinates: np.ndarray, radius: Union[float, np.ndarray] = 1
) -> np.ndarray:
    """Convert spherical coordinates to cartesian coordinates.

    Convert spherical coordinates provided in terms of phi and theta
    into cartesian coordinates. For the conversion to be possible, a
    sphere radius must also be specified. If none is provided, the
    sphere is assumed to be the unit sphere. The angles must be provided
    in **radians**.

    Parameters
    ----------
    angular_coordinates
        Array with >=2 columns representing :math:`\\phi` and
        :math:`\\theta`, respectively (see :class:`AngularIndex`), and
        ``n`` rows representing the data points. This function can also be
        used on the output of :func:`np.mgrid`, if the arrays have been
        stacked such that the final axis is used to distinguish between phi
        and theta.

    radius
        A :class:`float` or :class:`numpy.ndarray` representing the radius
        of the sphere. If the value passed is an array, it must have ``n``
        rows, one for each data point. Default: ``radius=1``.

    Return
    ------
    numpy.ndarray:
        Array with 3 columns, corresponding to the cartesian
        coordinates in X, Y, Z, and ``n`` rows, one for each data point.
        If mgrids are provided, then multiple sheets will be returned
        in this array, with the -1 axis still used to distinguish between
        x, y, z.

    Notes
    -----

    The equations governing the conversion are:

    .. math::

        x &= r \\sin(\\theta)\\sin(\\phi)

        y &= r \\cos(\\theta)\\sin(\\phi)

        z &= r \\cos(\\phi)

    The input is provided as a 2D array with 2 columns representing the
    angles phi and theta, and ``n`` rows, representing the datapoints.
    The returned array is also a 2D array, with three columns (X, Y, Z)
    and ``n`` rows.
    """

    # Simple definition of a sphere used here.
    phi: np.ndarray = angular_coordinates[..., AngularIndex.PHI]
    theta: np.ndarray = angular_coordinates[..., AngularIndex.THETA]

    x = radius * np.sin(theta) * np.sin(phi)
    y = radius * np.cos(theta) * np.sin(phi)
    z = radius * np.cos(phi)

    # Combine the coordinates together
    cartesian_coordinates = np.stack([x, y, z], axis=-1)

    return cartesian_coordinates


def compute_vector_orientation_angles(
    vectors: np.ndarray, use_degrees: bool = False
) -> np.ndarray:
    """Compute the vector orientation angles phi and theta.

    For all vectors passed in ``vectors``, compute the :math:`\\phi` and
    :math:`\\theta` orientation angles. The :math:`\\phi` angle corresponds
    to the tilt with respect to the ``z`` axis, while the :math:`\\theta`
    angle is the angle in the ``xy``-plane with respect to the ``y`` axis.
    See **Notes** for more details on the definition and calculations
    of these angles.

    The unit for the angles is *radians* unless ``use_degrees`` is set
    to ``True``. The returned angles are in the range of 0
    to :math:`\\pi` (180\u00b0) for :math:`\\phi` and 0 to :math:`2\\pi`
    (180\u00b0) for :math:`\\theta`. The first column in the returned array
    corresponds to :math:`\\phi` and the second to :math:`\\theta`. See
    :class:`AngularIndex` for more details about the ordering of the
    angles.

    Parameters
    ----------
    vectors
        2D NumPy array containing 3 columns, corresponding to the x, y and
        z **components** of the vectors, and ``n`` rows, one for each
        vector. **Note:** We only require the vector *components*, not the
        *coordinates* in space.

    use_degrees
        indicate whether the returned angles should be in degrees.
        If ``False`` (default), the angles will be returned in *radians*.

    Returns
    -------
    numpy.ndarray
        2D NumPy array containing 2 columns, corresponding to
        :math:`\\phi,\\theta` for ``n`` rows.

    Notes
    -----
    In this package, we define the angles to be:

    * :math:`\\phi` - The angle of tilt with respect to the positive
      :math:`z`-axis. A vector with :math:`\\phi=0` will be oriented
      parallel to the :math:`z`-axis, while a vector with
      :math:`\\phi=\\pi/2` will be oriented parallel to the
      :math:`(x,y)`-plane. A vector with :math:`\\phi=\\pi` will be
      oriented parallel to the negative :math:`z`-axis.

    * :math:`\\theta` - The orientation in the :math:`(x,y)`-plane with
      respect to the *positive* :math:`y`-axis. A vector with
      :math:`\\theta=0` will be parallel to the *positive*
      :math:`y`-axis, while a vector with :math:`\\theta=\\pi/2` will be
      oriented parallel to the *positive* :math:`x`-axis.

    These angles are computed in the following manner:

    .. math::

        \\phi_i &= \\textup{arctan} \\left( \\frac{\\sqrt{{x_i} ^ 2 +
        {y_i} ^ 2}}{z_i} \\right)

        \\theta_i &= \\textup{arctan} \\left( \\frac{x_i}{y_i} \\right)

    It is important to note that these angles have specific ranges to avoid
    the possibility of describing the same angle in two different ways. The
    :math:``\\phi`` angle must be in the range :math:`0 \\leq \\phi < \\pi`
    in radians, or :math:`0 \\leq \\phi < 180` in degrees. The value of
    :math:`\\theta` is defined in :math:`-\\pi \\leq \\theta < \\pi` in
    radians, or :math:`-180 \\leq \\theta < 180` in degrees. This function
    restricts the range of :math:`\\theta` further due to the nature of
    the orientations considered in the current application.
    """

    n = len(vectors)

    # Ensure that all vectors are in octants with positive x.
    # vectors = np.copy(vectors)
    # vectors[vectors[:, 0] < 0] = -vectors[vectors[:, 0] < 0]

    x: np.ndarray = vectors[:, 0]
    y: np.ndarray = vectors[:, 1]
    z: np.ndarray = vectors[:, 2]

    # Compute the raw angles using arctan2
    phi = np.arctan2(np.sqrt(x**2 + y**2), z)
    theta = np.arctan2(x, y)

    # Now, we need to fix the angles
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)
    # phi[phi == np.pi] = 0

    theta = np.where(theta < 0, theta + 2 * np.pi, theta)
    # theta[theta == np.pi] = 0

    # Convert to degrees if necessary
    if use_degrees:
        phi = np.degrees(phi)
        theta = np.degrees(theta)

    angular_coordinates = np.zeros((n, 2))
    angular_coordinates[:, AngularIndex.PHI] = phi
    angular_coordinates[:, AngularIndex.THETA] = theta

    return angular_coordinates


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
    phi = vector_orientations[:, AngularIndex.PHI]
    theta = vector_orientations[:, AngularIndex.THETA]

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
    non_zero_vectors = remove_zero_vectors(vectors)

    # If axial data, perform the mirroring
    if consider_axial_data:
        non_zero_vectors = convert_vectors_to_axes(non_zero_vectors)
        non_zero_vectors = create_symmetric_vectors_from_axes(non_zero_vectors)

    # Compute the angles
    vector_angles = compute_vector_orientation_angles(
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


def perform_binary_search(
    seq: Union[Sequence, np.ndarray], item: Any, lower_bound: int = 0
) -> int:
    """Perform a binary search.

    Find the index of a specified item, or of the greatest item less than
    the desired item.

    Parameters
    ----------
    seq
        Sequence to search.
    item
        Item to locate.
    lower_bound
        Start index of the sequence.

    Returns
    -------
    int
        Index of the requested item, or of the greatest item less than the
        requested one.

    """
    # Check the list length - return the current index if only one or no values.
    if len(seq) == 1:
        return lower_bound

    # Get the middle index
    middle_index = np.floor(len(seq) / 2).astype(int)
    middle_value = seq[middle_index]

    # print(f"Considering the item {middle_value} at index {middle_index}")

    if item == middle_value:
        return lower_bound + middle_index
    elif item < middle_value:
        # Recurse left
        return perform_binary_search(seq[:middle_index], item, lower_bound=lower_bound)
    elif item > middle_value:
        # Recurse right
        return perform_binary_search(
            seq[middle_index:], item, lower_bound=lower_bound + middle_index
        )
