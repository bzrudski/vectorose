# Copyright (c) 2024-, Benjamin Rudski, Joseph Deering
#
# This code is licensed under the MIT License. See the `LICENSE` file for
# more details about copying.

"""Utility functions

This module provides utility functions for manipulating vectors in
Cartesian and spherical coordinates.
"""

import enum
from typing import Any, Sequence, Tuple, Union

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
