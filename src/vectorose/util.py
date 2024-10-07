# Copyright (c) 2024-, Benjamin Rudski, Joseph Deering
#
# This code is licensed under the MIT License. See the `LICENSE` file for
# more details about copying.

"""Utility functions

This module provides utility functions for manipulating vectors in
Cartesian and spherical coordinates.
"""

import enum
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.spatial.transform import Rotation


class AngularIndex(enum.IntEnum):
    """Angular index definition.

    Stores the index of the different angles to avoid ambiguity in code.
    """

    PHI = 0
    """Angle phi, in-plane with respect to positive ``y``; index 0."""

    THETA = 1
    """Angle theta, incline with respect to positive ``z``; index 1."""


def flatten_vector_field(vector_field: np.ndarray) -> np.ndarray:
    """Flatten a vector field into a 2D vector list.

    Convert an n-dimensional vector image volume into a 2D list of vectors,
    with rows reflecting vectors and the columns reflecting each component.

    Parameters
    ----------
    vector_field
        Array containing the vector field. If this array is 2D, then the
        rows are considered to correspond to the vectors, while the columns
        correspond to the components. If the vector has higher dimension,
        the last axis is assumed to distinguish between the components.

    Returns
    -------
    numpy.ndarray
        2D array containing the vectors as rows and the components as
        columns. If the original array was 2D, this original array is
        returned without copying.
    """

    if vector_field.ndim > 2:
        d = vector_field.shape[-1]
        vector_field = vector_field.reshape(-1, d)

    return vector_field


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


def normalise_array(arr: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """Normalise an array.

    Normalise the provided array so that all entries sum to one along the
    specified axis.

    Parameters
    ----------
    arr
        The array to normalise. This array can have any shape.
    axis
        The axis along which to normalise. If `None`, then overall
        normalisation is performed.

    Returns
    -------
    numpy.ndarray
        The normalised array, such that the sum of all entries is 1 along
        the specified axis.

    """

    if axis is None:
        axis = tuple(np.arange(arr.ndim))

    sums_along_axis = arr.sum(axis=axis)
    sums_along_axis = np.expand_dims(sums_along_axis, axis)

    normalised_array = arr / sums_along_axis

    return normalised_array


def normalise_vectors(vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Normalise an array of vectors.

    Rescale a series of vectors to ensure that all non-zero vectors have
    unit length.

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

    original_dimensions = vectors.ndim

    vectors = np.atleast_2d(vectors)

    # Compute the vector magnitudes
    vector_components = vectors[:, -3:]
    vector_magnitudes = np.linalg.norm(vector_components, axis=-1)

    # Divide by the magnitudes
    stacked_magnitudes = vector_magnitudes[:, None]
    non_zero_rows_stacked = ~np.all(vector_components == 0, axis=-1)[:, None]

    normalised_components = np.true_divide(
        vector_components, stacked_magnitudes, where=non_zero_rows_stacked
    )

    # Create a new array with the modified components if necessary
    if normalised_components.shape != vectors.shape:
        normalised_vectors = normalised_components.copy()
        normalised_vectors[:, -3:] = normalised_components
    else:
        normalised_vectors = normalised_components

    if original_dimensions < 2:
        normalised_vectors = np.squeeze(normalised_vectors)

    return normalised_vectors, vector_magnitudes


def generate_representative_unit_vectors(
    vectors: np.ndarray,
    number_of_samples: Optional[int] = None,
) -> np.ndarray:
    """Generate a representative sample of unit vectors.

    Using the magnitudes of a set of non-zero, non-unit vectors as weight,
    create a sample of unit vectors whose frequency is proportional to the
    magnitudes.

    Parameters
    ----------
    vectors
        Array of shape ``(n, 3)`` containing non-normalised vectors in
        Cartesian coordinates.
    number_of_samples
        Number of vectors to draw randomly. If `None`, then the minimum of
        the ceiling of the number of vectors divided by the minimum of the
        normalised magnitudes, or simple ``10e7`` is used.

    Returns
    -------
    numpy.ndarray
        The randomly sampled vectors in an array of shape ``(m, 3)`` where
        ``m`` is either the value of `number_of_samples` or the
        automatically computed number described above if
        `number_of_samples` is `None`.

    Notes
    -----
    This function is included to allow computing directional statistics on
    the inputted vector fields. Most described approaches rely on
    collections of *unit* vectors. Simply normalising the vectors may alter
    the meaning of the presented data. The rationale behind this function
    is to sample unit vectors of direction with probability proportional to
    the respective magnitudes. This process produces a collection of unit
    vectors whose distribution of orientations matches the weights imposed
    by the original magnitudes.

    """

    unit_vectors, magnitudes = normalise_vectors(vectors)

    normalised_magnitudes = normalise_array(magnitudes, axis=0)

    if number_of_samples is None:
        number_of_vectors = len(normalised_magnitudes)
        number_of_samples = np.ceil(
            number_of_vectors / normalised_magnitudes.min()
        )

        number_of_samples = np.min([number_of_samples, 1e7]).astype(int)

    selected_vectors = np.random.default_rng().choice(
        unit_vectors,
        size=number_of_samples,
        replace=True,
        p=normalised_magnitudes,
        axis=0,
    )

    return selected_vectors


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
    angular_coordinates: np.ndarray,
    radius: Union[float, np.ndarray] = 1,
    use_degrees: bool = False
) -> np.ndarray:
    """Convert spherical coordinates to cartesian coordinates.

    Convert spherical coordinates provided in terms of phi and theta
    into cartesian coordinates. For the conversion to be possible, a
    sphere radius must also be specified. If none is provided, the
    sphere is assumed to be the unit sphere.

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
    use_degrees
        Indicate whether the provided angular coordinates are in degrees.
        If `False` (default), radians are assumed.

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

    # Convert to radians if necessary
    if use_degrees:
        angular_coordinates = np.radians(angular_coordinates)

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

    if vectors.ndim > 1:
        n = len(vectors)
    else:
        n = 1

    # Ensure that all vectors are in octants with positive x.
    # vectors = np.copy(vectors)
    # vectors[vectors[:, 0] < 0] = -vectors[vectors[:, 0] < 0]

    x: np.ndarray = vectors[..., 0]
    y: np.ndarray = vectors[..., 1]
    z: np.ndarray = vectors[..., 2]

    # Compute the raw angles using arctan2
    phi = np.arctan2(np.sqrt(x**2 + y**2), z)
    theta = np.arctan2(x, y)

    # Now, we need to fix the angles
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)
    # phi[phi == np.pi] = 0

    theta = np.where(theta < 0, theta + 2 * np.pi, theta)
    theta = np.where(theta >= 2 * np.pi, theta - 2 * np.pi, theta)
    # theta[theta == np.pi] = 0

    # Convert to degrees if necessary
    if use_degrees:
        phi = np.degrees(phi)
        theta = np.degrees(theta)

    angular_coordinates = np.zeros((n, 2))
    angular_coordinates[..., AngularIndex.PHI] = phi
    angular_coordinates[..., AngularIndex.THETA] = theta

    # If there is only one vector, squeeze out the extra axis
    angular_coordinates = np.squeeze(angular_coordinates)

    return angular_coordinates


def compute_spherical_coordinates(
    vectors: np.ndarray, use_degrees: bool=False
) -> np.ndarray:
    """Compute spherical coordinates for a set of vectors.

    Compute true spherical coordinates for a set of provided vectors. These
    coordinates express a vector as an orientation, consisting of the
    angles phi and theta, and a magnitude.

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
        Array of shape ``(n, 3)`` containing the vectors in spherical
        coordinates, consisting of ``phi``, ``theta`` and ``magnitude``
        columns.

    See Also
    --------
    compute_compute_vector_orientation_angles :
        Compute phi and theta angles from Cartesian coordinates.
    numpy.linalg.norm :
        Compute the magnitude (norm) of vectors in Cartesian coordinates.
    """

    # Compute the orientation angles
    orientations = compute_vector_orientation_angles(vectors, use_degrees)

    # Compute the magnitudes
    magnitudes = np.linalg.norm(vectors, axis=-1)[:, None]

    # Combine everything
    spherical_coordinates = np.hstack([orientations, magnitudes])

    # And return it all
    return spherical_coordinates


def convert_to_math_spherical_coordinates(
    original_angles: np.ndarray, use_degrees: bool = False
) -> np.ndarray:
    """Convert to the mathematical definition of spherical coordinates.

    Directional statistics texts, such as the work by Fisher, Lewis and
    Embleton, [#fisher-lewis-embleton]_ define the spherical coordinates
    differently than we do in this code. For compatibility with statistical
    procedures described in such works, this function converts spherical
    coordinates in our representation to the standard definition.

    Parameters
    ----------
    original_angles
        Array of shape ``(n, 2)`` containing the phi, theta angles computed
        using our definition of spherical coordinates, defined in the
        function :func:`.compute_vector_orientation_angles`.
    use_degrees
        Indicate whether the original spherical coordinates are in degrees,
        and whether the resulting transformed vectors should also be in
        degrees. If `False`, all angles are assumed to be in radians.

    Returns
    -------
    numpy.ndarray
        Array of the same shape as the input `original_angles`, but with
        the angles defined following Fisher, Lewis and Embleton's
        definitions. [#fisher-lewis-embleton]_


    Notes
    -----
    The polar coordinates in section 2.2 (a) of by Fisher, Lewis and
    Embleton [#fisher-lewis-embleton]_ define the angle :math:`\\theta` as
    the angle of inclination from the vertical axis, while the in-plane
    angle :math:`\\phi` is the counter-clockwise (anticlockwise) angle in
    the ``xy``-plane, measured with respect to the ``+y`` axis.

    """

    # Convert to radians if necessary
    if use_degrees:
        original_angles = np.radians(original_angles)

    # Extract the angular components
    phi = original_angles[..., AngularIndex.PHI]
    theta = original_angles[..., AngularIndex.THETA]

    # Take into account the different definitions of angles in FL&E
    new_theta = phi
    new_phi = (- theta + np.pi / 2) % (2 * np.pi)
    # new_phi = np.where(new_phi > 0, new_phi, new_phi + 2 * np.pi)
    # new_phi = np.where(new_phi >= 2 * np.pi, new_phi % (2 * np.pi), new_phi)

    # And now define the new array
    new_angles = np.zeros_like(original_angles)
    new_angles[..., AngularIndex.PHI] = new_phi
    new_angles[..., AngularIndex.THETA] = new_theta

    # Convert to degrees, if necessary
    if use_degrees:
        new_angles = np.degrees(new_angles)

    return new_angles

def convert_math_spherical_coordinates_to_vr_coordinates(
    original_angles: np.ndarray, use_degrees: bool = False
) -> np.ndarray:
    """Convert mathematical spherical coordinates to vectorose conventions.

    Directional statistics texts, such as the work by Fisher, Lewis and
    Embleton, [#fisher-lewis-embleton]_ define the spherical coordinates
    differently than we do in this code. For compatibility with statistical
    procedures described in such works, this function converts spherical
    coordinates in the standard definition to our representation of
    spherical coordinates.

    Parameters
    ----------
    original_angles
        Array of shape ``(n, 2)`` containing the phi, theta angles computed
        using the standard mathematical spherical coordinates, described by
        Fisher, Lewis and Embleton. [#fisher-lewis-embleton]_
    use_degrees
        Indicate whether the original spherical coordinates are in degrees,
        and whether the resulting transformed vectors should also be in
        degrees. If `False`, all angles are assumed to be in radians.

    Returns
    -------
    numpy.ndarray
        Array of the same shape as the input `original_angles`, but with
        the angles defined as in the function
        :func:`.compute_vector_orientation_angles`.


    Notes
    -----
    The polar coordinates in section 2.2 (a) of by Fisher, Lewis and
    Embleton [#fisher-lewis-embleton]_ define the angle :math:`\\theta` as
    the angle of inclination from the vertical axis, while the in-plane
    angle :math:`\\phi` is the counter-clockwise (anticlockwise) angle in
    the ``xy``-plane, measured with respect to the ``+y`` axis.

    """

    # Convert to radians if necessary
    if use_degrees:
        original_angles = np.radians(original_angles)

    # Extract the angular components
    phi = original_angles[..., AngularIndex.PHI]
    theta = original_angles[..., AngularIndex.THETA]

    # Take into account the different definitions of angles in FL&E
    new_phi = theta
    new_theta = (- phi + np.pi / 2) % (2 * np.pi)

    # And now define the new array
    new_angles = np.zeros_like(original_angles)
    new_angles[..., AngularIndex.PHI] = new_phi
    new_angles[..., AngularIndex.THETA] = new_theta

    # Convert to degrees, if necessary
    if use_degrees:
        new_angles = np.degrees(new_angles)

    return new_angles


def rotate_vectors(
    vectors: np.ndarray, new_pole: np.ndarray
) -> np.ndarray:
    """Rotate a set of vectors.

    Rotate vectors so that the top pole of the sphere is rotated to a
    specified location.

    Parameters
    ----------
    vectors
        Array containing the Cartesian vector components to rotate, of
        shape ``(n, 3)``, where ``n`` represents the number of 3D vectors.
    new_pole
        Vector coordinates corresponding to the new pole position after
        rotating, also in cartesian coordinates.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n, 3)`` containing the rotated vector components.

    Notes
    -----
    Although the approach described by Fisher, Lewis and
    Embleton [#fisher-lewis-embleton]_ was initially used, we replaced it
    with the :class:`scipy.spatial.transform.Rotation` class present in
    SciPy.
    """

    # Convert the new pole location into phi and theta angles
    new_pole_spherical_coordinates = compute_vector_orientation_angles(
        vectors=new_pole, use_degrees=False
    )

    rotation = Rotation.from_euler("xz", -new_pole_spherical_coordinates)

    rotated_vectors = rotation.apply(vectors)

    # Return the rotated components
    return rotated_vectors


# Non-vector operations
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
