# Copyright (c) 2024-, Benjamin Rudski, Joseph Deering
#
# This code is licensed under the MIT License. See the `LICENSE` file for
# more details about copying.

"""Utility functions

This module provides utility functions for manipulating vectors in
Cartesian and spherical coordinates.

References
----------
.. [#fisher-lewis-embleton] Fisher, N. I., Lewis, T., & Embleton, B. J.
   J. (1993). Statistical analysis of spherical data ([New ed.], 1.
   paperback ed). Cambridge Univ. Press.
"""

import enum
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


class AngularIndex(enum.IntEnum):
    """Angular index definition.

    Stores the index of the different angles to avoid ambiguity in code.
    """

    PHI = 0
    """Angle phi, in-plane with respect to positive ``y``; index 0."""

    THETA = 1
    """Angle theta, incline with respect to positive ``z``; index 1."""


class AngleName(str, enum.Enum):
    """Angular index definition.

    Stores the name of the different angles to avoid ambiguity in code.
    """

    PHI = "phi"
    """Angle phi, in-plane with respect to positive ``y``; index 0."""

    THETA = "theta"
    """Angle theta, incline with respect to positive ``z``; index 1."""


class MagnitudeType(enum.IntEnum):
    """Type of vector magnitude."""

    THREE_DIMENSIONAL = 0
    """Euclidean magnitude in 3D space."""

    IN_PLANE = 1
    """Magnitude of the ``(x,y)``-projection of the vector."""


def convert_vectors_to_data_frame(vectors: np.ndarray) -> pd.DataFrame:
    """Convert vector array into a DataFrame.

    Convert an array of vectors into a pandas :class:`pandas.DataFrame`.

    Parameters
    ----------
    vectors
        Array of shape ``(n, 3)`` or ``(n, 6)`` containing the vectors. If
        three columns are present, they are considered as the ``x, y, z``
        vector components, respectively. If six columns are present, the
        final three columns are considered as the vector components, while
        the first three columns are considered the ``x, y, z`` spatial
        locations of the vectors.

    Returns
    -------
    pandas.DataFrame
        Data frame of the same shape as `vectors`. Spatial columns, if
        present, are labelled ``x, y, z`` while the vector component
        columns are labelled ``vx, vy, vz``.
    """

    number_of_columns = vectors.shape[-1]

    columns = ["vx", "vy", "vz"]

    if number_of_columns > 3:
        columns = ["x", "y", "z"] + columns

    vector_df = pd.DataFrame(vectors, columns=columns)

    return vector_df


def compute_vector_magnitudes(vectors: np.ndarray) -> np.ndarray:
    """Compute vector magnitudes.

    Compute vector magnitudes in 3D, as well as the component of the
    magnitude in the ``(x,y)``-plane.

    Parameters
    ----------
    vectors
        Array of shape ``(n, 3)`` containing the x, y and z *components* of
        the ``n`` 3-dimensional vectors.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n, 2)`` containing the vector magnitudes. The
        first column contains the true 3D vector magnitude. While the
        second column contains the magnitude of the projection onto the
        ``xy``-plane.

    Notes
    -----
    The vector magnitudes are computed using the following equations:

    .. math::

        \\| v \\| &= \\sqrt{{v_x}^2 + {v_y}^2 + {v_z} ^ 2}

        \\| v \\|_{xy} &= \\sqrt{{v_x}^2 + {v_y}^2}

    The both magnitudes are implemented in :func:`numpy.linalg.norm`.

    """

    if vectors.ndim > 1:
        n = len(vectors)
    else:
        n = 1

    three_dimensional_magnitude = np.linalg.norm(vectors, axis=-1)
    in_plane_magnitude = np.linalg.norm(vectors[..., :2], axis=-1)

    magnitudes_array = np.zeros((n, 2))
    magnitudes_array[:, MagnitudeType.IN_PLANE] = in_plane_magnitude
    magnitudes_array[:, MagnitudeType.THREE_DIMENSIONAL] = three_dimensional_magnitude

    # Squeeze out single dimensions, if only a single vector is passed in.
    if n == 1:
        magnitudes_array = np.squeeze(magnitudes_array, axis=0)

    return magnitudes_array


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
    vector_components = vectors[..., -3:]
    vector_magnitudes = np.linalg.norm(vector_components, axis=-1)

    # Divide by the magnitudes
    stacked_magnitudes = vector_magnitudes[:, None]
    non_zero_rows_stacked = ~np.all(vector_components == 0, axis=-1)[:, None]

    normalised_components = np.true_divide(
        vector_components, stacked_magnitudes, where=non_zero_rows_stacked
    )

    # Create a new array with the modified components if necessary
    if normalised_components.shape != vectors.shape:
        normalised_vectors = vectors.copy()
        normalised_vectors[..., -3:] = normalised_components
    else:
        normalised_vectors = normalised_components

    if original_dimensions < 2:
        normalised_vectors = np.squeeze(normalised_vectors)

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
        Array of shape ``(n, 3)`` or ``(n, 6)`` containing the
        vectors.

    Returns
    -------
    numpy.ndarray
        Array of the same shape as the original, but with all vectors
        oriented towards a non-negative Z value.
    """

    # Get the vector components
    axes = vectors.copy()
    axes_components = axes[:, -3:]

    # Invert the vectors with z component below zero
    indices_to_flip = axes_components[:, -1] < 0
    axes_components[indices_to_flip] = -axes_components[indices_to_flip]

    return axes


def create_symmetric_vectors_from_axes(axes: np.ndarray) -> np.ndarray:
    """Create a set of symmetric vectors from axes.

    Duplicate a collection of axes to produce vectors pointing in both
    directions corresponding to each orientation.

    Parameters
    ----------
    axes
        Array of shape ``(n, 3)`` containing the axes. All entries in
        this array should have a positive Z-value.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(2n, 3)`` containing the vectors along each
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

    For all provided vectors, compute the ``phi`` and ``theta`` angles. The
    ``phi`` angle corresponds to the co-latitude, representing the tilt
    with respect to the ``z``-axis, while ``theta`` is the azimuthal angle
    in the ``xy``-plane with respect to the positive ``y``-axis.

    Parameters
    ----------
    vectors
        Array of shape ``(n, 3)`` containing 3 columns, corresponding to
        the x, y and z *components* of ``n`` 3-dimensional vectors.
    use_degrees
        Indicate whether the returned angles should be in degrees.
        Otherwise, the angles are in **radians**.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n, 2)`` containing the ``phi`` and ``theta``
        angles for each vector.

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

    To ensure that each direction has a unique description, we restrict the
    angles to specific ranges. The ``phi`` angle is in the range
    ``0 <= phi <= 180`` degrees, or ``0 <= phi <= pi`` radians, while the
    ``theta`` angle is in the range ``0 <= theta < 360`` degrees or
    ``0 <= theta < 2 * pi`` radians.
    """

    if vectors.ndim > 1:
        n = len(vectors)
    else:
        n = 1

    x: np.ndarray = vectors[..., 0]
    y: np.ndarray = vectors[..., 1]
    z: np.ndarray = vectors[..., 2]

    # Compute the orientation angles using arctan2
    phi = np.arctan2(np.sqrt(x**2 + y**2), z)
    theta = np.arctan2(x, y)

    # Now, we need to fix the theta angles to get the correct range
    theta %= (2 * np.pi)

    # Convert to degrees if necessary
    if use_degrees:
        phi = np.degrees(phi)
        theta = np.degrees(theta)

    angular_coordinates = np.zeros((n, 2))
    angular_coordinates[..., AngularIndex.PHI] = phi
    angular_coordinates[..., AngularIndex.THETA] = theta

    # If there is only one vector, squeeze out the extra axis
    if n == 1:
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

    # Get the number of vectors
    if vectors.ndim > 1:
        n = len(vectors)
    else:
        n = 1

    # Compute the orientation angles
    orientations = compute_vector_orientation_angles(vectors, use_degrees)

    # Compute the magnitudes
    magnitudes = np.atleast_1d(np.linalg.norm(vectors, axis=-1))[:, None]

    if n == 1:
        magnitudes = np.squeeze(magnitudes, axis=0)

    # Combine everything
    spherical_coordinates = np.hstack([orientations, magnitudes])

    # If there is only one vector, squeeze out the extra dimension
    # if n == 1:
    #     spherical_coordinates = np.squeeze(spherical_coordinates, axis=0)

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
    the ``xy``-plane, measured with respect to the ``+x`` axis.

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

    See Also
    --------
    scipy.spatial.transform.Rotation : Abstraction used for the rotations.

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


def compute_arc_lengths(vector: np.ndarray, vector_collection: np.ndarray) -> np.ndarray:
    """Compute the arc lengths between a vector and many vectors.

    For each vector in a set of vectors, compute the arc length to a
    specified vector. The arc length is the angular distance on the surface
    of the unit sphere.

    Parameters
    ----------
    vector
        Array of shape ``(3,)`` containing the reference vector from which
        all arc lengths are measured.
    vector_collection
        Array of shape ``(n, 3)`` containing ``n`` three-dimensional
        vectors. The arc lengths of each vector will be measured with
        respect to `vector`.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n,)`` containing the arc length from the
        reference `vector` to each of the vectors in the provided
        collection.

    Warnings
    --------
    The angular distance reported can also be interpreted as the angle in
    **radians** between the reference vector and each respective vector.

    Notes
    -----
    The results can only be interpreted on the unit sphere. While vectors
    of any length may be passed in, the arc length interpretation relies
    on a unit sphere. Otherwise, the results can still be interpreted as
    angles between the vector tails, but **should not** be considered any
    sort of magnitude-linked arc length.

    The explanation for this method comes in part from Section 5.3.1(ii) in
    Fisher, Lewis and Embleton [#fisher-lewis-embleton]_. This method
    relies on the relationship between dot-products and angles, as

    .. math::

        \\mathbf{u} \\cdot \\mathbf{v} = \\|\\mathbf{u}\\| \\cdot
        \\|\\mathbf{v}\\| \\cos \\theta

    where :math:`\\theta` is the angle between the tails of vectors
    :math:`\\mathbf{u}` and :math:`\\mathbf{v}`.
    """

    number_of_vectors = len(vector_collection)

    # Compute the dot products between each vector and the new vector
    dot_products = np.dot(vector_collection, vector)

    # To mitigate any floating point errors, divide by norms
    vectors_magnitudes = np.linalg.norm(vector_collection, axis=-1)
    new_vector_magnitudes = np.repeat(np.linalg.norm(vector), number_of_vectors)
    angle_cosines = dot_products / (new_vector_magnitudes * vectors_magnitudes)

    # Now, get the arc lengths, knowing that we are on a unit sphere.
    arc_lengths = np.arccos(angle_cosines)
    return arc_lengths
