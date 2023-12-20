"""
Anisotropy Rose - Angle Calculations

Joseph Deering, Benjamin Rudski
2023

This module provides the calculations for the :math:`\phi` and
:math:`\theta` angles from a vector field, as well as the functions for
binning the orientations.

"""

import enum
from typing import Optional, Tuple

import numpy as np


class AngularIndex(enum.IntEnum):
    """
    Angular Index

    Stores the index of the different angles to avoid ambiguity in
    code.

    Attributes:
        * PHI: Angle phi (:math:`\\phi`), representing the angle with
            respect to the positive :math:`y`-axis. Index 0 in all
            arrays.
        * THETA: Angle theta (:math:`\\theta`), representing the angle
            of incline with respect to the positive :math:`z`-axis.
            Index 1 in all arrays.
    """

    PHI = 0
    THETA = 1


class MagnitudeType(enum.IntEnum):
    """
    Type of magnitude.

    Type of magnitude considered when constructing the histograms.

    Attributes:
        * THREE_DIMENSIONAL: Euclidean magnitude in 3D space. Index 0 in
            all arrays.
        * IN_PLANE: Magnitude of the :math:`x,y`-projection of the
            vector. Index 1 in all arrays.
    """

    THREE_DIMENSIONAL = 0
    IN_PLANE = 1
    COUNT = 2


def remove_zero_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Prune zero-vectors.

    Remove vectors of zero magnitude from the list of vectors.

    :param vectors: ``n`` by 6 or ``n`` by 3 array of vectors. If the
        array has 6 columns, **the last 3 are assumed to be the vector
        components**.
    :return: list of vectors with the same number of columns as the
        original, without any vectors of zero magnitude.
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


def convert_spherical_to_cartesian_coordinates(
    angular_coordinates: np.ndarray, radius: float | np.ndarray = 1
) -> np.ndarray:
    """
    Convert spherical coordinates to cartesian coordinates.

    Convert spherical coordinates provided in terms of phi and theta
    into cartesian coordinates. For the conversion to be possible, a
    sphere radius must also be specified. If none is provided, the
    sphere is assumed to be the unit sphere. The angles must be provided
     in **radians**.

    The equations governing the conversion are:

    .. math::

        x = r \\sin(\\theta)\\sin(\\phi)

        y = r \\cos(\\theta)\\sin(\\phi)

        z = r \\cos(\\phi)

    The input is provided as a 2D array with 2 columns representing the
    angles phi and theta, and ``n`` rows, representing the datapoints.
    The returned array is also a 2D array, with three columns (X, Y, Z)
    and ``n`` rows.

    :param angular_coordinates: Array with >=2 columns representing phi
        and theta, respectively, and ``n`` rows representing the
        datapoints. This function can also be used on the output of
        ``np.mgrid``, if the arrays have been stacked such that the
        final axis is used to distinguish between phi and theta.
    :param radius: A float or array representing the radius of the
        sphere (default: unit radius). If array, the array must have
        ``n`` rows.
    :return: Array with 3 columns, corresponding to the cartesian
        coordinates in X, Y, Z, and ``n`` rows, one for each data point.
        If mgrids are provided, then multiple sheets will be returned in
        this array, with the -1 axis still used to distinguish between
        x, y, z.
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
    """
    Compute the vector orientation angles phi and theta.

    For all vectors passed in ``vectors``, compute the :math:`\\phi` and
    :math:`\\theta` orientation angles. We define the angles to be:

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

        \\phi_i = \\textup{arctan} ( \\frac{\\sqrt{{x_i} ^ 2 +
        {y_i} ^ 2}}{z_i} )

        \\theta_i = \\textup{arctan} ( \\frac{x_i}{y_i} )

    The unit for the angles is **radians** unless ``use_degrees`` is set
    to ``True``. The angles are modified to be in the range of :math:`0`
    to :math:`\\pi` for :math:`\\phi` and 0 to :math:`\\pi` for
    :math:`\\theta`. The first column in the returned array corresponds
    to :math:`\\phi` and the second to :math:`\\theta`. See
    ``AngularIndex`` for more details about the ordering of the angles.

    :param vectors: 2D NumPy array containing 3 columns, corresponding
        to the x, y and z **components** of the vectors, and ``n`` rows,
        one for each vector. **Note:** We only require the vector
        *components*, not the *coordinates* in space.
    :param use_degrees: indicate whether the returned angles should be
        in degrees. If ``False`` (default), the angles will be returned
        in **radians**.
    :return: 2D NumPy array containing 2 columns, corresponding to
        :math:`\phi,\theta` for ``n`` rows.
    """

    n = len(vectors)

    # Ensure that all vectors are in octants with positive x.
    vectors = np.copy(vectors)
    vectors[vectors[:, 0] < 0] = -vectors[vectors[:, 0] < 0]

    x: np.ndarray = vectors[:, 0]
    y: np.ndarray = vectors[:, 1]
    z: np.ndarray = vectors[:, 2]

    # Compute the raw angles using arctan2
    phi = np.arctan2(np.sqrt(x**2 + y**2), z)
    theta = np.arctan2(x, y)

    # Now, we need to fix the angles so that we keep them in the appropriate ranges of zero to pi
    phi = np.where(phi < 0, phi + np.pi, phi)
    phi[phi == np.pi] = 0

    theta = np.where(theta < 0, theta + np.pi, theta)
    theta[theta == np.pi] = 0

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

    Compute vector magnitudes in 3D, as well as the component of the magnitude in the :math:`(x,y)`-plane. These
    magnitudes are calculated as:

    .. math::

        \| v \| = \\sqrt{{v_x}^2 + {v_y}^2 + {v_z} ^ 2}

        \| v \|_{xy} = \\sqrt{{v_x}^2 + {v_y}^2}

    :param vectors: 2D NumPy array containing 3 columns, corresponding to the x, y and z **components** of the
                    vectors, and ``n`` rows, one for each vector. **Note:** We only require the vector *components*,
                    not the *coordinates* in space.
    :return: 2D NumPy array containing two columns and ``n`` rows. The first column corresponds to the 3D vector
             magnitude while the second column corresponds to the :math:`(x,y)` in-plane magnitude.
    """

    n = len(vectors)
    x: np.ndarray = vectors[:, 0]
    y: np.ndarray = vectors[:, 1]
    z: np.ndarray = vectors[:, 2]

    three_dimensional_magnitude = np.sqrt(x**2 + y**2 + z**2)
    in_plane_magnitude = np.sqrt(x**2 + y**2)

    magnitudes_array = np.zeros((n, 2))
    magnitudes_array[:, MagnitudeType.IN_PLANE] = in_plane_magnitude
    magnitudes_array[:, MagnitudeType.THREE_DIMENSIONAL] = three_dimensional_magnitude

    return magnitudes_array


def create_binned_orientation(
    vector_orientations: np.ndarray,
    vector_magnitudes: np.ndarray,
    half_number_of_bins: int = 16,
    use_degrees: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bin the vector orientation data.

    Construct an array containing the histogram data obtained by binning
    the orientation data.

    The input angles must be in the range :math:`0 \\leq \\phi < \\pi`
    and :math:`0 \\leq \\theta < \\pi` in radians, or
    :math:`0 \\leq \\phi < 180` and :math:`0 \\leq \\theta < 180` in
    degrees. There **cannot** be any other orientations included, as
    these will be overwritten.

    Once the bins are assigned, a mirroring step is performed to fill in
    the missing angles. In each case, the corresponding mirrored bin is
    obtained by subtracting the half-number of bins.

    :param vector_orientations: 2D NumPy array containing ``n`` rows,
        one for each vector, and 2 columns, corresponding to the
        angles :math:`\phi,\theta`.
    :param vector_magnitudes: 2D NumPy array containing ``n`` rows, one
        for each vector, and 2 columns, corresponding to the 3D and
        in-plane magnitudes, respectively.
    :param half_number_of_bins: The half-number of bins. This represents
        the number of bins that should be produced in the 180\u00b0
        (:math:`\\pi` rad) range for each set of angles.
    :param use_degrees: Indicate whether the angles are provided in
        degrees. If ``True``, angles are interpreted as degrees,
        otherwise the angles are interpreted as radians.
    :return: Tuple containing 2D histograms of :math:`\phi,\theta` and
        an array providing bounds of the histogram bins. The histogram
        contains 3 sheets and has dimensions
        ``(half_number_of_bins * 2, half_number_of_bins * 2, 3)``.
         See ``MagnitudeType`` for the correct indexing for each sheet.
         Axis zero corresponds to :math:`\phi` and axis one corresponds
         to :math:`\theta`. The histogram bins array
         is of shape ``(2, 2 * half_half_number_of_bins + 1)``, where
         the first row/sheet represents the bins for :math:`\phi` and
         the second represents the bins for :math:`\theta`.
    """

    # Indicate whether we are weighting by magnitude
    # magnitude_weighted = vector_magnitudes is not None

    # Get the number of vectors
    number_of_vectors = len(vector_orientations)

    # Augment the vector counts with a `1` so that we can easily perform
    # the count-based approach.
    one_column = np.ones((number_of_vectors, 1))

    vector_magnitudes = np.concatenate([vector_magnitudes, one_column], axis=-1)

    # Extract the angles
    phi = vector_orientations[:, AngularIndex.PHI]
    theta = vector_orientations[:, AngularIndex.THETA]

    number_of_bins = 2 * half_number_of_bins

    if use_degrees:
        minimum_angle = -180
        maximum_angle = 180
    else:
        minimum_angle = -np.pi
        maximum_angle = np.pi

    phi_histogram_bins = np.histogram_bin_edges(
        phi, bins=number_of_bins, range=(minimum_angle, maximum_angle)
    )
    theta_histogram_bins = np.histogram_bin_edges(
        theta, bins=number_of_bins, range=(minimum_angle, maximum_angle)
    )

    # Digitize returns indices which are off-by-one
    phi_bin_indices = np.digitize(phi, phi_histogram_bins) - 1
    theta_bin_indices = np.digitize(theta, theta_histogram_bins) - 1

    # Now, to prepare the histogram array:
    angular_histogram_2d = np.zeros((number_of_bins, number_of_bins, 3))

    # Now, for the iterations. We can easily mirror through subtraction
    # so that we can actually modify both the original and the reflected
    # cells at the same time.
    for i in range(number_of_vectors):
        phi_bin = phi_bin_indices[i]
        mirrored_phi_bin = phi_bin - half_number_of_bins

        theta_bin = theta_bin_indices[i]
        mirrored_theta_bin = theta_bin - half_number_of_bins

        # Why divide by 2? So that we aren't double-counting.
        angular_histogram_2d[phi_bin, theta_bin] += vector_magnitudes[i] / 2
        angular_histogram_2d[mirrored_phi_bin, mirrored_theta_bin] += (
            vector_magnitudes[i] / 2
        )

    # Create an array that contains both the phi and theta
    # histogram boundaries.
    bin_boundaries = np.stack([phi_histogram_bins, theta_histogram_bins])

    return angular_histogram_2d, bin_boundaries


def create_angular_binning_from_vectors(
    vectors: np.ndarray,
    half_number_of_bins: int = 18,
    use_degrees: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the complete binning procedure on a list of vectors.

    This function takes a 2D array of vectors, with either 3 columns or
    6 columns, and creates a 2D angular histogram. If the vector field
    has 6 columns, the first three will be considered the location
    coordinates and the final three will be considered the vector
    components, in the order ``x,y,z``.

    :param vectors: ``n`` by 6 or ``n`` by 3 array of vectors whose
        orientations will be analysed. If the vector array contains 6
        columns, **the last three are assumed to be the components**.
    :param half_number_of_bins: number of bins in :math:`\phi,\theta`
        in half the angular range. The number of bins will be the same
        in both angular directions.
    :param use_degrees: Indicates whether the angles should be computed
        in degrees. If ``True``, all angles will be stored in degrees
        (default). Otherwise, all angles will be stored in radians.
    :return: Tuple containing 2D histogram of :math:`\phi,\theta` and
        an array providing bounds of the histogram bins. If
        ``weight_by_magnitude`` is ``True``, this will be a three-sheet
        histogram, with dimensions ``(half_number_of_bins * 2,
        half_number_of_bins * 2, 3)``. Axis zero corresponds to
        :math:`\phi` and axis one corresponds to :math:`\theta`. The
        histogram bins array is of shape
        ``(2, 2 * half_half_number_of_bins + 1)``, where the first row
        represents the bins for :math:`\phi` and the second represents
        the bins for :math:`\theta`.
    """

    # First, check the size of the vector array. Only keep the
    # components. We can discard the coordinates.
    if vectors.shape[1] > 3:
        vectors = vectors[:, 3:6]

    # Remove the zero-magnitude vectors
    non_zero_vectors = remove_zero_vectors(vectors)

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
