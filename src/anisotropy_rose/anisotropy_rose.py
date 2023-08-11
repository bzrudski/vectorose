"""
Anisotropy Rose

Joseph Deering, Benjamin Rudski
2023

This package provides the ability to construct 2D and 3D rose diagrams of anisotropy vector fields.

"""

import enum
from typing import Optional, Tuple

import mpl_toolkits.mplot3d.axes3d
import matplotlib.pyplot as plt
import numpy as np


class MagnitudeType(enum.IntEnum):
    """
    Type of magnitude.

    Type of magnitude considered when constructing the histograms.
    """
    THREE_DIMENSIONAL = 0
    IN_PLANE = 1


class CardinalDirection(str, enum.Enum):
    """
    Cardinal Directions

    This string-based enumerated type is useful when preparing 2D polar figures.

    See: https://matplotlib.org/stable/api/projections/polar.html#matplotlib.projections.polar.PolarAxes.set_theta_zero_location
    """
    NORTH = "N"
    NORTH_WEST = "NW"
    WEST = "W"
    SOUTH_WEST = "SW"
    SOUTH = "S"
    SOUTH_EAST = "SE"
    EAST = "E"
    NORTH_EAST = "NE"


class RotationDirection(enum.IntEnum):
    """
    Rotation Direction

    This int-based enumerated type defines two members:

    * Clockwise: -1
    * Counter-clockwise / anti-clockwise: 1

    See: https://matplotlib.org/stable/api/projections/polar.html#matplotlib.projections.polar.PolarAxes.set_theta_direction
    """
    CLOCKWISE = -1
    COUNTER_CLOCKWISE = 1


class AngularIndex(enum.IntEnum):
    """
    Angular Index

    Stores the index of the different angles to avoid ambiguity in code. Angle indices are:

    * Phi (:math:`\\phi`): 0
    * Theta (:math:`\\theta`): 1
    """
    PHI = 0
    THETA = 1


def remove_zero_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Prune zero-vectors.

    Remove vectors of zero magnitude from the list of vectors.

    :param vectors: ``n`` by 6 or ``n`` by 3 array of vectors. If the array has 6 columns, **the last 3 are assumed
                    to be the vector components**.
    :return: list of vectors with the same number of columns as the original, without any vectors of zero magnitude.
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


def convert_spherical_to_cartesian_coordinates(angular_coordinates: np.ndarray, radius: float = 1) -> np.ndarray:
    """
    Convert spherical coordinates to cartesian coordinates.

    Convert spherical coordinates provided in terms of phi and theta into cartesian coordinates. For the conversion
    to be possible, a sphere radius must also be specified. If none is provided, the sphere is assumed to be the
    unit sphere. The angles must be provided in **radians**.

    The equations governing the conversion are:

    .. math::

        x = r \\sin(\\theta)\\sin(\\phi)

        y = r \\cos(\\theta)\\sin(\\phi)

        z = r \\cos(\\phi)

    The input is provided as a 2D array with 2 columns representing the angles phi and theta, and ``n`` rows,
    representing the datapoints. The returned array is also a 2D array, with three columns (X, Y, Z) and ``n`` rows.

    :param angular_coordinates: Array with >=2 columns representing phi and theta, respectively, and ``n`` rows
                                representing the datapoints. This function can also be used on the output of
                                ``np.mgrid``, if the arrays have been stacked such that the final axis is used to
                                distinguish between phi and theta.
    :param radius: A single float representing the radius of the sphere (default: unit radius).
    :return: Array with 3 columns, corresponding to the cartesian coordinates in X, Y, Z, and ``n`` rows,
             one for each data point. If mgrids are provided, then multiple sheets will be returned in this array,
             with the -1 axis still used to distinguish between x, y, z.
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


def compute_vector_orientation_angles(vectors: np.ndarray, use_degrees: bool = False) -> np.ndarray:
    """
    Compute the vector orientation angles phi and theta.

    For all vectors passed in ``vectors``, compute the :math:`\\phi` and :math:`\\theta` orientation angles. We
    define the angles to be:

    * :math:`\\phi` - The angle of tilt with respect to the positive :math:`z`-axis. A vector with :math:`\\phi=0`
      will be oriented parallel to the :math:`z`-axis, while a vector with :math:`\\phi=\\pi/2` will be oriented
      parallel to the :math:`(x,y)`-plane. A vector with :math:`\\phi=\\pi` will be oriented parallel to the
      negative :math:`z`-axis.

    * :math:`\\theta` - The orientation in the :math:`(x,y)`-plane with respect to the *positive* :math:`y`-axis. A
      vector with :math:`\\theta=0` will be parallel to the *positive* :math:`y`-axis, while a vector with
      :math:`\\theta=\\pi/2` will be oriented parallel to the *positive* :math:`x`-axis.

    These angles are computed in the following manner:

    .. math::

        \\phi_i = \\textup{arctan} ( \\frac{\\sqrt{{x_i} ^ 2 + {y_i} ^ 2}}{z_i} )

        \\theta_i = \\textup{arctan} ( \\frac{x_i}{y_i} )

    The unit for the angles is **radians** unless ``use_degrees`` is set to ``True``. The angles are modified to be
    in the range of :math:`0` to :math:`\\pi` for :math:`\\phi` and 0 to :math:`\\pi` for
    :math:`\\theta`. The first column in the returned array corresponds to :math:`\\phi` and the second to
    :math:`\\theta`.

    :param vectors: 2D NumPy array containing 3 columns, corresponding to the x, y and z **components** of the
                    vectors, and ``n`` rows, one for each vector. **Note:** We only require the vector *components*,
                    not the *coordinates* in space.
    :param use_degrees: indicate whether the returned angles should be in degrees. If ``False`` (default),
                        the angles will be returned in **radians**.
    :return: 2D NumPy array containing 2 columns, corresponding to :math:`\phi,\theta` for ``n`` rows.
    """
    n = len(vectors)
    x: np.ndarray = vectors[:, 0]
    y: np.ndarray = vectors[:, 1]
    z: np.ndarray = vectors[:, 2]

    # Compute the raw angles using arctan2
    phi = np.arctan2(np.sqrt(x ** 2 + y ** 2), z)
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

    three_dimensional_magnitude = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    in_plane_magnitude = np.sqrt(x ** 2 + y ** 2)

    magnitudes_array = np.zeros((n, 2))
    magnitudes_array[:, MagnitudeType.IN_PLANE] = in_plane_magnitude
    magnitudes_array[:, MagnitudeType.THREE_DIMENSIONAL] = three_dimensional_magnitude

    return magnitudes_array


def create_binned_orientation(vector_orientations: np.ndarray, vector_magnitudes: Optional[np.ndarray],
                              half_number_of_bins: int = 16, use_degrees: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bin the vector orientation data.

    Construct an array containing the histogram data obtained by binning the orientation data. This function will
    return a >=2-dimensional array. If ``vector_magnitudes`` is ``None``, the result is a simple count-based
    2D histogram of :math:`\\phi` vs. :math:`\\theta`. If ``vector_magnitudes`` is provided and contains **both**
    the in-plane and 3D magnitudes, the output is two 2D histograms (stacked one on top of the other), which contain
    magnitude-weighted frequencies.

    The input angles must be in the range :math:`0 \\leq \\phi < \\pi` and :math:`0 \\leq \\theta < \\pi`
    in radians, or :math:`0 \\leq \\phi < 180` and :math:`0 \\leq \\theta < 180` in degrees. There **cannot** be
    any other orientations included, as these will be overwritten.

    Once the bins are assigned, a mirroring step is performed to fill in the missing angles. The mirroring is as
    follows for the case where the magnitudes are provided:

    * In the 3D magnitude-weighted histogram, no mirroring occurs, as :math:`\\phi` only extends from 0 to 180
      degrees. The values are copied to the second half of the bins to allow simpler visualisation. However,
      these values must **not** be used in a 3D visualisation.
    * In the in-plane magnitude-weighted histogram, the values for the :math:`\\theta` bins going from 180 to 360
      degrees are assigned the same values as the bins from 0 to 180 degrees.

    In the case of count-weighted histograms, a simple mirroring approach is used that copies the counts for the
    first half of the theta bins to the second half. No :math:`\\phi`-mirroring is necessary.

    :param vector_orientations: 2D NumPy array containing ``n`` rows, one for each vector, and 2 columns,
                                corresponding to the angles :math:`\phi,\theta`.
    :param vector_magnitudes: Optional 2D NumPy array containing ``n`` rows, one for each vector, and 2 columns,
                              corresponding to the 3D and in-plane magnitudes, respectively. If ``None``,
                              then the histogram is based simply on counts, not on magnitudes.
    :param half_number_of_bins: The half-number of bins. This represents the number of bins that should be produced
                                in the 180\u00b0 (:math:`\\pi` rad) range for each set of angles. This number
                                **must** be even.
    :param use_degrees: Indicate whether the angles are provided in degrees. If ``True``, angles are interpreted as
                        degrees, otherwise the angles are interpreted as radians.
    :return: Tuple containing 2D histogram of :math:`\phi,\theta` and an array providing bounds of the histogram bins.
             If the magnitudes are provided, this will be a two-sheet histogram, with dimensions
             ``(half_number_of_bins * 2, half_number_of_bins * 2, 2)``. If the histogram is count-based,
             the array is 2D (omitting the final index). Axis zero corresponds to :math:`\phi` and axis one
             corresponds to :math:`\theta`. The histogram bins array is of shape ``(2,
             2 * half_half_number_of_bins + 1)``, where the first row/sheet represents the bins for :math:`\phi` and
             the second represents the bins for :math:`\theta`.
    """
    # Indicate whether we are weighting by magnitude
    magnitude_weighted = vector_magnitudes is not None

    # Get the number of vectors
    number_of_vectors = len(vector_orientations)

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

    phi_histogram_bins = np.histogram_bin_edges(phi, bins=number_of_bins, range=(minimum_angle, maximum_angle))
    theta_histogram_bins = np.histogram_bin_edges(theta, bins=number_of_bins, range=(minimum_angle, maximum_angle))

    # Digitize returns indices which are off-by-one
    phi_bin_indices = np.digitize(phi, phi_histogram_bins) - 1
    theta_bin_indices = np.digitize(theta, theta_histogram_bins) - 1

    # Now, to prepare the histogram array, we need to check if we're weighing by magnitude
    if magnitude_weighted:
        angular_histogram_2d = np.zeros((number_of_bins, number_of_bins, 2))
    else:
        angular_histogram_2d = np.zeros((number_of_bins, number_of_bins))

    # Now, for the iterations. We can easily mirror through subtraction so that we can actually modify both the original
    # and the reflected cells at the same time.
    for i in range(number_of_vectors):
        phi_bin = phi_bin_indices[i]
        mirrored_phi_bin = phi_bin - half_number_of_bins

        theta_bin = theta_bin_indices[i]
        mirrored_theta_bin = theta_bin - half_number_of_bins

        if magnitude_weighted:
            angular_histogram_2d[phi_bin, theta_bin] += vector_magnitudes[i]
            angular_histogram_2d[mirrored_phi_bin, mirrored_theta_bin] += vector_magnitudes[i]
        else:
            angular_histogram_2d[phi_bin, theta_bin] += 1
            angular_histogram_2d[mirrored_phi_bin, mirrored_theta_bin] += 1

    # Create an array that contains both the phi and theta histogram boundaries.
    bin_boundaries = np.stack([phi_histogram_bins, theta_histogram_bins])

    return angular_histogram_2d, bin_boundaries


def produce_phi_theta_1d_histogram_data(binned_data: np.ndarray) -> np.ndarray:
    """
    Return the marginal 1D :math:`\\phi,\\theta` histogram arrays.

    This function computes the marginal histogram frequencies for :math:`\\phi, \\theta`. The :math:`\\phi`
    histogram relies on the 3D magnitude while the :math:`\\theta` histogram relies on the in-plane magnitude. If the
    binned data is a 2D array, containing a count-based histogram, both marginals are computed using the same array,
    summing along their respective axes.

    :param binned_data: NumPy array containing the 2D :math:`\phi,\theta` histogram. This may be either
                        magnitude-weighted or count-weighted.
    :return: NumPy array containing the marginal :math:`\phi,\theta` histograms. The zero-axis will have size 2,
             with the first element containing the :math:`\phi` histogram and the second element containing the
             :math:`theta` histogram.
    """
    # Sum along an axis to compute the marginals
    if binned_data.ndim == 3:
        phi_histogram = np.sum(binned_data[..., MagnitudeType.THREE_DIMENSIONAL], axis=AngularIndex.THETA)
        theta_histogram = np.sum(binned_data[..., MagnitudeType.IN_PLANE], axis=AngularIndex.PHI)
    else:
        phi_histogram = np.sum(binned_data, axis=AngularIndex.THETA)
        theta_histogram = np.sum(binned_data, axis=AngularIndex.PHI)

    one_dimensional_histograms = np.stack([phi_histogram, theta_histogram])

    return one_dimensional_histograms


def produce_histogram_plots(binned_data: np.ndarray, bins: np.ndarray, sphere_radius: float = 2.0,
                            zero_position_2d: CardinalDirection = CardinalDirection.NORTH,
                            rotation_direction: RotationDirection = RotationDirection.CLOCKWISE,
                            use_degrees: bool = True, colour_map: str = "gray"):
    """
    Produce a show the anisotropy rose histograms.

    This function produces and shows a 3-panel figure containing (from left to right):

    * The 3D hemisphere plot of :math:`\\phi,\\theta`.
    * The 2D polar histogram of :math:`\\theta`.
    * The 2D polar histogram of :math:`\\phi`.

    A number of plotting parameters may be modified here. See the parameter descriptions for more details.

    :param binned_data: The binned histogram data for the :math:`\phi,\theta` plane.
    :param bins: The boundaries of the bins.
    :param sphere_radius: The radius of the sphere used for 3D plotting.
    :param zero_position_2d: The cardinal direction where zero should be placed in the 2D polar histograms
                             (default: North).
    :param rotation_direction: The direction of increasing angles in the 2D polar histograms (default: clockwise).
    :param use_degrees: Indicate whether the values are in degrees. If ``True``, values are assumed to be in
                        degrees. Otherwise, radians are assumed.
    :param colour_map: Name of the matplotlib colourmap to use to colour the hemisphere. If an invalid name is
                       provided, a default greyscale colourmap ("gray") will be used.
    :return: ``None``, but produces a figure on the screen.
    """
    # Compute the 1D histograms from the binned data
    one_dimensional_histograms = produce_phi_theta_1d_histogram_data(binned_data)
    phi_histogram: np.ndarray = one_dimensional_histograms[AngularIndex.PHI]
    theta_histogram: np.ndarray = one_dimensional_histograms[AngularIndex.THETA]

    # Need to convert the bins back to radians if things have been done in degrees
    if use_degrees:
        bins = np.radians(bins)

    # Remove the last element
    bins = bins[:, :-1]

    # Begin preparing the sphere...
    number_of_bins = binned_data.shape[0]
    half_number_of_bins = number_of_bins // 2

    # Now, the age-old question... What do we want the bounds to be... Well, we want to have the phi go from zero to
    # 180 only! But, we want theta to go from -180 to +180. So, we're going to do just that (but remember,
    # we're in radians, so it'll be 0 to pi for phi and -pi to pi for theta).

    # In terms of the number of bins, we want there to be half as many bins in phi as in theta (I think...),
    # since the phi bins only actually cover half the sphere while the theta bins go all the way around.

    # In the mgrid, we are defining where the bin **dividers** go, not where the bins are! So, recall that we need to
    # have one more divider than the number of bins.
    number_of_phi_dividers = half_number_of_bins + 1
    number_of_theta_dividers = number_of_bins + 1

    sphere_phi, sphere_theta = np.mgrid[
        0: np.pi: number_of_phi_dividers * 1j,
        -np.pi: np.pi: number_of_theta_dividers * 1j
    ]

    sphere_angles = np.stack([sphere_phi, sphere_theta], axis=-1)

    # Get the data to plot on the sphere
    if binned_data.ndim == 3:
        # This first array corresponds to the mirrored angles on the back of the sphere
        sphere_intensity_data_first_half: np.ndarray = binned_data[:half_number_of_bins, :half_number_of_bins,
                                                                   MagnitudeType.THREE_DIMENSIONAL]

        # This second array corresponds to the values on the front half of the sphere
        sphere_intensity_data_second_half: np.ndarray = binned_data[half_number_of_bins:, half_number_of_bins:,
                                                                    MagnitudeType.THREE_DIMENSIONAL]

    else:
        sphere_intensity_data_first_half: np.ndarray = binned_data[:half_number_of_bins, :half_number_of_bins]

        # This second array corresponds to the values on the front half of the sphere
        sphere_intensity_data_second_half: np.ndarray = binned_data[half_number_of_bins:, half_number_of_bins:]

    # Combine the two arrays together
    sphere_intensity_data_first_half: np.ndarray = np.flip(sphere_intensity_data_first_half, axis=0)
    sphere_intensity_data = np.concatenate([sphere_intensity_data_first_half, sphere_intensity_data_second_half],
                                           axis=-1)

    # Get the cartesian coordinates of the sphere
    sphere_cartesian_coordinates = convert_spherical_to_cartesian_coordinates(angular_coordinates=sphere_angles,
                                                                              radius=sphere_radius)

    sphere_x = sphere_cartesian_coordinates[..., 0]
    sphere_y = sphere_cartesian_coordinates[..., 1]
    sphere_z = sphere_cartesian_coordinates[..., 2]

    try:
        mpl_colour_map = plt.get_cmap(colour_map)
    except ValueError:
        mpl_colour_map = plt.get_cmap("gray")

    normaliser = plt.Normalize()
    normalised_sphere_intensities = normaliser(sphere_intensity_data)
    sphere_face_colours = mpl_colour_map(normalised_sphere_intensities)

    # Now, let's also make the axis labels for the 3D plot. We'll have them at a distance of radius * 1.5
    phi_label_positions = np.arange(0, np.pi + 1e-2, np.pi / 6)

    # Remove pi/2 (overlap between both rings)
    phi_label_positions = phi_label_positions[phi_label_positions != np.pi / 2]
    number_of_phi_labels = len(phi_label_positions)
    theta_position_for_phi_labels = np.ones(number_of_phi_labels) * np.pi / 2

    spherical_coordinates_of_phi_labels = np.zeros((number_of_phi_labels, 2))
    spherical_coordinates_of_phi_labels[:, AngularIndex.PHI] = phi_label_positions
    spherical_coordinates_of_phi_labels[:, AngularIndex.THETA] = theta_position_for_phi_labels

    phi_label_positions_cartesian = convert_spherical_to_cartesian_coordinates(
        angular_coordinates=spherical_coordinates_of_phi_labels, radius=1.6*sphere_radius
    )

    phi_label_angles_degrees = np.degrees(phi_label_positions)
    
    # Same thing for theta labels
    theta_label_positions = np.arange(0, 2 * np.pi, np.pi / 6)
    number_of_theta_labels = len(theta_label_positions)
    phi_position_for_theta_labels = np.ones(number_of_theta_labels) * np.pi / 2

    spherical_coordinates_of_theta_labels = np.zeros((number_of_theta_labels, 2))
    spherical_coordinates_of_theta_labels[:, AngularIndex.THETA] = theta_label_positions
    spherical_coordinates_of_theta_labels[:, AngularIndex.PHI] = phi_position_for_theta_labels

    theta_label_positions_cartesian = convert_spherical_to_cartesian_coordinates(
        angular_coordinates=spherical_coordinates_of_theta_labels, radius=1.6 * sphere_radius
    )

    theta_label_angles_degrees = np.degrees(theta_label_positions)

    # Before we produce the 3D plot, let's also handle the 2D information.
    # These two bin widths should be IDENTICAL.
    phi_bins = bins[AngularIndex.PHI]
    theta_bins = bins[AngularIndex.THETA]

    phi_bin_width = phi_bins[1] - phi_bins[0]
    theta_bin_width = theta_bins[1] - theta_bins[0]

    # Now, let's make all the plots together

    # Construct the 3D plot
    plt.figure(figsize=(10, 3.5))
    ax: mpl_toolkits.mplot3d.axes3d.Axes3D = plt.subplot(131, projection="3d")
    ax.set_proj_type('ortho')
    surface = ax.plot_surface(sphere_x, sphere_y, sphere_z, rstride=1, cstride=1, facecolors=sphere_face_colours,
                              alpha=1)
    surface.set_edgecolor("white")
    surface.set_linewidth(0.25)
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_zlim(-1.75, 1.75)
    ax.set_title("Vector Intensity Distribution", fontweight="bold", fontsize=14)

    # Hide the 3D axis
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    # ax.set_axis_off()

    # Add the plane for the spherical axes
    # phi_axis = CirclePolygon((0, 0), radius=1.4 * sphere_radius, fill=False, linewidth=0.5, linestyle="--")
    # theta_axis = CirclePolygon((0, 0), radius=1.4 * sphere_radius, fill=False, linewidth=0.5, linestyle="--")
    #
    # art3d.patch_2d_to_3d(phi_axis, z=0, zdir="y")
    # art3d.patch_2d_to_3d(theta_axis, z=0, zdir="z")

    # ax.add_patch(phi_axis)
    # ax.add_patch(theta_axis)
    phi_axis_positions = np.linspace(0, np.pi)
    phi_position_theta = np.ones_like(phi_axis_positions) * np.pi/2

    phi_axis_polar_positions = np.stack([phi_axis_positions, phi_position_theta], axis=-1)

    phi_axis_cartesian = convert_spherical_to_cartesian_coordinates(phi_axis_polar_positions,
                                                                    radius=1.4 * sphere_radius)

    theta_axis_positions = np.linspace(0, 2*np.pi)
    theta_axis_polar_positions = np.stack([phi_position_theta, theta_axis_positions], axis=-1)

    theta_axis_cartesian = convert_spherical_to_cartesian_coordinates(theta_axis_polar_positions,
                                                                      radius=1.4 * sphere_radius)

    ax.plot(phi_axis_cartesian[:, 0],
            phi_axis_cartesian[:, 1],
            phi_axis_cartesian[:, 2], "k:",
            linewidth=0.5)
    ax.plot(theta_axis_cartesian[:, 0],
            theta_axis_cartesian[:, 1],
            theta_axis_cartesian[:, 2], "k:",
            linewidth=0.5)

    # Add the spherical axis labels
    ax.text3D(0, 0, 1.2 * sphere_radius, r"$\phi$", fontsize='large', clip_on=True, alpha=0.5,
              ha="center")

    for i in range(number_of_phi_labels):
        phi_in_degrees = phi_label_angles_degrees[i]
        phi_label_text = f"{phi_in_degrees:.01f}\u00b0"
        label_position = phi_label_positions_cartesian[i]
        label_x = label_position[0]
        label_y = label_position[1]
        label_z = label_position[2]

        ax.text3D(label_x, label_y, label_z, phi_label_text, ha="center", alpha=0.5, clip_on=True)

    ax.text3D(0, 1.2 * sphere_radius,  0, r"$\theta$", fontsize='large', clip_on=True, alpha=0.5,
              ha="center")
    for i in range(number_of_theta_labels):
        theta_in_degrees = theta_label_angles_degrees[i]
        theta_label_text = f"{theta_in_degrees:.01f}\u00b0"
        label_position = theta_label_positions_cartesian[i]
        label_x = label_position[0]
        label_y = label_position[1]
        label_z = label_position[2]

        ax.text3D(label_x, label_y, label_z, theta_label_text, ha="center", alpha=0.5, clip_on=True)

    # Construct the theta polar plot
    ax2 = plt.subplot(132, projection="polar")
    ax2.set_theta_direction(rotation_direction)
    ax2.set_theta_zero_location(zero_position_2d)
    ax2.set_title(r'$\theta$ (Angle in $XY$)', fontweight="bold", fontsize=14)
    ax2.axes.yaxis.set_ticklabels([])
    start, end = ax2.get_xlim()
    ax2.xaxis.set_ticks(np.arange(start, end, 30 * np.pi / 180))
    ax2.bar(theta_bins, theta_histogram, align='edge', width=theta_bin_width, color="blue")

    # Construct the phi polar plot
    ax3 = plt.subplot(133, projection="polar")
    ax3.set_theta_direction(rotation_direction)
    ax3.set_theta_zero_location(zero_position_2d)
    ax3.set_title(r'$\phi$ (Angle from $+Z$)', fontweight="bold", fontsize=14)
    ax3.axes.yaxis.set_ticklabels([])
    start, end = ax3.get_xlim()
    ax3.xaxis.set_ticks(np.arange(start, end, 30 * np.pi / 180))
    ax3.bar(phi_bins, phi_histogram, align='edge', width=phi_bin_width, color="blue")

    # Show the plots
    plt.subplots_adjust(left=0.05, right=0.95, wspace=0.25)
    plt.show()


def perform_anisotropy_rose_pipeline(vectors: np.ndarray, half_number_of_bins: int = 18, use_degrees: bool = True,
                                     sphere_radius: float = 2.0, weight_by_magnitude: bool = True,
                                     zero_position_2d: CardinalDirection = CardinalDirection.NORTH,
                                     rotation_direction: RotationDirection = RotationDirection.CLOCKWISE,
                                     colour_map: str = "gray"):
    """
    Run the entire anisotropy rose pipeline.

    Construct anisotropy roses from a set of vectors. For more details about each step, please consult the relevant
    functions.

    :param vectors: ``n`` by 6 or ``n`` by 3 array of vectors whose orientations will be analysed. If the vector
                    array contains 6 columns, **the last three are assumed to be the components**.
    :param half_number_of_bins: number of bins in :math:`\phi,\theta` in half the angular range.
    :param use_degrees: Indicates whether the angles should be computed in degrees. If ``True``, all angles will be
                        stored in degrees (default). Otherwise, all angles will be stored in radians.
    :param sphere_radius: Radius of the sphere for plotting.
    :param weight_by_magnitude: Indicate whether the histograms should be weighted by magnitude. If ``True``,
                                the :math:`\phi` histogram is weighted by the 3D magnitude and the :math:`\theta`
                                histogram is weighted by the magnitude in the :math:`(x,y)`-plane.
    :param zero_position_2d: The cardinal orientation of zero in the 2D polar histograms. Default: North.
    :param rotation_direction: The direction of increasing angles in the 2D polar histograms. Default: Clockwise.
    :param colour_map: Name of the matplotlib colour map to be used in the 3D hemisphere plot. If an invalid name is
                       specified, the default greyscale map ("gray") is used.
    :return: ``None``, but produces a figure on screen.

    **TODO: Add the ability to easily save from here.**
    """

    # First, check the size of the vector array. Only keep the components. We can discard the coordinates
    if vectors.shape[1] > 3:
        vectors = vectors[:, 3:6]

    # Remove the zero-magnitude vectors
    non_zero_vectors = remove_zero_vectors(vectors)

    # Compute the angles
    vector_angles = compute_vector_orientation_angles(vectors=non_zero_vectors, use_degrees=use_degrees)

    # If weighing by magnitude, compute the magnitudes.
    if weight_by_magnitude:
        vector_magnitudes = compute_vector_magnitudes(non_zero_vectors)
    else:
        vector_magnitudes = None

    # Bin the data into the 2D histogram
    binned_data, bins = create_binned_orientation(vector_orientations=vector_angles,
                                                  vector_magnitudes=vector_magnitudes,
                                                  half_number_of_bins=half_number_of_bins, use_degrees=use_degrees)

    produce_histogram_plots(binned_data=binned_data, bins=bins, sphere_radius=sphere_radius,
                            zero_position_2d=zero_position_2d, rotation_direction=rotation_direction,
                            use_degrees=use_degrees, colour_map=colour_map)
