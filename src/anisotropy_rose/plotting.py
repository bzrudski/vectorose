"""
Anisotropy Rose

Joseph Deering, Benjamin Rudski
2023

This module provides the ability to construct 2D and 3D rose diagrams
of anisotropy vector fields.

"""

import enum
from typing import Optional, Tuple

import mpl_toolkits.mplot3d.axes3d
import matplotlib.pyplot as plt
import numpy as np

from .core import (
    MagnitudeType,
    AngularIndex,
    convert_spherical_to_cartesian_coordinates,
)


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
        phi_histogram = np.sum(
            binned_data[..., MagnitudeType.THREE_DIMENSIONAL], axis=AngularIndex.THETA
        )
        theta_histogram = np.sum(
            binned_data[..., MagnitudeType.IN_PLANE], axis=AngularIndex.PHI
        )
    else:
        phi_histogram = np.sum(binned_data, axis=AngularIndex.THETA)
        theta_histogram = np.sum(binned_data, axis=AngularIndex.PHI)

    one_dimensional_histograms = np.stack([phi_histogram, theta_histogram])

    return one_dimensional_histograms


def produce_histogram_plots(
    binned_data: np.ndarray,
    bins: np.ndarray,
    sphere_radius: float = 2.0,
    zero_position_2d: CardinalDirection = CardinalDirection.NORTH,
    rotation_direction: RotationDirection = RotationDirection.CLOCKWISE,
    use_degrees: bool = True,
    colour_map: str = "gray",
    plot_title: Optional[str] = None
):
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
    :param plot_title: title of the overall plot (optional).
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
        0 : np.pi : number_of_phi_dividers * 1j,
        -np.pi : np.pi : number_of_theta_dividers * 1j,
    ]

    sphere_angles = np.stack([sphere_phi, sphere_theta], axis=-1)

    # Get the data to plot on the sphere
    if binned_data.ndim == 3:
        # This first array corresponds to the mirrored angles on the back of the sphere
        sphere_intensity_data_first_half: np.ndarray = binned_data[
            :half_number_of_bins, :half_number_of_bins, MagnitudeType.THREE_DIMENSIONAL
        ]

        # This second array corresponds to the values on the front half of the sphere
        sphere_intensity_data_second_half: np.ndarray = binned_data[
            half_number_of_bins:, half_number_of_bins:, MagnitudeType.THREE_DIMENSIONAL
        ]

    else:
        sphere_intensity_data_first_half: np.ndarray = binned_data[
            :half_number_of_bins, :half_number_of_bins
        ]

        # This second array corresponds to the values on the front half of the sphere
        sphere_intensity_data_second_half: np.ndarray = binned_data[
            half_number_of_bins:, half_number_of_bins:
        ]

    # Combine the two arrays together
    sphere_intensity_data_first_half: np.ndarray = np.flip(
        sphere_intensity_data_first_half, axis=0
    )
    sphere_intensity_data = np.concatenate(
        [sphere_intensity_data_first_half, sphere_intensity_data_second_half], axis=-1
    )

    # Get the cartesian coordinates of the sphere
    sphere_cartesian_coordinates = convert_spherical_to_cartesian_coordinates(
        angular_coordinates=sphere_angles, radius=sphere_radius
    )

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
    spherical_coordinates_of_phi_labels[
        :, AngularIndex.THETA
    ] = theta_position_for_phi_labels

    phi_label_positions_cartesian = convert_spherical_to_cartesian_coordinates(
        angular_coordinates=spherical_coordinates_of_phi_labels,
        radius=1.6 * sphere_radius,
    )

    phi_label_angles_degrees = np.degrees(phi_label_positions)

    # Same thing for theta labels
    theta_label_positions = np.arange(0, 2 * np.pi, np.pi / 6)
    number_of_theta_labels = len(theta_label_positions)
    phi_position_for_theta_labels = np.ones(number_of_theta_labels) * np.pi / 2

    spherical_coordinates_of_theta_labels = np.zeros((number_of_theta_labels, 2))
    spherical_coordinates_of_theta_labels[:, AngularIndex.THETA] = theta_label_positions
    spherical_coordinates_of_theta_labels[
        :, AngularIndex.PHI
    ] = phi_position_for_theta_labels

    theta_label_positions_cartesian = convert_spherical_to_cartesian_coordinates(
        angular_coordinates=spherical_coordinates_of_theta_labels,
        radius=1.6 * sphere_radius,
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
    ax.set_proj_type("ortho")
    surface = ax.plot_surface(
        sphere_x,
        sphere_y,
        sphere_z,
        rstride=1,
        cstride=1,
        facecolors=sphere_face_colours,
        alpha=1,
    )
    # surface.set_edgecolor("white")
    # surface.set_linewidth(0.25)
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_zlim(-1.75, 1.75)
    ax.set_title("Vector Intensity Distribution", fontsize=14)

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
    phi_position_theta = np.ones_like(phi_axis_positions) * np.pi / 2

    phi_axis_polar_positions = np.stack(
        [phi_axis_positions, phi_position_theta], axis=-1
    )

    phi_axis_cartesian = convert_spherical_to_cartesian_coordinates(
        phi_axis_polar_positions, radius=1.4 * sphere_radius
    )

    theta_axis_positions = np.linspace(0, 2 * np.pi)
    theta_axis_polar_positions = np.stack(
        [phi_position_theta, theta_axis_positions], axis=-1
    )

    theta_axis_cartesian = convert_spherical_to_cartesian_coordinates(
        theta_axis_polar_positions, radius=1.4 * sphere_radius
    )

    ax.plot(
        phi_axis_cartesian[:, 0],
        phi_axis_cartesian[:, 1],
        phi_axis_cartesian[:, 2],
        "k:",
        linewidth=0.5,
    )
    ax.plot(
        theta_axis_cartesian[:, 0],
        theta_axis_cartesian[:, 1],
        theta_axis_cartesian[:, 2],
        "k:",
        linewidth=0.5,
    )

    # Add the spherical axis labels
    ax.text3D(
        0,
        0,
        1.2 * sphere_radius,
        r"$\phi$",
        fontsize="large",
        clip_on=True,
        alpha=0.5,
        ha="center",
    )

    for i in range(number_of_phi_labels):
        phi_in_degrees = phi_label_angles_degrees[i]
        phi_label_text = f"{phi_in_degrees:.01f}\u00b0"
        label_position = phi_label_positions_cartesian[i]
        label_x = label_position[0]
        label_y = label_position[1]
        label_z = label_position[2]

        ax.text3D(
            label_x,
            label_y,
            label_z,
            phi_label_text,
            ha="center",
            alpha=0.5,
            clip_on=True,
        )

    ax.text3D(
        0,
        1.2 * sphere_radius,
        0,
        r"$\theta$",
        fontsize="large",
        clip_on=True,
        alpha=0.5,
        ha="center",
    )
    for i in range(number_of_theta_labels):
        theta_in_degrees = theta_label_angles_degrees[i]
        theta_label_text = f"{theta_in_degrees:.01f}\u00b0"
        label_position = theta_label_positions_cartesian[i]
        label_x = label_position[0]
        label_y = label_position[1]
        label_z = label_position[2]

        ax.text3D(
            label_x,
            label_y,
            label_z,
            theta_label_text,
            ha="center",
            alpha=0.5,
            clip_on=True,
        )

    # Construct the theta polar plot
    ax2 = plt.subplot(132, projection="polar")
    ax2.set_theta_direction(rotation_direction)
    ax2.set_theta_zero_location(zero_position_2d)
    ax2.set_title(r"$\theta$ (Angle in $XY$)", fontsize=14)
    ax2.axes.yaxis.set_ticklabels([])
    start, end = ax2.get_xlim()
    ax2.xaxis.set_ticks(np.arange(start, end, 30 * np.pi / 180))
    ax2.bar(
        theta_bins, theta_histogram, align="edge", width=theta_bin_width, color="blue"
    )

    # Construct the phi polar plot
    ax3 = plt.subplot(133, projection="polar")
    ax3.set_theta_direction(rotation_direction)
    ax3.set_theta_zero_location(zero_position_2d)
    ax3.set_title(r"$\phi$ (Angle from $+Z$)", fontsize=14)
    ax3.axes.yaxis.set_ticklabels([])
    start, end = ax3.get_xlim()
    ax3.xaxis.set_ticks(np.arange(start, end, 30 * np.pi / 180))
    ax3.bar(phi_bins, phi_histogram, align="edge", width=phi_bin_width, color="blue")

    # Show the plots
    plt.suptitle(plot_title, fontweight="bold", fontsize=14)
    plt.subplots_adjust(left=0.05, right=0.95, wspace=0.25)
    plt.show()
