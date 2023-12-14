"""
Anisotropy Rose

Joseph Deering, Benjamin Rudski
2023

This module provides the ability to construct 2D and 3D rose diagrams
of anisotropy vector fields.

"""

import enum
from typing import Optional, Tuple, Any

import mpl_toolkits.mplot3d.axes3d
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.projections
import numpy
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


class AngularUnits(enum.Enum):
    """
    Angular Units

    This enumerated type represents angular units (degrees or radians).
    It **does not** provide any implementation for converting from one
    to the other, as this functionality is already very well included
    in NumPy.

    Attributes:
        * DEGREES: Indicates that angle is in degrees (typically in the
            range 0 to 360 or -180 to +180).
        * RADIANS: Indicates that angle is in radians (typically in the
            range 0 to :math:`2\\pi` or :math:`-\\pi` to :math:`\\pi`).
    """

    DEGREES = 0
    RADIANS = 1


def produce_phi_theta_1d_histogram_data(
    binned_data: np.ndarray,
    weight_by_magnitude: bool = True,
) -> np.ndarray:
    """
    Return the marginal 1D :math:`\\phi,\\theta` histogram arrays.

    This function computes the marginal histogram frequencies for
    :math:`\\phi, \\theta`. The :math:`\\phi` histogram relies on the 3D
    magnitude while the :math:`\\theta` histogram relies on the in-plane
    magnitude. If the binned data is a 2D array, containing a
    count-based histogram, both marginals are computed using the same
    array, summing along their respective axes.

    :param binned_data: NumPy array containing the
        2D :math:`\phi,\theta` histogram. This may be either
        magnitude-weighted or count-weighted.
    :param weight_by_magnitude: Indicate whether the 1D histograms
        should be weighted by magnitude. If ``False``, the produced
        histograms will be weighted by count.
    :return: NumPy array containing the marginal :math:`\phi,\theta`
        histograms. The zero-axis will have size 2, with the first
        element containing the :math:`\phi` histogram and the second
        element containing the :math:`theta` histogram.
    """
    # Sum along an axis to compute the marginals
    if weight_by_magnitude:
        phi_histogram = np.sum(
            binned_data[..., MagnitudeType.THREE_DIMENSIONAL], axis=AngularIndex.THETA
        )
        theta_histogram = np.sum(
            binned_data[..., MagnitudeType.IN_PLANE], axis=AngularIndex.PHI
        )
    else:
        phi_histogram = np.sum(
            binned_data[..., MagnitudeType.COUNT], axis=AngularIndex.THETA
        )
        theta_histogram = np.sum(
            binned_data[..., MagnitudeType.COUNT], axis=AngularIndex.PHI
        )

    one_dimensional_histograms = np.stack([phi_histogram, theta_histogram])

    return one_dimensional_histograms


def prepare_two_dimensional_histogram(binned_data: np.ndarray) -> np.ndarray:
    """
    Prepare the binned data for plotting as a spherical histogram.

    This function takes the 2D binned data of shape ``(2n, 2n)`` and
    returns the binned histogram data to plot on a sphere, having shape
    ``(n, 2n)``, where ``n`` corresponds to the half-number of bins. In
    the final array that is returned, the ``n`` rows correspond to the
    ``n`` histogram bins in :math:`\\phi` going from :math:`0` to
    :math:`\\pi/2`, while the ``2n`` columns represent the bins in
    :math:`\\theta`, going from :math:`-\\pi` to :math:`\\pi`. To
    prepare for wrapping around the sphere, the first ``n`` columns are
    row-inverted with respect to the last ``n`` columns. This is done
    to correspond to the "negative" values of :math:`\\phi`` (which do
    not actually exist, but are used as a convenience to plot the full
    sphere).

    :param binned_data: Binned 2D histogram data of shape ``(2n, 2n)``.
    :return: Adjusted histogram data of shape ``(n, 2n)``.
    """

    number_of_bins = binned_data.shape[0]
    half_number_of_bins = number_of_bins // 2

    # This first array corresponds to the mirrored angles on the
    # back of the sphere
    sphere_intensity_data_first_half: np.ndarray = binned_data[
        :half_number_of_bins, :half_number_of_bins
    ]

    # This second array corresponds to the values on the
    # front half of the sphere
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

    return sphere_intensity_data


class SphereProjection(enum.Enum):
    """
    Sphere projection method for 3D figures.

    Enumerated type for the projection method for 3D figures. Value
    type: ``str``.

    Attributes:
        * ORTHOGRAPHIC: orthographic projection.
        * PERSPECTIVE: perspective projection.
    """

    ORTHOGRAPHIC = "ortho"
    PERSPECTIVE = "persp"


def produce_spherical_histogram_plot(
    ax: mpl_toolkits.mplot3d.axes3d.Axes3D,
    sphere_radius: float,
    histogram_data: np.ndarray,
    weight_by_magnitude: bool,
    plot_title: Optional[str] = None,
    minimum_value: Optional[float] = None,
    maximum_value: Optional[float] = None,
    colour_map: str = "viridis",
    sphere_projection: SphereProjection = SphereProjection.ORTHOGRAPHIC,
    sphere_alpha: float = 1.0,
    plot_phi_axis: bool = True,
    plot_theta_axis: bool = True,
    label_phi_axis: bool = True,
    label_theta_axis: bool = True,
    phi_label_positions: np.ndarray = np.arange(0, np.pi + 1e-2, np.pi / 6),
    theta_label_positions: np.ndarray = np.arange(0, 2 * np.pi, np.pi / 6),
    axes_x_limits: tuple[float] = (-2.2, 2.2),
    axes_y_limits: tuple[float] = (-2.2, 2.2),
    axes_z_limits: tuple[float] = (-1.75, 1.75),
    hide_cartesian_axis_labels: bool = False,
    hide_cartesian_axis_ticks: bool = True,
    plot_colourbar: bool = False,
    colour_bar_kwargs: dict[str, Any] = {},
) -> mpl_toolkits.mplot3d.axes3d.Axes3D:
    """
    Produce a spherical histogram plot on the provided axes.

    Using the provided axes, produce a spherical histogram plot. This
    plot has a constant radius and is coloured using the provided data
    using the specified colour map. The data can optionally be
    normalised to fit a new range. The provided axes must have the
    projection set to 3D using ``projection='3d'``.

    The histogram data provided must occupy the entire sphere. No
    manipulations will be performed in this function to get the face
    colours to match the number of faces.

    :param ax: Matplotlib ``Axes3D`` on which to plot the 3D spherical
        histogram. The projection of these axes **must** be 3D.
    :param sphere_radius: Radius of the sphere to plot.
    :param histogram_data: Binned data to plot. This data should have
        the shape ``(2n, 2n)`` where ``n`` represents the half-number of
        histogram bins. We currently assume that the half-number of bins
        is the **same** in both :math:`\\phi` and :math:`\\theta`. This
        function separates the data to plot half of it on the sphere.
    :param weight_by_magnitude: Indicate whether plots should be
        weighted by magnitude or simply by count.
    :param plot_title: Title of the plot produced (optional).
    :param minimum_value: Minimum value for data normalisation. If not
        specified, the minimum of the data is automatically used
        instead.
    :param maximum_value: Maximum value for data normalisation. If not
        specified, the maximum of the data is automatically used
        instead.
    :param colour_map: Name of the Matplotlib colour map to be used for
        colouring the histogram data.
    :param sphere_projection: 3D projection method to be used for the
        spherical figure. Options are orthographic and perspective
        projection.
    :param sphere_alpha: Opacity of the sphere.
    :param plot_phi_axis: Indicate whether the :math:`\\phi` axis should
        be plotted in 3D.
    :param plot_theta_axis: Indicate whether the :math:`\\theta` axis
        should be plotted in 3D.
    :param label_phi_axis: Indicate whether to label the :math:`\\phi`
        axis.
    :param label_theta_axis: Indicate whether to label the
        :math:`\\theta` axis.
    :param phi_label_positions: Indicate angular positions for the
        labels for :math:`\\phi` along its circular axis.
    :param theta_label_positions: Indicate angular positions for the
        labels for :math:`\\theta` along its circular axis.
    :param axes_x_limits: Figure size limits along the ``x``-axis.
    :param axes_y_limits: Figure size limits along the ``y``-axis.
    :param axes_z_limits: Figure size limits along the ``z``-axis.
    :param hide_cartesian_axis_labels: Indicate whether to hide the
        axis labels for the cartesian axes.
    :param hide_cartesian_axis_ticks: Indicate whether to hide the axis
        ticks for the cartesian axes.
    :param plot_colourbar: Indicate whether to include a colour bar on
        the plot.
    :param colour_bar_kwargs: keyword arguments for the colour bar.
    :return: a reference to the ``Axes3D`` object passed in as ``ax``.
    """

    # Get the data to plot on the sphere. We must determine if we want
    # it to be magnitude-weighted or count weighted.
    if weight_by_magnitude:
        original_intensity_data = histogram_data[..., MagnitudeType.THREE_DIMENSIONAL]
    else:
        original_intensity_data = histogram_data[..., MagnitudeType.COUNT]

    cleaned_histogram_data = prepare_two_dimensional_histogram(
        binned_data=original_intensity_data
    )

    half_number_of_bins, number_of_bins = cleaned_histogram_data.shape

    # Now, the age-old question... What do we want the bounds to be...
    # Well, we want to have the phi go from zero to 180 only! But, we
    # want theta to go from -180 to +180. So, we're going to do just
    # that (but remember, we're in radians, so it'll be 0 to pi for phi
    # and -pi to pi for theta).

    # In terms of the number of bins, we want there to be half as many
    # bins in phi as in theta (I think...), since the phi bins only
    # actually cover half the sphere while the theta bins go all the way
    # around.

    # In the mgrid, we are defining where the bin **dividers** go, not
    # where the bins are! So, recall that we need to have one more
    # divider than the number of bins.
    number_of_phi_dividers = half_number_of_bins + 1
    number_of_theta_dividers = number_of_bins + 1

    sphere_phi, sphere_theta = np.mgrid[
        0 : np.pi : number_of_phi_dividers * 1j,
        -np.pi : np.pi : number_of_theta_dividers * 1j,
    ]

    sphere_angles = np.stack([sphere_phi, sphere_theta], axis=-1)

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

    normaliser = plt.Normalize(vmin=minimum_value, vmax=maximum_value)
    normalised_sphere_intensities = normaliser(cleaned_histogram_data)
    sphere_face_colours = mpl_colour_map(normalised_sphere_intensities)

    # Construct the 3D plot
    ax.set_proj_type(sphere_projection.value)

    surface = ax.plot_surface(
        sphere_x,
        sphere_y,
        sphere_z,
        rstride=1,
        cstride=1,
        facecolors=sphere_face_colours,
        alpha=sphere_alpha,
    )
    # surface.set_edgecolor("white")
    # surface.set_linewidth(0.25)
    ax.set_xlim(*axes_x_limits)
    ax.set_ylim(*axes_y_limits)
    ax.set_zlim(*axes_z_limits)
    ax.set_title(plot_title, fontsize=14)

    # Hide the 3D axis
    if hide_cartesian_axis_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    if hide_cartesian_axis_labels:
        ax.set_axis_off()
    else:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    if plot_phi_axis:
        phi_axis_positions = np.linspace(0, np.pi)
        phi_position_theta = np.ones_like(phi_axis_positions) * np.pi / 2

        phi_axis_polar_positions = np.stack(
            [phi_axis_positions, phi_position_theta], axis=-1
        )

        phi_axis_cartesian = convert_spherical_to_cartesian_coordinates(
            phi_axis_polar_positions, radius=1.4 * sphere_radius
        )

        ax.plot(
            phi_axis_cartesian[:, 0],
            phi_axis_cartesian[:, 1],
            phi_axis_cartesian[:, 2],
            "k:",
            linewidth=0.5,
        )

    if plot_theta_axis:
        theta_axis_positions = np.linspace(0, 2 * np.pi)
        theta_position_phi = np.ones_like(theta_axis_positions) * np.pi / 2
        theta_axis_polar_positions = np.stack(
            [theta_position_phi, theta_axis_positions], axis=-1
        )

        theta_axis_cartesian = convert_spherical_to_cartesian_coordinates(
            theta_axis_polar_positions, radius=1.4 * sphere_radius
        )

        ax.plot(
            theta_axis_cartesian[:, 0],
            theta_axis_cartesian[:, 1],
            theta_axis_cartesian[:, 2],
            "k:",
            linewidth=0.5,
        )

    # Add the spherical axis labels
    if label_phi_axis:
        # Now, let's also make the axis labels for the 3D plot. We'll
        # have them at a distance of radius * 1.6
        if label_theta_axis:
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

    if label_theta_axis:
        # Same thing as for the phi axis
        number_of_theta_labels = len(theta_label_positions)
        phi_position_for_theta_labels = np.ones(number_of_theta_labels) * np.pi / 2

        spherical_coordinates_of_theta_labels = np.zeros((number_of_theta_labels, 2))

        spherical_coordinates_of_theta_labels[
            :, AngularIndex.THETA
        ] = theta_label_positions

        spherical_coordinates_of_theta_labels[
            :, AngularIndex.PHI
        ] = phi_position_for_theta_labels

        theta_label_positions_cartesian = convert_spherical_to_cartesian_coordinates(
            angular_coordinates=spherical_coordinates_of_theta_labels,
            radius=1.6 * sphere_radius,
        )

        theta_label_angles_degrees = np.degrees(theta_label_positions)

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

    if plot_colourbar:
        scalar_mappable = mpl.cm.ScalarMappable(norm=normaliser, cmap=mpl_colour_map)
        plt.colorbar(mappable=scalar_mappable, ax=ax, **colour_bar_kwargs)

    return ax


def produce_polar_histogram_plot(
    ax: mpl.projections.polar.PolarAxes,
    data: numpy.ndarray,
    bins: np.ndarray,
    zero_position: CardinalDirection = CardinalDirection.NORTH,
    rotation_direction: RotationDirection = RotationDirection.CLOCKWISE,
    plot_title: Optional[str] = None,
    axis_ticks_increment: Optional[float] = 30,
    axis_ticks_increment_unit: AngularUnits = AngularUnits.DEGREES,
    colour: str = "blue",
) -> mpl.projections.polar.PolarAxes:
    """
    Produce 2D polar histogram.

    Produce a 2D polar histogram using the specified data on the
    provided axes. The axes provided **must** be created using
    ``projection="Polar"``.

    :param ax: Matplotlib ``PolarAxes`` on which to plot the data.
    :param data: Data to plot. This should have the same size
        as ``bins``.
    :param bins: Lower value of each bin in the histogram.
    :param zero_position: Cardinal direction corresponding to where 0
        should be placed on the polar axes.
    :param rotation_direction: Direction in which the bin values should
        increase from the zero-point specified in ``zero_position``.
    :param plot_title: Optional title of the plot.
    :param axis_ticks_increment: Increment of the polar axis ticks. This
        value may be specified in degrees or radians. Unit is specified
        in ``axis_ticks_increment_unit``. If ``None``, then no axis
        ticks are included.
    :param axis_ticks_increment_unit: Indicates what angular unit is
        used for specifying the axis ticks. Default is degrees.
    :param colour: Colour used for the histogram bars. Must be a valid
        matplotlib colour.
    :return: The ``PolarAxes`` used for plotting.
    """

    bin_width = bins[1] - bins[0]
    ax.set_theta_direction(rotation_direction)
    ax.set_theta_zero_location(zero_position)
    ax.set_title(plot_title)
    ax.axes.yaxis.set_ticklabels([])

    if axis_ticks_increment is not None:
        start, end = ax.get_xlim()

        if axis_ticks_increment_unit is AngularUnits.DEGREES:
            axis_ticks_increment = np.radians(axis_ticks_increment)

        ax.xaxis.set_ticks(np.arange(start, end, axis_ticks_increment))
    else:
        ax.xaxis.set_ticks([])

    ax.bar(bins, data, align="edge", width=bin_width, color=colour)

    return ax


def produce_histogram_plots(
    binned_data: np.ndarray,
    bins: np.ndarray,
    sphere_radius: float = 2.0,
    zero_position_2d: CardinalDirection = CardinalDirection.NORTH,
    rotation_direction: RotationDirection = RotationDirection.CLOCKWISE,
    use_degrees: bool = True,
    colour_map: str = "gray",
    plot_title: Optional[str] = None,
    weight_by_magnitude: bool = True,
    **kwargs: dict[str, Any],
):
    """
    Produce a show the anisotropy rose histograms.

    This function produces and shows a 3-panel figure containing
    (from left to right):

    * The 3D hemisphere plot of :math:`\\phi,\\theta`.
    * The 2D polar histogram of :math:`\\theta`.
    * The 2D polar histogram of :math:`\\phi`.

    A number of plotting parameters may be modified here. See the
    parameter descriptions for more details.

    :param binned_data: The binned histogram data for the
        :math:`\phi,\theta` plane.
    :param bins: The boundaries of the bins.
    :param sphere_radius: The radius of the sphere used for 3D plotting.
    :param zero_position_2d: The cardinal direction where zero should be
        placed in the 2D polar histograms (default: North).
    :param rotation_direction: The direction of increasing angles in the
        2D polar histograms (default: clockwise).
    :param use_degrees: Indicate whether the values are in degrees. If
        ``True``, values are assumed to be in degrees. Otherwise,
        radians are assumed.
    :param colour_map: Name of the matplotlib colourmap to use to colour
        the hemisphere. If an invalid name is provided, a default
        greyscale colourmap ("gray") will be used.
    :param plot_title: title of the overall plot (optional).
    :param weight_by_magnitude: Indicate whether plots should be
        weighted by magnitude or simply by count.
    :param kwargs: extra keyword arguments for plotting.
    :return: ``None``, but produces a figure on the screen.
    """
    # Compute the 1D histograms from the binned data
    one_dimensional_histograms = produce_phi_theta_1d_histogram_data(
        binned_data, weight_by_magnitude=weight_by_magnitude
    )
    phi_histogram: np.ndarray = one_dimensional_histograms[AngularIndex.PHI]
    theta_histogram: np.ndarray = one_dimensional_histograms[AngularIndex.THETA]

    # Construct the 3D plot
    plt.figure(figsize=(10, 3.5))
    ax: mpl_toolkits.mplot3d.axes3d.Axes3D = plt.subplot(131, projection="3d")

    ax = produce_spherical_histogram_plot(
        ax=ax,
        sphere_radius=sphere_radius,
        histogram_data=binned_data,
        weight_by_magnitude=weight_by_magnitude,
        plot_title="Vector Intensity Distribution",
        colour_map=colour_map,
        **kwargs,
    )

    # Construct the 2D plots
    # Need to convert the bins back to radians if things have been done in degrees
    if use_degrees:
        bins = np.radians(bins)

    # Remove the last element
    bins = bins[:, :-1]

    # These two bin widths should be IDENTICAL.
    phi_bins = bins[AngularIndex.PHI]
    theta_bins = bins[AngularIndex.THETA]

    # Construct the theta polar plot
    ax2 = plt.subplot(132, projection="polar")
    ax2 = produce_polar_histogram_plot(
        ax=ax2,
        data=theta_histogram,
        bins=theta_bins,
        zero_position=zero_position_2d,
        rotation_direction=rotation_direction,
        plot_title=r"$\theta$ (Angle in $XY$)",
    )

    # Construct the phi polar plot
    ax3 = plt.subplot(133, projection="polar")
    ax3 = produce_polar_histogram_plot(
        ax=ax3,
        data=phi_histogram,
        bins=phi_bins,
        zero_position=zero_position_2d,
        rotation_direction=rotation_direction,
        plot_title=r"$\phi$ (Angle from $+Z$)",
    )

    # Show the plots
    plt.suptitle(plot_title, fontweight="bold", fontsize=14)
    plt.subplots_adjust(left=0.05, right=0.95, wspace=0.25)
    plt.show()
