# Copyright (c) 2023-, Benjamin Rudski, Joseph Deering
#
# This code is licensed under the MIT License. See the `LICENSE` file for
# more details about copying.

"""
Functions for plotting vector roses.

This module provides the ability to construct 2D and 3D rose diagrams
of orientation/vector fields.

"""

import enum
import functools
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.animation
import matplotlib.cm
import matplotlib.colorbar
import matplotlib.colors
import matplotlib.figure
import matplotlib.projections
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d
import mpl_toolkits.mplot3d.art3d
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from .vectorose import MagnitudeType, produce_phi_theta_1d_histogram_data
from .util import (AngularIndex, convert_spherical_to_cartesian_coordinates,
                   compute_vector_orientation_angles)

from .tregenza_sphere import TregenzaSphereBase


class CardinalDirection(str, enum.Enum):
    """
    Cardinal Directions

    This string-based enumerated type is useful when preparing 2D polar
    figures. Members reflect cardinal directions, which may be used to
    indicate positions on circular (polar) axes. The values are consistent
    with the Matplotlib convention (see
    :meth:`matplotlib.projections.polar.PolarAxes.set_theta_zero_location`
    for details).

    Members
    -------
    NORTH
        Location directly upwards.

    NORTH_WEST
        Location in the upper left corner.

    WEST
        Location on the left side.

    SOUTH_WEST
        Location in the lower left corner.

    SOUTH
        Location directly downwards.

    SOUTH_EAST
        Location in the lower right corner.

    EAST
        Location on the right side.

    NORTH_EAST
        Location in the upper right corner.

    See Also
    --------
    matplotlib.projections.polar.PolarAxes.set_theta_zero_location:
        Set the zero position of a polar plot using one of the member
        values for this type.

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

    This integer-based enumerated type represents two-dimensional rotation
    direction. The convention used is consistent with the Matplotlib
    documentation (see
    :meth:matplotlib.projections.polar.PolarAxes.set_theta_direction for
    details).

    Members
    -------
    CLOCKWISE:
        Clockwise, or rightward rotation.

    COUNTER_CLOCKWISE:
        Counter-clockwise, anti-clockwise, or leftward rotation.

    See Also
    --------
    matplotlib.projections.polar.PolarAxes.set_theta_direction:
        Set the rotation direction of a polar plot using one of the member
        values for this type.
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

    Members
    -------
    DEGREES:
        Represent angles in degrees (typically in the range 0 to 360 or
        -180 to +180).

    RADIANS:
        Indicates that angle is in radians (typically in the range 0
        to :math:`2\\pi` or :math:`-\\pi` to :math:`\\pi`).

    See Also
    --------
    numpy.degrees: Convert numeric values from radians into degrees.
    numpy.radians: Convert numeric values from degrees into radians.
    """

    DEGREES = 0
    RADIANS = 1


class SphereProjection(enum.Enum):
    """Projection type for 3D figures.

    Enumerated type representing the projection type for 3D figures. The
    values of the members are compatible with the Matplotlib 3D axes method
    :meth:`mpl_toolkits.mplot3d.axes3d.Axes3D.set_proj_type`.


    Members
    -------

    ORTHOGRAPHIC
        Orthographic projection.

    PERSPECTIVE
        Perspective projection.
    """

    ORTHOGRAPHIC = "ortho"
    PERSPECTIVE = "persp"


def produce_labelled_3d_plot(
    ax: mpl_toolkits.mplot3d.axes3d.Axes3D,
    radius: float,
    limits_factor: float = 1.1,
    plot_title: Optional[str] = None,
    sphere_projection: SphereProjection = SphereProjection.ORTHOGRAPHIC,
    plot_phi_axis: bool = True,
    plot_theta_axis: bool = True,
    label_phi_axis: bool = True,
    label_theta_axis: bool = True,
    phi_label_positions: np.ndarray = np.arange(0, np.pi + 1e-2, np.pi / 6),
    theta_label_positions: np.ndarray = np.arange(0, 2 * np.pi, np.pi / 6),
    phi_axis_colour: str = "black",
    theta_axis_colour: str = "black",
    hide_cartesian_axes: bool = True,
    hide_cartesian_axis_labels: bool = False,
    hide_cartesian_axis_ticks: bool = True,
    plot_colour_bar: bool = False,
    minimum_value: Optional[float] = None,
    maximum_value: Optional[float] = None,
    colour_map: str = "viridis",
    colour_bar_kwargs: Optional[dict[str, Any]] = None,
    axis_label_factor: float = 1.4,
    axis_tick_factor: float = 1.6,
    norm: Optional[matplotlib.colors.Normalize] = None,
) -> mpl_toolkits.mplot3d.axes3d.Axes3D:
    """Modify a 3D plot to label it with spherical axes.

    Modify existing axes to add spherical phi and theta axes, as well as
    labels and a colour bar.

    Parameters
    ----------
    ax
        Axes to modify. These must be 3D axes.
    radius
        Radius of the 3D plot. This value is multiplied by
        the `limits_factor` to obtain the radius of the spherical axes.
    limits_factor
        Factor used to add padding to the sphere, by default 1.1. The same
        factor is used along all axes, and is multiplied by the radius of
        the sphere to define the axis bounds.
    plot_title
        Title of the plot produced (optional).
    sphere_projection
        Projection used to plot the sphere, by default
        :attr:`SphereProjection.ORTHOGRAPHIC`
    plot_phi_axis
        Indicate whether the phi axis should be plotted in 3D.
    plot_theta_axis
        Indicate whether the theta axis should be plotted in 3D.
    label_phi_axis
        Indicate whether to label the phi axis.
    label_theta_axis
        Indicate whether to label the theta axis.
    phi_label_positions
        Indicate angular positions for the labels for phi along its
        circular axis.
    theta_label_positions
        Indicate angular positions for the labels for theta along its
        circular axis.
    phi_axis_colour
        Colour for the phi axis.
    theta_axis_colour
        Colour for the theta axis..
    hide_cartesian_axes
        Indicate whether to hide the Cartesian axes, by default True.
    hide_cartesian_axis_labels
        Indicate whether to hide the Cartesian axis labels, by default
        False. This has no effect if `hide_cartesian_axes` is True.
    hide_cartesian_axis_ticks
        Indicate whether to hide the Cartesian axis ticks, by default True.
    plot_colour_bar
        Indicate whether to plot the colour bar, by default False.
    minimum_value
        Minimum data value. Required if plotting the colour bar.
    maximum_value
        Maximum data value. Required if plotting the colour bar.
    colour_map
        Colour map for the colour bar.
    colour_bar_kwargs
        Keyword arguments for the colour bar.
    norm
        Normaliser to use for the colour bar (if applicable)
    axis_tick_factor
        Multiplicative factor providing the distance from the origin to
        the plotted axes and axis tick labels, based on the sphere radius.
    axis_label_factor
        Multiplicative factor providing the distance from the origin to
        the plotted axis labels, based on the sphere radius.

    Returns
    -------
    mpl_toolkits.mplot3d.axes3d.Axes3D
        The same axes as `ax`, with the new elements added.
    """

    ax.set_proj_type(sphere_projection.value)
    ax.set_aspect("equal")

    bound = limits_factor * radius

    ax.set_xlim(-bound, bound)
    ax.set_ylim(-bound, bound)
    ax.set_zlim(-bound, bound)

    ax.set_title(plot_title)

    # Hide the 3D axis
    if hide_cartesian_axis_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    if not hide_cartesian_axis_labels:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    if hide_cartesian_axes:
        ax.set_axis_off()

    if plot_phi_axis:
        phi_axis_positions = np.linspace(0, np.pi)
        phi_position_theta = np.ones_like(phi_axis_positions) * np.pi / 2

        phi_axis_polar_positions = np.stack(
            [phi_axis_positions, phi_position_theta], axis=-1
        )

        phi_axis_cartesian = convert_spherical_to_cartesian_coordinates(
            phi_axis_polar_positions, radius=axis_label_factor * radius
        )

        ax.plot(
            phi_axis_cartesian[:, 0],
            phi_axis_cartesian[:, 1],
            phi_axis_cartesian[:, 2],
            ":",
            linewidth=0.5,
            color=phi_axis_colour,
        )

    if plot_theta_axis:
        theta_axis_positions = np.linspace(0, 2 * np.pi)
        theta_position_phi = np.ones_like(theta_axis_positions) * np.pi / 2
        theta_axis_polar_positions = np.stack(
            [theta_position_phi, theta_axis_positions], axis=-1
        )

        theta_axis_cartesian = convert_spherical_to_cartesian_coordinates(
            theta_axis_polar_positions, radius=axis_label_factor * radius
        )

        ax.plot(
            theta_axis_cartesian[:, 0],
            theta_axis_cartesian[:, 1],
            theta_axis_cartesian[:, 2],
            ":",
            linewidth=0.5,
            color=theta_axis_colour,
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
            radius=axis_tick_factor * radius,
        )

        phi_label_angles_degrees = np.degrees(phi_label_positions)

        ax.text3D(
            0,
            0,
            1.2 * radius,
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
            radius=axis_tick_factor * radius,
        )

        theta_label_angles_degrees = np.degrees(theta_label_positions)

        ax.text3D(
            0,
            1.2 * radius,
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

    colour_bar: Optional[matplotlib.colorbar.Colorbar] = None

    if plot_colour_bar:
        if norm is None:
            norm = plt.Normalize(vmin=minimum_value, vmax=maximum_value)
        scalar_mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=colour_map)
        # print(f"Colour bar has colour map {colour_map}.")
        if colour_bar_kwargs is None:
            colour_bar_kwargs = {}

        plt.colorbar(mappable=scalar_mappable, ax=ax, **colour_bar_kwargs)

    return ax


def produce_spherical_histogram_plot(
    ax: mpl_toolkits.mplot3d.axes3d.Axes3D,
    sphere_radius: float,
    histogram_data: np.ndarray,
    weight_by_magnitude: bool,
    minimum_value: Optional[float] = None,
    maximum_value: Optional[float] = None,
    colour_map: str = "viridis",
    sphere_projection: SphereProjection = SphereProjection.ORTHOGRAPHIC,
    norm: Optional[plt.Normalize] = None,
    sphere_alpha: float = 1.0,
    **kwargs: Optional[dict[str, Any]],
) -> mpl_toolkits.mplot3d.axes3d.Axes3D:
    """Produce a spherical histogram plot on the provided axes.

    Using the provided axes, produce a spherical plot of provided 2D
    histogram data. This plot has a constant radius and is coloured using
    the provided data using the specified colour map. The data can
    optionally be normalised to fit a new range.

    Parameters
    ----------
    ax
        Matplotlib :class:`Axes3D` on which to plot the 3D spherical
        histogram. The projection of these axes **must** be 3D.
    sphere_radius
        Radius of the sphere to plot.
    histogram_data
        Binned data to plot. This data should have
        the shape ``(n, n)`` where ``n`` represents the half-number of
        histogram bins. We currently assume that the half-number of bins
        is the **same** in both :math:`\\phi` and :math:`\\theta`. This
        function separates the data to plot half of it on the sphere.
    weight_by_magnitude
        Indicate whether plots should be weighted by magnitude or by count.
    minimum_value
        Minimum value for data normalisation. If not specified, the minimum
        of the data is automatically used instead.
    maximum_value
        Maximum value for data normalisation. If not specified, the maximum
        of the data is automatically used instead.
    colour_map
        Name of the Matplotlib colour map to be used for colouring the
        histogram data.
    norm
        Optional :class:`matplotlib.colors.Normalize` object to use to
        normalise the colours.
    sphere_projection
        3D projection method to be used for the spherical figure. Options
        are orthographic and perspective projection.
        See :class:`SphereProjection`.
    sphere_alpha
        Opacity of the sphere.
    **kwargs
        Keyword arguments for the plot labelling.
        See :func:`.produce_labelled_3d_plot` for options.

    Returns
    -------
    mpl_toolkits.mplot3d.axes3d.Axes3D
        A reference to the :class:`Axes3D` object passed in as ``ax``.

    Warnings
    --------
    The provided axes must have the projection set to 3D
    using ``projection="3d"``.

    The histogram data provided must occupy the entire sphere. No
    manipulations will be performed in this function to get the face
    colours to match the number of faces.

    See Also
    --------
    .prepare_two_dimensional_histogram:
        Prepare the 2D histogram data to be plotted on a sphere.
    .produce_labelled_3d_plot:
        Label the axes of the 3D plot.
    .produce_3d_triangle_sphere_plot:
        Similar function for an icosphere.

    Notes
    -----
    The data provided to this function is stored as a square array of shape
    ``(n, n)``. To generate the sphere colouring data, we duplicate the
    data and invert it with respect to the ``phi`` axis. The copying
    provides the values for :math:`\\theta \\in [-180^\\circ, 0^\\circ]`.
    We then invert the array to correct the ``phi`` values.
    """

    # Get the data to plot on the sphere. We must determine if we want
    # it to be magnitude-weighted or count weighted.
    if weight_by_magnitude:
        original_intensity_data = histogram_data[..., MagnitudeType.THREE_DIMENSIONAL]
    else:
        original_intensity_data = histogram_data[..., MagnitudeType.COUNT]

    # cleaned_histogram_data = prepare_two_dimensional_histogram(
    #     binned_data=original_intensity_data
    # )

    cleaned_histogram_data = original_intensity_data

    # half_number_of_bins, number_of_bins = cleaned_histogram_data.shape
    half_number_of_bins, number_of_bins = cleaned_histogram_data.shape

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

    if norm is None:
        norm = plt.Normalize(vmin=minimum_value, vmax=maximum_value)
    else:
        norm.vmax = maximum_value or norm.vmax
        norm.vmin = minimum_value or norm.vmin

    normalised_sphere_intensities = norm(cleaned_histogram_data)
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
        shade=False,
    )
    # surface.set_edgecolor("white")
    # surface.set_linewidth(0.25)

    ax = produce_labelled_3d_plot(ax=ax, radius=sphere_radius, norm=norm, **kwargs)

    ax.set_aspect("equal")

    return ax


def produce_polar_histogram_plot(
    ax: matplotlib.projections.polar.PolarAxes,
    data: np.ndarray,
    bins: np.ndarray,
    zero_position: CardinalDirection = CardinalDirection.NORTH,
    rotation_direction: RotationDirection = RotationDirection.CLOCKWISE,
    plot_title: Optional[str] = None,
    label_axis: bool = True,
    axis_ticks: np.ndarray = np.arange(0, 360, 30),
    axis_ticks_unit: AngularUnits = AngularUnits.DEGREES,
    colour: str = "blue",
    mirror_histogram: bool = True,
) -> matplotlib.projections.polar.PolarAxes:
    """Produce 1D polar histogram.

    Produce a 1D polar histogram using the specified data on the
    provided axes.


    Parameters
    ----------
    ax
        Matplotlib :class:`matplotlib.projections.polar.PolarAxes` on which
        to plot the data.

    data
        Data to plot. This should have the same size as ``bins``.

    bins
        *Lower value* of each bin in the histogram.

    zero_position
        Zero-position on the polar axes, expressed as a member of the
        enumerated class :class:`CardinalDirection`.

    rotation_direction
        Rotation direction indicating how the bin values should
        increase from the zero-point specified in ``zero_position``,
        represented as a member of :class:`RotationDirection`.

    plot_title
        Optional title of the plot.

    label_axis
        Indicate whether the circumferential axis should be labelled.

    axis_ticks
        Axis ticks for the histogram. Units specified in
        ``axis_ticks_unit``.

    axis_ticks_unit
        :class:`AngularUnits` indicating what unit should be used for
        specifying the axis ticks. Default is :attr:`AngularUnits.DEGREES`.

    colour
        Colour used for the histogram bars. Must be a valid matplotlib
        colour [#f1]_.

    mirror_histogram
        Indicate whether the histogram should be mirrored to plot data on
        the complete circle.

    Returns
    -------

    matplotlib.projections.polar.PolarAxes
        The ``PolarAxes`` used for plotting.

    Warnings
    --------
    The axes provided **must** be created using ``projection="Polar"``.

    See Also
    --------
    matplotlib.projections.polar.PolarAxes:
        Polar axes used for plotting the polar histogram.

    References
    ----------
    .. [#f1] https://matplotlib.org/stable/users/explain/colors/colors.html
    """

    bin_width = bins[1] - bins[0]
    ax.set_theta_direction(rotation_direction.value)
    ax.set_theta_zero_location(zero_position.value)
    ax.set_title(plot_title)
    ax.axes.yaxis.set_ticklabels([])

    # Prepare the data for plotting, mirroring if necessary
    if mirror_histogram:
        # Duplicate the bins and the data.
        mirrored_bins = bins.copy()
        mirrored_data = data.copy()

        # Offset the mirrored bins and flip the signs
        mirrored_bins = -(mirrored_bins + bin_width)

        # Flip the mirrored bins, but NOT the data
        mirrored_bins = np.flip(mirrored_bins)

        # Tack the bins and values onto the respective arrays
        bins = np.concatenate([mirrored_bins, bins])
        data = np.concatenate([mirrored_data, data])

    if label_axis:
        # start, end = ax.get_xlim()

        if axis_ticks_unit is AngularUnits.DEGREES:
            axis_ticks = np.radians(axis_ticks)
            # axis_ticks_increment = np.radians(axis_ticks_increment)

        ax.xaxis.set_ticks(axis_ticks)
    else:
        ax.xaxis.set_ticks([])

    ax.bar(bins, data, align="edge", width=bin_width, color=colour)

    return ax


def produce_polar_histogram_plot_from_2d_bins(
    ax: matplotlib.projections.polar.PolarAxes,
    angle: AngularIndex,
    data: np.ndarray,
    bins: np.ndarray,
    bin_angle_unit: AngularUnits = AngularUnits.DEGREES,
    weight_by_magnitude: bool = True,
    **kwargs: Dict[str, Any],
) -> matplotlib.projections.polar.PolarAxes:
    """Produce polar histogram from 2D binned data

    Produce a polar histogram for the specified angle, starting from the 2D
    histogram data. This function takes in existing Matplotlib axes and
    plots the histogram on them.

    Parameters
    ----------
    ax
        Matplotlib polar axes on which the histogram will be plotted.

    angle
        Indicate which angle to extract from the data. See
        :class:``AngularIndex`` for details.

    data
        2D histogram binned data of shape ``(n, n, 3)`` where
        ``n`` is the half-number of bins used in the binning process.

    bins
        Bounds of the bins used to construct the histograms. If
        the array is 2D, then the angular indexing will be used to
        extract the phi bin boundaries. If ``n + 1`` entries are
        present, the last value is removed to ensure that only the lower
        bound of each bin is kept.

    bin_angle_unit
        Unit for the bin angles. See :class:`AngularUnits`.

    weight_by_magnitude
        Indicate whether the histograms should be weighted by count or by
        magnitude. If magnitude is used, the phi histogram is weighted by
        3D magnitude while the theta histogram is weighted by the projected
        magnitude in the XY plane.

    **kwargs
        Keyword arguments for the plotting. See
        :func:`produce_polar_histogram_plot` for more details.

    Returns
    -------
    matplotlib.projections.polar.PolarAxes
        A reference to the axes ``ax`` used for plotting.

    Warnings
    --------
    The axes passed in ``ax`` must be of type
    :class:`matplotlib.projections.polar.PolarAxes`. To create these axes,
    ensure to set ``projection="Polar"``.

    See Also
    --------
    produce_polar_histogram_plot:
        Produce a polar plot from 1D data. See documentation for the
        keyword arguments that can be passed to this function.
    """

    # Compute the 1D histograms from the binned data
    one_dimensional_histograms = produce_phi_theta_1d_histogram_data(
        data, weight_by_magnitude=weight_by_magnitude
    )

    # Select the relevant 1D histogram
    selected_histogram_data = one_dimensional_histograms[angle]

    if bin_angle_unit == AngularUnits.DEGREES:
        bins = np.radians(bins)

    data_shape = selected_histogram_data.shape
    bin_shape = bins.shape

    if len(bin_shape) == 2:
        bins = bins[angle]

    if len(bins) > data_shape[0]:
        bins = bins[:-1]

    ax = produce_polar_histogram_plot(
        ax=ax, data=selected_histogram_data, bins=bins, **kwargs
    )

    return ax


def produce_planar_2d_histogram_plot(
    ax: plt.Axes,
    data: np.ndarray,
    bins: Tuple[np.ndarray, np.ndarray],
    # bin_angle_unit: AngularUnits = AngularUnits.DEGREES,
    weight_by_magnitude: bool = True,
    colour_map: str = "viridis",
    show_axes: bool = True,
    phi_axis_ticks: np.ndarray = np.arange(0, 181, 30),
    theta_axis_ticks: np.ndarray = np.arange(0, 361, 30),
    norm: Optional[matplotlib.colors.Normalize] = None,
    show_colour_bar: bool = False,
    colour_bar_position: str = "right",
    # axis_ticks_units: AngularUnits = AngularUnits.DEGREES,
    plot_title: Optional[str] = None,
) -> plt.Axes:
    """
    Produce a 2D planar plot of a flattened phi, theta histogram.

    Produce a 2D histogram of the phi, theta

    Parameters
    ----------
    ax
        Axes on which to plot the 2D histogram.
    data
        Histogram data to plot. This array should consist of 2D sheets,
        with the last axis determining magnitude vs. count.
    bins
        Histogram bins for phi and theta, respectively.
    bin_angle_unit
        Angular unit for the provided bin boundaries.
    weight_by_magnitude
        Indicate whether the histogram should be weighted by magnitude.
    colour_map
        Name of the colour map to use to visualise the histogram.
    show_axes
        Indicate whether to show the axes.
    phi_axis_ticks
        Axis ticks along the vertical phi axis.
    theta_axis_ticks
        Axis ticks along the horizontal theta axis.
    norm
        Normaliser to change how the data are plotted.
    show_colour_bar
        Indicate whether to show a colour bar.
    colour_bar_position
        Position of the colour bar with respect to the plot.
    axis_ticks_units
        Angular units for the axis ticks.
    plot_title
        Optional title for the produced plot.

    Returns
    -------
    matplotlib.axes.Axes
        The axes on which the histogram is plotted. This is the same object
        as `ax` in the parameters.
    """

    # First isolate the correct sheet of data
    histogram_data: np.ndarray

    if weight_by_magnitude:
        histogram_data = data[..., MagnitudeType.THREE_DIMENSIONAL]
    else:
        histogram_data = data[..., MagnitudeType.COUNT]

    # Now, we need to be careful with the extent for the image.
    phi_bins, theta_bins = bins

    phi_max = phi_bins.max()
    phi_bin_spacing = phi_bins[1] - phi_bins[0]
    phi_half_bin = phi_bin_spacing / 2

    theta_max = theta_bins.max()
    theta_bin_spacing = theta_bins[1] - theta_bins[0]
    theta_half_bin = theta_bin_spacing / 2

    image_extent = (0, theta_max, phi_max, 0)

    # Plot on the axes
    ax.imshow(
        histogram_data,
        cmap=colour_map,
        norm=norm,
        extent=image_extent,
        interpolation="None",
    )

    # Deal with the axis ticks
    if show_axes:
        ax.set_ylabel(r"$\phi$")
        ax.set_xlabel(r"$\theta$")

        ax.set_yticks(phi_axis_ticks)
        ax.set_xticks(theta_axis_ticks)
    else:
        ax.axis("off")

    if show_colour_bar:
        if norm is None:
            norm = matplotlib.colors.Normalize(
                vmin=histogram_data.min(), vmax=histogram_data.max()
            )
        scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=colour_map)
        plt.colorbar(scalar_mappable, ax=ax, location=colour_bar_position)

    ax.set_title(plot_title)

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
    mirror_polar_plots: bool = True,
    **kwargs: Dict[str, Any],
):
    """Produce and show the vector rose histograms.

    This function takes in 2D binned histogram input and shows a 3-panel
    figure containing (from left to right):

    * The 2D sphere plot of :math:`\\phi,\\theta`.
    * The 1D polar histogram of :math:`\\theta`.
    * The 1D polar histogram of :math:`\\phi`.

    A number of plotting parameters may be modified here. See the
    parameter descriptions for more details, as well as the above plotting
    functions.

    Parameters
    ----------
    binned_data
        The binned histogram data for the :math:`\\phi,\\theta` plane. This
        NumPy array should have size ``(n, n, 3)`` where ``n`` is the
        half-number of histogram bins used to construct the histogram.
        See :func:`.create_binned_orientation` for more details. See
        :class:`MagnitudeType` for the indexing rules along the last axis.

    bins
        The boundaries of the bins. This array should be of size
        ``(2, n + 1)`` where ``n`` represents the half-number of bins.
        See :class:`AngularIndex` for the indexing rules along the first
        axis.

    sphere_radius
        The radius of the sphere used for 3D plotting.

    zero_position_2d
        The :class:`CardinalDirection` where zero should be placed in the
        1D polar histograms (default: North).

    rotation_direction
        The :class:`RotationDirection` of increasing angles in the 1D polar
        histograms (default: clockwise).

    use_degrees
        Indicate whether the values are in degrees. If ``True``, values are
        assumed to be in degrees. Otherwise, radians are assumed.

    colour_map
        Name of the matplotlib colour map [#f2]_ to use to colour
        the hemisphere. If an invalid name is provided, a default
        greyscale colour map ("gray") will be used.

    plot_title
        Title of the overall plot (optional).

    weight_by_magnitude
        Indicate whether plots should be weighted by magnitude or by count.

    mirror_polar_plots
        Indicate whether the polar histogram data should be mirrored to
        fill the complete plot.

    **kwargs
        Extra keyword arguments for the sphere plotting. See
        :func:`produce_spherical_histogram_plot` for available arguments.


    See Also
    --------
    produce_spherical_histogram_plot:
        Create the spherical plot in isolation.

    produce_polar_histogram_plot:
        Create 1D polar histograms in isolation from 1D histogram data.

    .create_binned_orientation:
        Bin vectors into a 2D :math:`\\phi,\\theta` histogram. The return
        values from that function may be passed as arguments to this
        function.


    References
    ----------
    .. [#f2] https://matplotlib.org/stable/users/explain/colors/colormaps.html
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

    ax, _ = produce_spherical_histogram_plot(
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
        mirror_histogram=mirror_polar_plots,
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
        mirror_histogram=mirror_polar_plots,
    )

    # Show the plots
    plt.suptitle(plot_title, fontweight="bold", fontsize=14)
    plt.subplots_adjust(left=0.05, right=0.95, wspace=0.25)
    plt.show()


def __update_sphere_viewing_angle(
    frame: int,
    sphere_axes: mpl_toolkits.mplot3d.axes3d.Axes3D,
    angle_increment: int,
) -> Iterable[mpl_toolkits.mplot3d.axes3d.Axes3D]:
    """Update the sphere viewing angle.

    Updates the sphere viewing angle to be the current angle, with
    the azimuth increased by an increment.

    Parameters
    ----------
    frame
        The number of the current frame in the animation (required
        to fit the animation signature, but unused here).

    Returns
    -------
    Iterable[mpl_toolkits.mplot3d.axes3d.Axes3D]
        An iterable containing a reference to the 3D sphere axes.
    """

    # Get the information about the current viewing angle
    elev = sphere_axes.elev
    azim = sphere_axes.azim
    roll = sphere_axes.roll

    # Increment the azim
    azim += angle_increment

    # Set the new values
    sphere_axes.view_init(elev=elev, azim=azim, roll=roll)

    return [sphere_axes]


def animate_sphere_plot(
    sphere_figure: matplotlib.figure.Figure,
    sphere_axes: mpl_toolkits.mplot3d.axes3d.Axes3D,
    rotation_direction: RotationDirection = RotationDirection.CLOCKWISE,
    angle_increment: int = 10,
    animation_delay: int = 250,
    reset_initial_orientation: bool = True,
) -> matplotlib.animation.FuncAnimation:
    """Animate the sphere plot.

    Create an animation of the sphere plot rotating about its central
    axis (i.e., the axis running from the sphere's north pole to its
    south pole).

    Parameters
    ----------
    sphere_figure
        The :class:`matplotlib.figure.Figure` containing the sphere
        plot.

    sphere_axes
        The :class:`mpl_toolkits.mplot3d.axes3d.Axes3D` containing the
        sphere plot. These axes **must** be 3D axes.

    rotation_direction
        Direction for the **sphere** to rotate.
        See :class:`RotationDirection` for more information.

    angle_increment
        Increment of the angle in **degrees** for the rotation at each
        frame. This value should be positive.

    animation_delay
        Time delay between frames in milliseconds.

    reset_initial_orientation
        Indicate whether the sphere should be reset to its original
        orientation before recording the animation. This argument should
        be set to ``False`` to allow a custom starting position.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The matplotlib animation produced by the sphere rotation.

    Warnings
    --------
    We recommend to hide the polar axis ticks while performing the
    animation. Otherwise, the result may look odd.

    See Also
    --------
    matplotlib.animation.FuncAnimation:
        The class that serves as the basis for the animations created
        here.

    mpl_toolkits.mplot3d.axes3d.Axes3D.view_init:
        The method used to update the 3D viewing angle to produce the
        animations.

    """

    # Determine the sign of the angle increment
    angle_increment = -rotation_direction.value * abs(angle_increment)

    # Create the function that will update the frame.
    update_angle_func = functools.partial(
        __update_sphere_viewing_angle,
        sphere_axes=sphere_axes,
        angle_increment=angle_increment,
    )

    # Check if we need to reset the orientation
    if reset_initial_orientation:
        sphere_axes.view_init()

    # Get the number of frames necessary to do a full 360° rotation
    number_of_frames = np.abs(np.ceil(360 / angle_increment)).astype(int)

    # Create the animation
    animation = matplotlib.animation.FuncAnimation(
        fig=sphere_figure,
        func=update_angle_func,
        frames=number_of_frames,
        interval=animation_delay,
    )

    return animation


def produce_3d_triangle_sphere_plot(
    ax: mpl_toolkits.mplot3d.axes3d.Axes3D,
    sphere: trimesh.primitives.Sphere,
    face_counts: np.ndarray,
    colour_map: str = "viridis",
    sphere_alpha: float = 1.0,
    norm: Optional[plt.Normalize] = None,
    **kwargs: Optional[dict[str, Any]],
) -> mpl_toolkits.mplot3d.axes3d.Axes3D:
    """Produce a 3D sphere plot based on a triangle mesh.

    Using the provided axes, plot a sphere with face colours corresponding
    to the provided values. This sphere has constant radius.

    Parameters
    ----------
    ax
        Axes on which to plot the sphere.
    sphere
        Mesh of type :class:`trimesh.primitives.Sphere` to plot.
    face_counts
        Values assigned to each face in the `sphere`.
    colour_map
        Colour map used to colour the sphere, by default "viridis".
    norm
        Optional :class:`matplotlib.colors.Normalize` object to use to
        normalise the colours.
    sphere_alpha
        Opacity of the sphere.
    **kwargs
        Keyword arguments for the plot labelling.
        See :func:`.produce_labelled_3d_plot` for options.


    Returns
    -------
    mpl_toolkits.mplot3d.axes3d.Axes3D
        The axes on which the provided sphere is plotted.

    Warnings
    --------
    The provided axes must have the projection set to 3D
    using ``projection="3d"``.

    The histogram data provided must occupy the entire sphere. No
    manipulations will be performed in this function to get the face
    colours to match the number of faces.

    See Also
    --------
    vectorose.triplot.run_spherical_histogram_pipeline:
        Produce a sphere and histogram labellings to pass to this function.
    .produce_labelled_3d_plot:
        Label the axes of the 3D plot.
    .produce_spherical_histogram_plot:
        Similar function for a UV sphere.
    """

    # Get the face colours
    if norm is None:
        norm = matplotlib.colors.Normalize(
            vmin=face_counts.min(), vmax=face_counts.max()
        )
    else:
        norm.autoscale(face_counts)

    scalar_mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=colour_map)

    face_colours = scalar_mapper.to_rgba(face_counts)

    # Now, prepare the sphere for plotting
    vertices: np.ndarray = sphere.vertices
    x_coordinates = vertices[:, 0]
    y_coordinates = vertices[:, 1]
    z_coordinates = vertices[:, 2]

    triangles = sphere.faces

    # Plot the sphere
    ax.plot_trisurf(
        x_coordinates,
        y_coordinates,
        z_coordinates,
        triangles=triangles,
        facecolor=face_colours,
        alpha=sphere_alpha,
        shade=False,
    )

    # Now, configure the axes
    sphere_bounds = sphere.bounds
    min_location = sphere_bounds.min()
    max_location = sphere_bounds.max()

    sphere_radius = (max_location - min_location) / 2

    # print(f"Sphere has radius {sphere_radius}...")

    kwargs["radius"] = sphere_radius

    ax = produce_labelled_3d_plot(ax=ax, norm=norm, colour_map=colour_map, **kwargs)

    ax.set_aspect("equal")

    return ax


def produce_3d_tregenza_sphere_plot(
    ax: mpl_toolkits.mplot3d.axes3d.Axes3D,
    tregenza_sphere: TregenzaSphereBase,
    histogram_data: List[np.ndarray],
    sphere_alpha: float = 1.0,
    colour_map: str = "viridis",
    norm: Optional[plt.Normalize] = None,
    correct_area_weighting: bool = True,
    **kwargs: Optional[dict[str, Any]],
) -> mpl_toolkits.mplot3d.axes3d.Axes3D:
    """Produce a 3D sphere plot based on a Tregenza sphere.

    Using the provided axes, plot a Tregenza sphere with face colours
    corresponding to the provided histogram values.

    Parameters
    ----------
    ax
        Axes on which to plot the sphere.
    tregenza_sphere
        Tregenza sphere on which to plot the values.
    histogram_data
        Histogram data to plot. The length of this list must correspond to
        the number of rings in the `tregenza_sphere` and the length of each
        entry must correspond to the respective patch count.
    sphere_alpha
        Opacity of the sphere.
    colour_map
        Colour map to use when plotting the sphere, by default "viridis".
    norm
        Optional :class:`matplotlib.colors.Normalize` object to use to
        normalise the colours.
    correct_area_weighting
        Indicate whether to correct for area weighting in the face colours.
    **kwargs
        Keyword arguments for the plot labelling.
        See :func:`.produce_labelled_3d_plot` for options.

    Returns
    -------
    mpl_toolkits.mplot3d.axes3d.Axes3D
        The axes on which the provided sphere is plotted.

    Warnings
    --------
    The histogram data must have a size matching the provided Tregenza
    sphere.

    """

    # Get the plot on the axes
    flattened_histogram_data = np.concatenate(histogram_data)

    if norm is None:
        norm = matplotlib.colors.Normalize()

    norm.autoscale(flattened_histogram_data)

    ax = tregenza_sphere.create_tregenza_plot(
        ax=ax,
        face_data=histogram_data,
        cmap=colour_map,
        norm=norm,
        sphere_alpha=sphere_alpha,
    )

    # Define the sphere radius
    sphere_radius = 1

    kwargs["radius"] = sphere_radius

    # Add the labels to the plot
    ax = produce_labelled_3d_plot(ax=ax, norm=norm, colour_map=colour_map, **kwargs)

    ax.set_aspect("equal")

    return ax


def construct_confidence_cone(
    angular_radius: float,
    number_of_patches: int = 80,
    mean_orientation: Optional[np.ndarray] = None,
    two_sided_cone: bool = True,
    **kwargs
) -> List[mpl_toolkits.mplot3d.art3d.Poly3DCollection]:
    """Construct the patches for a confidence cone.

    Construct the triangular patches for a confidence cone with a specified
    angular radius, and optionally rotated to a specified mean direction.

    Parameters
    ----------
    angular_radius
        Angular radius for the confidence cone bounds in radians.
    number_of_patches
        Number of patches to construct. Increase for a better approximation
        to a cone.
    mean_orientation
        Mean orientation to rotate the confidence cone, in cartesian
        coordinates. If `None`, then the cone is not rotated and remains
        vertically oriented.
    two_sided_cone
        Indicate whether the cone should be two-sided. If `True`, two cones
        will be constructed, radiating from the centre. If `False`, then a
        single cone is created.
    **kwargs
        Keyword arguments for the patch construction.
        See :class:`Poly3DCollection` for details.

    Returns
    -------
    list[mpl_toolkits.mplot3d.art3d.Poly3DCollection]
        List of :class:`Poly3DCollection` representing each patch of the
        confidence cone. These patches are triangular.
    """

    # Create a list of patches
    patches: List[mpl_toolkits.mplot3d.art3d.Poly3DCollection] = []

    # Construct the rotation matrix
    if mean_orientation is not None:
        mean_orientation_spherical = compute_vector_orientation_angles(
            vectors=mean_orientation[None, :]
        )[0]

        mean_phi = mean_orientation_spherical[AngularIndex.PHI]
        mean_theta = mean_orientation_spherical[AngularIndex.THETA]

        rotation = Rotation.from_euler("xz", [-mean_phi, -mean_theta])
    else:
        rotation = None

    angular_increment = 2 * np.pi / number_of_patches

    origin = np.zeros((1, 3))
    start_vertex = np.array([angular_radius, 0])
    increment_array = np.array([0, angular_increment])

    for i in range(number_of_patches):
        end_vertex = start_vertex + increment_array

        ring_vertices = np.stack([start_vertex, end_vertex], axis=0)

        ring_vertices_cartesian = convert_spherical_to_cartesian_coordinates(
            ring_vertices
        )

        patch_vertices = np.concatenate([ring_vertices_cartesian, origin], axis=0)

        if rotation is not None:
            patch_vertices = rotation.apply(patch_vertices)

        patch = mpl_toolkits.mplot3d.art3d.Poly3DCollection([patch_vertices], **kwargs)

        patches.append(patch)

        if two_sided_cone:
            inverse_vertices = -patch_vertices
            inverse_patch = mpl_toolkits.mplot3d.art3d.Poly3DCollection(
                [inverse_vertices], **kwargs
            )
            patches.append(inverse_patch)

        start_vertex = end_vertex

    return patches


def construct_uv_sphere(
    phi_steps: int = 80,
    theta_steps: int = 160,
    radius: float = 1
) -> np.ndarray:
    """Construct a sphere with rectangular faces.

    Construct a UV sphere where each ring has the same number of faces.

    Parameters
    ----------
    phi_steps
        Number of faces along the phi axis.
    theta_steps
        Number of faces along the theta axis, within a ring.
    radius
        Sphere radius.

    Returns
    -------
    numpy.ndarray
        Array containing the Cartesian coordinates of the sphere vertices
        in a format to plot using :meth:`Axes3D.plt_surface`. This array
        will have shape ``(phi_steps + 1, theta_steps + 1, 3)`` where the
        last axis corresponds to the ``X, Y, Z`` components.

    Warnings
    --------
    This sphere should not be used to plot histograms. It is provided for
    visualisations that do not involve plotting data on the surface of the
    sphere.

    Notes
    -----
    The coordinates computed using this function can easily be used to plot
    a sphere using :meth:`Axes3D.plt_surface`. To do so, the X, Y and Z
    coordinate sheets must be separated by indexing along the last axis.

    """

    # Compute the phi and theta values
    phi = np.linspace(start=0, stop=np.pi, num=phi_steps + 1, endpoint=True)
    theta = np.linspace(start=0, stop=2*np.pi, num=theta_steps + 1, endpoint=True)

    # Now, build the 2D spherical coordinates
    phi_grid, theta_grid = np.meshgrid(phi, theta)

    # Convert to Cartesian coordinates
    sphere_angles = np.stack([phi_grid, theta_grid], axis=-1)

    sphere_cartesian_coordinates = convert_spherical_to_cartesian_coordinates(
        angular_coordinates=sphere_angles,
        radius=radius
    )

    return sphere_cartesian_coordinates


def produce_3d_confidence_cone_plot(
    ax: mpl_toolkits.mplot3d.axes3d.Axes3D,
    confidence_cone_patches: List[mpl_toolkits.mplot3d.art3d.Poly3DCollection],
    sphere_vertices: np.ndarray,
    sphere_radius: float = 1,
    sphere_alpha: float = 0.5,
    sphere_colour: str = "#a8a8a8",
    **kwargs
) -> mpl_toolkits.mplot3d.axes3d.Axes3D:
    """Produce a 3D confidence cone plot.

    Using the provided confidence cone patches and sphere vertices, create
    a plot containing the confidence cone inside a sphere.

    Parameters
    ----------
    ax
        Axes on which to plot. These must be 3D axes.
    confidence_cone_patches
        Patches for the confidence cone.
    sphere_vertices
        Vertices for the UV sphere.
    sphere_radius
        Radius of the sphere.
    sphere_alpha
        Sphere opacity level.
    sphere_colour
        Colour of the sphere.
    **kwargs
        Arguments passed to :func:`produce_labelled_3d_plot` to alter the
        labelling of the 3D axes.

    Returns
    -------
    mpl_toolkits.mplot3d.axes3d.Axes3D
        Axes on which the confidence cone has been plotted.

    Warnings
    --------
    The provided axes must have been constructed using a 3D projection, by
    setting ``projection="3d"``.

    See Also
    --------
    .construct_confidence_cone : Generate the confidence cone patches.
    .construct_uv_sphere : Generate vertices for a quad-based sphere.
    """

    # Plot the confidence cone
    for patch in confidence_cone_patches:
        ax.add_collection3d(patch)

    # Plot the sphere
    xs = sphere_vertices[..., 0]
    ys = sphere_vertices[..., 1]
    zs = sphere_vertices[..., 2]
    ax.plot_surface(xs, ys, zs, color=sphere_colour, alpha=sphere_alpha)

    # Label the plot
    ax = produce_labelled_3d_plot(ax=ax, radius=sphere_radius, **kwargs)

    ax.set_aspect("equal")

    return ax
