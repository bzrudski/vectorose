# Copyright (c) 2023-, Benjamin Rudski, Joseph Deering
#
# This code is licensed under the MIT License. See the `LICENSE` file for
# more details about copying.

"""Plotting functions for VectoRose.

After constructing the various histograms using the classes and functions
present in :mod:`.tregenza_sphere`, :mod:`.triangle_sphere` and
:mod:`.polar_data`, this module can be used to visualise the results.
"""

import enum
import functools
from typing import Any, Iterable, List, Optional

import imageio_ffmpeg
import matplotlib.animation
import matplotlib.cm
import matplotlib.colors
import matplotlib.figure
import matplotlib.projections
import matplotlib.pyplot as plt
import matplotlib.ticker
import mpl_toolkits.mplot3d.art3d
import mpl_toolkits.mplot3d.axes3d
import numpy as np
import pandas as pd
import vtk
import pyvista as pv
from scipy.spatial.transform import Rotation

from .triangle_sphere import TriangleSphere
from . import util
from .tregenza_sphere import TregenzaSphere

# Configure the SVG export, per https://stackoverflow.com/a/35734729
plt.rcParams["svg.fonttype"] = "none"

# Configure the ffmpeg for export, per
# https://stackoverflow.com/questions/13316397#comment115906431_44483126
plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()


# Configure PyVista options
pv.global_theme.font.fmt = "%.6g"
pv.global_theme.colorbar_vertical.position_x = 0.85
pv.global_theme.colorbar_vertical.position_y = 0.3
pv.global_theme.colorbar_orientation = "vertical"


class CardinalDirection(str, enum.Enum):
    """Cardinal directions.

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
    """Rotation directions.

    This integer-based enumerated type represents two-dimensional rotation
    direction. The convention used is consistent with the Matplotlib
    documentation (see
    :meth:matplotlib.projections.polar.PolarAxes.set_theta_direction for
    details).

    Members
    -------
    CLOCKWISE
        Clockwise, or rightward rotation.

    COUNTER_CLOCKWISE
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
    """Angular units.

    This enumerated type represents angular units (degrees or radians).
    It **does not** provide any implementation for converting from one
    to the other, as this functionality is already very well included
    in NumPy.

    Members
    -------
    DEGREES
        Represent angles in degrees (typically in the range 0 to 360 or
        -180 to +180).
    RADIANS
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


class ViewingPlanes(str, enum.Enum):
    """Built-in viewing angles.

    Each member represents a viewing plane (or the isometric angle) defined
    in PyVista.

    See Also
    --------
    pyvista.Plotter.camera_position :
        The function used to set the view to one of these planes.
    """

    XY = "xy"
    XZ = "xz"
    YZ = "yz"
    YX = "yx"
    ZX = "zx"
    ZY = "zy"
    ISO = "iso"


class SpherePlotter:
    """Produce beautiful, fast 3D sphere plots using PyVista."""

    _sphere_meshes: List[pv.PolyData]
    """The meshes representing individual shells."""

    _largest_radius: float
    """The largest radius of a plotted sphere."""

    _active_shell: int
    """The index of the currently-active shell."""

    _visible_shells: List[int]
    """The indicies of the shells to plot as visible."""

    _active_shell_opacity: float
    """The opacity of the currently active shell."""

    _inactive_shell_opacity: float
    """The opacity of the inactive shells."""

    _plotter: Optional[pv.Plotter]
    """Plotter to use to visualise the spheres."""

    _sphere_actors: List[pv.Actor]
    """The actors representing the plotted spheres."""

    _phi_axis_actor: Optional[pv.Actor]
    """The actor representing the semicircular phi axis."""

    _theta_axis_actor: Optional[pv.Actor]
    """The actor representing the circular theta axis."""

    _point_label_actor: Optional[vtk.vtkActor2D]
    """The actor controlling the point labels."""

    cmap: str
    """The colour map to use when visualising the data."""

    _has_movie_open: bool
    """Indicate whether a movie is currently a being manually written."""

    @property
    def sphere_meshes(self) -> List[pv.PolyData]:
        """Access the wrapped sphere meshes."""
        return self._sphere_meshes

    @property
    def phi_axis_visible(self) -> bool:
        """Indicate whether the phi axis is visible."""
        if self._phi_axis_actor is None:
            return False
        return self._phi_axis_actor.visibility

    @property
    def theta_axis_visible(self) -> bool:
        """Indicate whether the theta axis is visible."""
        if self._theta_axis_actor is None:
            return False
        return self._theta_axis_actor.visibility

    @property
    def axis_labels_visible(self) -> bool:
        """Indicate whether the axis labels are visible."""
        if self._point_label_actor is None:
            return False
        return bool(self._point_label_actor.GetVisibility())

    @property
    def sliders_visible(self) -> bool:
        """Indicate whether the sliders are visible.

        If the plot was created without sliders, this always evaluates to
        `False`.
        """

        sliders = self._plotter.slider_widgets

        if len(sliders) == 0:
            return False

        return all(slider.GetEnabled() for slider in sliders)

    @property
    def scalar_bars_visible(self) -> bool:
        """Indicate whether the scalar bars are visible.

        If there are no scalar bars, this will always return `False`.
        """

        scalar_bars = self._plotter.scalar_bars

        if len(scalar_bars) == 0:
            return False

        return all(scalar_bars[k].GetVisibility() for k in scalar_bars.keys())

    @property
    def active_shell(self) -> int:
        """Get or set the active shell index.

        Warnings
        --------
        This property describes the **index**, not the shell number. The
        values provided here are offset by one compared with the slider
        values.
        """
        return self._active_shell

    @active_shell.setter
    def active_shell(self, index: int):
        self._update_active_sphere(index + 1)

    @property
    def active_shell_opacity(self) -> float:
        """Get the active shell opacity.

        Active sphere opacity between 0 (transparent) and 1 (opaque).
        """
        return self._active_shell_opacity

    @active_shell_opacity.setter
    def active_shell_opacity(self, opacity: float):
        self._update_active_sphere_opacity(opacity)

    @property
    def inactive_shell_opacity(self) -> float:
        """Get the opacity of the inactive shells.

        Inactive sphere opacity between 0 (transparent) and 1 (opaque).
        """
        return self._inactive_shell_opacity

    @inactive_shell_opacity.setter
    def inactive_shell_opacity(self, opacity: float):
        self._update_inactive_sphere_opacity(opacity)

    @property
    def radius(self) -> float:
        """Access the sphere radius."""
        return self._largest_radius

    @property
    def has_produced_plot(self) -> bool:
        """Indicate whether the plot has been produced."""
        return len(self._sphere_actors) > 0

    @property
    def current_phi(self) -> float:
        """Get the phi value under the current view in degrees.

        Notes
        -----
        This value makes the most sense if the camera has the origin as the
        focal point.
        """
        spherical_coordinates = self._get_spherical_coordinates_from_camera()

        phi = spherical_coordinates[util.AngularIndex.PHI]

        return phi

    @property
    def current_theta(self) -> float:
        """Get the theta value under the current view in degrees.

        Notes
        -----
        This value makes the most sense if the camera has the origin as the
        focal point.
        """
        spherical_coordinates = self._get_spherical_coordinates_from_camera()

        theta = spherical_coordinates[util.AngularIndex.THETA]

        return theta

    @property
    def has_movie_open(self) -> bool:
        """Indicate whether a movie is currently being manually written."""
        return self._has_movie_open

    def __init__(
        self,
        sphere_meshes: List[pv.PolyData] | pv.PolyData,
        visible_shells: Optional[List[int]] = None,
        off_screen: bool = None,
        cmap: str = "viridis",
    ):
        # If only a single mesh has been passed it, wrap in a list.
        if isinstance(sphere_meshes, pv.PolyData):
            sphere_meshes = [sphere_meshes]

        self._sphere_meshes = sphere_meshes
        self._sphere_actors = []

        self._phi_axis_actor = None
        self._theta_axis_actor = None
        self._point_label_actor = None
        self.cmap = cmap

        # Determine the visible shells
        self._visible_shells = visible_shells or np.arange(len(sphere_meshes)).tolist()

        # Set the active shell
        self._active_shell = self._visible_shells[-1]
        self._active_shell_opacity = 1
        self._inactive_shell_opacity = 0

        # Compute the radius
        bounds = np.abs([m.bounds for m in sphere_meshes])
        self._largest_radius = bounds.max()

        # Create the plotter
        self._plotter = pv.Plotter(off_screen=off_screen)

        # We don't have a movie open when creating the plotter
        self._has_movie_open = False

    def _get_spherical_coordinates_from_camera(self) -> np.ndarray:
        """Get the spherical coordinates for the camera location.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(3,)`` containing the phi, theta and radius
            for the camera's position in 3D space.

        Notes
        -----
        The values in this function are not recentred if the camera is
        translated in space.
        """
        camera_position = self._plotter.camera_position
        camera_location = camera_position[0]

        spherical_coordinates = util.compute_spherical_coordinates(
            np.array(camera_location), use_degrees=True
        )

        return spherical_coordinates

    def _update_active_sphere_opacity(self, new_opacity: float):
        """Update the opacity level of the active sphere."""

        self._active_shell_opacity = new_opacity
        actor = self._sphere_actors[self._active_shell]
        actor.prop.opacity = new_opacity

    def _update_inactive_sphere_opacity(self, new_opacity: float):
        """Update the opacity level of the inactive spheres."""

        self._inactive_shell_opacity = new_opacity

        for i, actor in enumerate(self._sphere_actors):
            if i == self._active_shell:
                continue

            actor.prop.opacity = new_opacity

    def _update_active_sphere(self, new_selected_shell_number: float):
        """Update the active sphere number."""
        current_shell = self._active_shell
        new_shell = np.round(new_selected_shell_number).astype(int) - 1

        if new_shell != current_shell:
            self._active_shell = new_shell
            self._update_active_sphere_opacity(self._active_shell_opacity)
            self._update_inactive_sphere_opacity(self._inactive_shell_opacity)

    def _update_show_axes(self, show_axes: bool):
        """Update whether the axes are drawn."""

        if self._phi_axis_actor is not None:
            self._phi_axis_actor.visibility = show_axes

        if self._theta_axis_actor is not None:
            self._theta_axis_actor.visibility = show_axes

        if self._point_label_actor is not None:
            self._point_label_actor.SetVisibility(show_axes)

    # TODO: Create a new class to encapsulate the spherical axes.
    def add_spherical_axes(
        self,
        plot_phi: bool = True,
        plot_theta: bool = True,
        phi_increment: int = 30,
        theta_increment: int = 30,
        axis_distance_fixed: float = 0.2,
        axis_distance_relative: float = 1,
        axis_thickness: float = 0.05,
    ):
        """Add spherical axes to the current plotter.

        Parameters
        ----------
        plot_phi
            Indicate that the phi axis is to be plotted.
        plot_theta
            Indicate that the theta axis is to be plotted.
        phi_increment
            Label increment for the phi values.
        theta_increment
            Label increment for the theta values.
        axis_distance_fixed
            Fixed distance from the outer shell to the axis.
        axis_distance_relative
            Relative distance from the outer shell to the axis.
        axis_thickness
            Absolute thickness of the axes.
        """

        # Clear the axes if they are already present
        self.clear_axes()

        # Create a disc for the theta axis and a circular arc for phi.
        max_radius = self._largest_radius
        inner_radius = axis_distance_relative * max_radius + axis_distance_fixed
        outer_radius = inner_radius + axis_thickness

        if plot_theta:
            theta_axis = pv.Disc(inner=inner_radius, outer=outer_radius, c_res=36)
            self._theta_axis_actor = self._plotter.add_mesh(theta_axis, show_edges=True)
        else:
            self._theta_axis_actor = None

        if plot_phi:
            phi_axis = pv.Disc(
                inner=inner_radius,
                outer=outer_radius,
                c_res=36,
                normal=(0, 1, 0),
            ).clip(normal="-x")

            self._phi_axis_actor = self._plotter.add_mesh(phi_axis, show_edges=True)
        else:
            self._phi_axis_actor = None

        # Create the labels
        angular_label_positions = []
        angular_label_strings = []
        # if plot_phi and plot_theta:
        #     angular_labels.append(np.array([90, 90]))

        if plot_phi:
            phi_label_phi_angles = np.arange(0, 181, phi_increment)
            phi_label_theta_angles = 90 * np.ones_like(phi_label_phi_angles)
            phi_label_angles = np.stack(
                [phi_label_phi_angles, phi_label_theta_angles], axis=-1
            )
            angular_label_positions.append(phi_label_angles)
            phi_labels = [f"{phi}\u00b0" for phi in phi_label_phi_angles]
            angular_label_strings.extend(phi_labels)

        if plot_theta:
            theta_label_theta_angles = np.arange(0, 360, theta_increment)
            theta_label_phi_angles = 90 * np.ones_like(theta_label_theta_angles)
            theta_label_angles = np.stack(
                [theta_label_phi_angles, theta_label_theta_angles], axis=-1
            )
            angular_label_positions.append(theta_label_angles)
            theta_labels = [f"{theta}\u00b0" for theta in theta_label_theta_angles]
            angular_label_strings.extend(theta_labels)

        # Put everything together and remove potential duplicates
        angular_label_positions = np.concatenate(angular_label_positions)
        angular_label_positions, unique_indices = np.unique(
            angular_label_positions, axis=0, return_index=True
        )

        angular_label_strings = np.array(angular_label_strings)
        angular_label_strings = angular_label_strings[unique_indices]

        # Get the Cartesian coordinates
        label_cartesian_coordinates = util.convert_spherical_to_cartesian_coordinates(
            np.radians(angular_label_positions), radius=outer_radius + axis_thickness
        )

        # And now construct the labels
        self._point_label_actor = self._plotter.add_point_labels(
            label_cartesian_coordinates, angular_label_strings, show_points=False
        )

        # And finally, add a button to toggle axis visibility
        self._plotter.add_checkbox_button_widget(
            self._update_show_axes, value=True, position=(10, 5), size=50
        )

    def clear_axes(self):
        """Clear the plotted axes."""

        assert self._plotter is not None

        self._plotter.remove_actor(self._phi_axis_actor)
        self._phi_axis_actor = None

        self._plotter.remove_actor(self._theta_axis_actor)
        self._theta_axis_actor = None

        self._plotter.remove_actor(self._point_label_actor)
        self._point_label_actor = None

    def clear_plotter(self):
        """Clear and remove the current plotter."""
        for sphere_actor in self._sphere_actors:
            self._plotter.remove_actor(sphere_actor)
        self._sphere_actors.clear()
        self.clear_axes()

        self._plotter.clear()

    def produce_plot(
        self,
        add_sliders: bool = True,
        series_name: str = "frequency",
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ):
        """Produce the 3D visual plot for the current spheres.

        Parameters
        ----------
        add_sliders
            Indicate whether the opacity sliders should be added to the
            plotting window.
        series_name
            Name of the scalars to consider.
        min_value
            Optional minimum value for the colour map.
        max_value
            Optional maximum value for the colour map.

        Warnings
        --------
        This function produces the :class:`pyvista.Plotter`. The method
        :meth:`SpherePlotter.show` must be called to view the plot.
        """

        plotter = self._plotter

        # Clean up from previous plots
        plotter.clear()
        self._sphere_actors.clear()

        sphere_meshes = [self._sphere_meshes[i] for i in self._visible_shells]

        if min_value is None or max_value is None:
            # Get the bounds for the plotting
            all_frequencies = np.concatenate(
                [m.cell_data[series_name] for m in sphere_meshes]
            )

            # Remove any NaN values for the clim
            all_frequencies = all_frequencies[~np.isnan(all_frequencies)]

            if min_value is None:
                min_value = all_frequencies.min()

            if max_value is None:
                max_value = all_frequencies.max()

        # Add the sphere actors
        for i, mesh in enumerate(self._sphere_meshes):
            actor: pv.Actor
            actor = plotter.add_mesh(
                mesh,
                clim=[min_value, max_value],
                cmap=self.cmap,
                scalars=series_name,
            )
            actor.visibility = i in self._visible_shells
            self._sphere_actors.append(actor)

        # Add the slider widgets
        if add_sliders:
            number_of_shells = len(self._sphere_meshes)

            plotter.add_slider_widget(
                self._update_active_sphere_opacity,
                [0, 1],
                value=self._active_shell_opacity,
                title="Active shell opacity",
                pointa=(0.4, 0.9),
                pointb=(0.6, 0.9),
                title_height=0.01,
                interaction_event="always",
                style="modern",
            )

            if number_of_shells > 1:
                plotter.add_slider_widget(
                    self._update_active_sphere,
                    [1, number_of_shells],
                    value=self._active_shell + 1,
                    title="Active shell",
                    pointa=(0.1, 0.9),
                    pointb=(0.3, 0.9),
                    title_height=0.01,
                    fmt="%.0f",
                    interaction_event="always",
                    style="modern",
                )

                plotter.add_slider_widget(
                    self._update_inactive_sphere_opacity,
                    [0, 1],
                    value=self._inactive_shell_opacity,
                    title="Inactive shell opacity",
                    pointa=(0.7, 0.9),
                    pointb=(0.9, 0.9),
                    title_height=0.01,
                    interaction_event="always",
                    style="modern",
                )

        plotter.add_axes()
        plotter.add_camera_orientation_widget()
        plotter.enable_parallel_projection()

        self._update_active_sphere(self._active_shell + 1)

    def show(self, *args, **kwargs):
        """Show the plotter window.

        Parameters
        ----------
        *args
            Arguments to pass to the method :meth:`pyvista.Plotter.show`.
        **kwargs
            Keyword arguments to pass to the method
            :meth:`pyvista.Plotter.show`.

        Raises
        ------
        AssertionError
            If the plot has not yet been produced using the method
            :meth:`.produce_plot`.

        See Also
        --------
        pyvista.Plotter.show : Method used to show a PyVista plotter.
        """

        assert self.has_produced_plot

        self._plotter.show(*args, **kwargs)

    def close(self, deep_clean: bool = True):
        """Close the plotting window and clear the associated memory.

        Parameters
        ----------
        deep_clean
            Indicate whether a deep clean of the memory should be
            performed after closing.
        """

        self._plotter.close()

        if deep_clean:
            self._plotter.deep_clean()

    def rotate_to_view(
        self,
        phi: Optional[float] = None,
        theta: Optional[float] = None,
        use_degrees: bool = True,
        zoom: Optional[float] = None,
        focal_depth: Optional[float] = None,
    ):
        """Move the camera to focus on a specific orientation.

        Parameters
        ----------
        phi
            The co-latitude to centre on. If `None`, then the value of phi
            is unchanged.
        theta
            The azimuthal angle to centre on. If `None`, then the value of
            theta is unchanged.
        use_degrees
            Indicate whether the angles should be interpreted in degrees.
        zoom
            Factor to zoom the view as a floating point value. See
            :meth:`pyvista.Camera.zoom` for a full explanation.
        focal_depth
            Distance between the camera and the focal point.

        See Also
        --------
        pyvista.Plotter.camera_position :
            Information about the camera from the plotter, used to compute
            the orientation angles.
        pyvista.Camera.zoom :
            Control the camera zoom.

        Notes
        -----
        The point of focus of the camera remains at the origin. This method
        simply changes the position of the camera in space and changes the
        up direction.
        """

        # Fill in any missing parameters and convert angles, if necessary
        if phi is None:
            phi = self.current_phi
        elif not use_degrees:
            phi = np.degrees(phi)

        if theta is None:
            theta = self.current_theta
        elif not use_degrees:
            theta = np.degrees(theta)

        if focal_depth is None:
            focal_depth = self._plotter.camera.distance

        if zoom is None:
            zoom = 1

        # Now, find the location in space
        camera_location = util.convert_spherical_to_cartesian_coordinates(
            np.array([phi, theta]), focal_depth, use_degrees=True
        )

        # And now for the up vector
        up_vector = np.array([0, -1, 0])

        # Tilt it by phi degrees clockwise over x
        phi_rotation = Rotation.from_euler("x", -phi, degrees=True)
        theta_rotation = Rotation.from_euler("z", -theta, degrees=True)

        compound_rotation = theta_rotation * phi_rotation

        # And now apply it to the up vector
        up_vector = compound_rotation.apply(up_vector)

        # And now for setting the camera position
        camera_position_parameters = [
            tuple(camera_location.tolist()),
            (0, 0, 0),
            tuple(up_vector.tolist()),
        ]

        self._plotter.camera_position = camera_position_parameters
        self._plotter.camera.zoom(zoom)

        if self._plotter.iren.initialized:
            self._plotter.update()

    def open_movie_file(
        self,
        filename: str,
        quality: int = 5,
        fps: int = 24,
    ):
        """Open a movie file to record an animation.

        Parameters
        ----------
        filename
            The destination for the created video. Must end with either
            ``.mp4`` or ``.gif``.
        quality
            Image quality for the video export, between 0 and 10. Ignored
            for exports as GIF.
        fps
            Frame rate, number of frames per second in the exported video.

        Warnings
        --------
        This method only opens the video file. It does not save any of the
        frames. These must be written using the :meth:`.write_frame` method
        and then the file must be closed using :meth:`.close_movie`.

        See Also
        --------
        pyvista.Plotter.open_movie :
            The wrapped function that actually creates the movie file.
        .write_frame :
            Add frames to the open movie.
        .close_movie :
            Close and finalise the current movie.
        """

        if filename.endswith(".gif"):
            self._plotter.open_gif(filename, 0, fps)
        else:
            self._plotter.open_movie(filename, fps, quality)

        self._has_movie_open = True

    def write_frame(self):
        """Add a new frame to the current movie.

        Warnings
        --------
        The current plotter must have a movie file that is open.
        """

        assert self.has_movie_open, "No movie is open."

        self._plotter.write_frame()

    def close_movie(self):
        """Close the movie currently being written.

        Warnings
        --------
        The current plotter must have a movie file that is open.
        """

        assert self.has_movie_open, "No movie is open."

        self._plotter.mwriter.close()
        self._has_movie_open = False

    def produce_rotating_video(
        self,
        filename: str,
        quality: int = 5,
        fps: int = 24,
        zoom_factor: float = 1.0,
        number_of_frames: int = 36,
        vertical_shift: Optional[float] = None,
        hide_sliders: bool = True,
    ):
        """Produce a video orbiting around the sphere.

        Using the current plotter, produce a video orbiting around the
        visible shell. If there is no current plotter, a new one is
        produced.

        Parameters
        ----------
        filename
            Movie export filename, with extension provided.
        quality
            Video export quality, between 0 and 10.
        fps
            Frame rate for the exported video.
        zoom_factor
            Factor to zoom when creating the frames.
        number_of_frames
            Total number of frames to produce in the video. A higher number
            results in a smoother animation, but a larger file and longer
            processing time.
        vertical_shift
            Upward shift of the orbital plane with respect to the ground.
        hide_sliders
            Indicate whether to hide the sliders when producing the video.

        See Also
        --------
        pyvista.Plotter.generate_orbital_path:
            generate the path necessary for the animation.

        References
        ----------
        This code is based on an example from the PyVista documentation,
        found at https://docs.pyvista.org/examples/02-plot/orbit.
        """

        # Hide the sliders if requested
        if hide_sliders:
            self.hide_sliders()

        plotter = self._plotter

        # Open a movie
        self.open_movie_file(filename, quality, fps)

        # Get the vertical shift
        if vertical_shift is None:
            vertical_shift = self._sphere_meshes[self._active_shell].length

        plotter.camera.zoom(zoom_factor)

        # Start by generating the orbital path
        path = plotter.generate_orbital_path(
            n_points=number_of_frames, shift=vertical_shift
        )

        # Perform the orbit!
        plotter.orbit_on_path(path, write_frames=True)

        # Close the plotter's writer
        self.close_movie()

        # Re-show the sliders
        if hide_sliders:
            self.show_sliders()

    def produce_shells_video(
        self,
        filename: str,
        quality: int = 5,
        fps: int = 24,
        zoom_factor: float = 1.0,
        inward_direction: bool = True,
        boomerang: bool = False,
        azimuth: Optional[float] = None,
        elevation: Optional[float] = None,
        hide_sliders: bool = True,
        add_shell_text: bool = False,
    ):
        """Produce a video revealing all shells in a plot.

        Parameters
        ----------
        filename
            Movie export filename, including file extension.
        quality
            Video export quality, between 0 and 10.
        fps
            Frame rate for the exported video.
        zoom_factor
            Factor to zoom when creating the frames.
        inward_direction
            Indicate whether the animation should progress from the
            outermost shell inwards (``True``) or from the innermost shell
            outwards (``False``).
        boomerang
            Indicate whether the animation should have a boomerang effect,
            where it is symmetric in time.
        azimuth
            Camera azimuthal angle in degrees (theta). If not specified,
            the current plotter's angles are used.
        elevation
            Camera elevation angle in degrees (phi). If not specified, the
            current plotter's angles are used.
        hide_sliders
            Indicate whether the sliders should be hidden before producing
            the video.
        add_shell_text
            Indicate whether to show text at the bottom of the screen
            indicating the shell number.
        """

        if hide_sliders:
            self.hide_sliders()

        total_number_of_shells = len(self._sphere_meshes)
        shell_order = np.arange(total_number_of_shells)

        if inward_direction:
            shell_order = np.flip(shell_order)

        if boomerang:
            reverse_shell_order = np.flip(shell_order)
            shell_order = np.concatenate([shell_order, reverse_shell_order])

        # Convert indices to numbers to be consistent with other methods
        shell_order += 1

        plotter = self._plotter

        # Activate the first shell manually
        first_shell = shell_order[0]

        self._update_active_sphere(first_shell)
        self._update_active_sphere_opacity(1.0)
        self._update_inactive_sphere_opacity(0.0)

        # Set the camera properties
        if azimuth is not None:
            plotter.camera.azimuth = azimuth

        if elevation is not None:
            plotter.camera.elevation = elevation

        plotter.camera.zoom(zoom_factor)

        # Open the output file
        self.open_movie_file(filename, quality, fps)

        # Make the movie!
        i: int
        for i in shell_order:
            self._update_active_sphere(i)

            shell_text_actor: Optional[vtk.vtkTextActor]

            if add_shell_text:
                shell_text_actor = self._plotter.add_text(
                    f"Shell {i}", position="lower_edge"
                )
            else:
                shell_text_actor = None

            self.write_frame()

            if add_shell_text:
                self._plotter.remove_actor(shell_text_actor)

        # Close the plotter's writer
        self.close_movie()

        if hide_sliders:
            self.show_sliders()

    def hide_sliders(self):
        """Hide the plotter's slider widgets."""

        for slider in self._plotter.slider_widgets:
            slider.SetEnabled(False)

    def show_sliders(self):
        """Show the plotter's slider widgets."""

        for slider in self._plotter.slider_widgets:
            slider.SetEnabled(True)

    def hide_scalar_bars(self):
        """Hide the scalar bars."""

        for scalar_bar_key in self._plotter.scalar_bars.keys():
            scalar_bar = self._plotter.scalar_bars[scalar_bar_key]
            scalar_bar.VisibilityOff()

    def show_scalar_bars(self):
        """Show the scalar bars."""

        for scalar_bar_key in self._plotter.scalar_bars.keys():
            scalar_bar = self._plotter.scalar_bars[scalar_bar_key]
            scalar_bar.VisibilityOn()

    def show_axes(self):
        """Show the spherical axes."""

        self._update_show_axes(True)

    def hide_axes(self):
        """Hide the spherical axes."""

        self._update_show_axes(False)

    def export_screenshot(
        self,
        filename: str,
        transparent_background: bool = True,
        window_size: Optional[tuple[int, int]] = None,
        scale: Optional[int] = None,
        hide_sliders: bool = True,
        hide_scalar_bar: bool = False,
    ):
        """Export a screenshot from the plotter.

        Parameters
        ----------
        filename
            Output destination for the screenshot, including file
            extension. Must be of type PNG, bitmap, JPEG or TIFF.
        transparent_background
            Indicate whether the background should be transparent.
        window_size
            Desired window size before exporting.
        scale
            Factor by which to scale the window before exporting to
            increase resolution.
        hide_sliders
            Indicate whether to hide sliders before exporting. If no
            sliders have been added to the plot, this option has no effect.
        hide_scalar_bar
            Indicate whether to hide the scalar bar when exporting. The
            scalar bar will be made visible again after exporting.

        See Also
        --------
        pyvista.Plotter.screenshot :
            Function wrapped by this function, which actually produces the
            screenshot. The current function borrows some parameters from
            this function.
        """

        if hide_sliders:
            self.hide_sliders()

        if hide_scalar_bar:
            self.hide_scalar_bars()

        self._plotter.screenshot(
            filename, transparent_background, False, window_size, scale
        )

        if hide_sliders:
            self.show_sliders()

        if hide_scalar_bar:
            self.show_scalar_bars()

    def export_graphic(
        self,
        filename: str,
        title: str,
        raster: bool = True,
        painter: bool = True,
        window_size: Optional[tuple[int, int]] = None,
        scale: Optional[int] = None,
        hide_sliders: bool = True,
        hide_scalar_bar: bool = False,
    ):
        """Export a graphic from the plotter.

        Parameters
        ----------
        filename
            Output destination for the graphic, including file
            extension. Must be of type SVG, PDF, TEX, PS or EPS.
        title
            Name of the graphics (see PyVista documentation).
        raster
            Indicate whether to write the properties as a raster image (see
            PyVista documentation).
        painter
            Indicate whether to perform a certain painting step (see
            PyVista documentation).
        window_size
            Desired window size before exporting.
        scale
            Factor by which to scale the window before exporting to
            increase resolution.
        hide_sliders
            Indicate whether to hide sliders before exporting. If no
            sliders have been added to the plot, this option has no effect.
        hide_scalar_bar
            Indicate whether to hide the scalar bar when exporting. The
            scalar bar will be made visible again after exporting.

        See Also
        --------
        pyvista.Plotter.save_graphic :
            Function wrapped by this function, which actually produces the
            graphic. The current function borrows some parameters from
            this function.
        """

        if hide_sliders:
            self.hide_sliders()

        if hide_scalar_bar:
            self.hide_scalar_bars()

        # Save the old parameters
        old_window_size = self._plotter.window_size
        old_scale = self._plotter.scale

        if window_size is not None:
            self._plotter.window_size = window_size

        if scale is not None:
            self._plotter.scale = scale

        self._plotter.save_graphic(filename, title, raster, painter)

        # Reset the parameters
        self._plotter.window_size = old_window_size
        self._plotter.scale = old_scale

        if hide_sliders:
            self.show_sliders()

        if hide_scalar_bar:
            self.show_scalar_bars()

    def set_view_plane(self, viewing_plane: ViewingPlanes):
        """Set the plotter camera to a predefined viewing angle."""

        self._plotter.camera_position = viewing_plane


def produce_1d_scalar_histogram(
    counts: pd.Series | np.ndarray,
    bin_edges: np.ndarray,
    fill: bool = True,
    ax: Optional[plt.Axes] = None,
    log: bool = False,
    **kwargs,
) -> plt.Axes:
    """Produce a 1D scalar histogram.

    This function is mostly used to visualise the marginal magnitude
    histogram.

    Parameters
    ----------
    counts
        The counts in each bin.
    bin_edges
        The edges of the bins used to compute the histogram.
    fill
        Indicate whether to fill the histogram. If ``False`` then only an
        outline of the bars is drawn.
    ax
        Optional axes on which to plot. If ``None``, then new axes are
        created.
    log
        Indicate whether to use a logarithmic scale for the y-axis.
    **kwargs
        Keyword arguments for plotting the histogram.
        See :func:`matplotlib.axes.Axes.bar` and
        :class:`matplotlib.patches.Rectangle` for more details.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the histogram plot.

    See Also
    --------
    matplotlib.pyplot.bar: create a bar plot using provided heights.

    """

    ax = ax or plt.axes()
    bin_widths = bin_edges - np.roll(bin_edges, 1)
    ax.bar(bin_edges[:-1], counts, bin_widths[1:], align="edge", fill=fill, **kwargs)

    if log:
        ax.set_yscale("log")

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
    colour: str = "C0",
    max_angle: Optional[float] = None,
) -> matplotlib.projections.polar.PolarAxes:
    """Produce a 1D polar histogram plot.

    Produce a 1D polar histogram using the specified data on provided axes.

    Parameters
    ----------
    ax
        Matplotlib :class:`matplotlib.projections.polar.PolarAxes` on which
        to plot the data.
    data
        Histogram data to plot. This should have the same size as ``bins``.
    bins
        Lower edge of each histogram bin.
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
        Histogram bar colour. Must be a valid matplotlib colour [#f1]_.
    max_angle
        Maximum angle to represent on the angular axis in **degrees**. Must
        be between 0 and 360°. If `None`, a complete circle is drawn.

    Returns
    -------
    matplotlib.projections.polar.PolarAxes
        The ``PolarAxes`` used for plotting.

    Warnings
    --------
    The axes provided **must** be created using ``projection="Polar"``.

    The `max_angle` parameter allows for a portion of the circular plot to
    be drawn. **Any values beyond this maximum will be truncated.**

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
    ax.set_title(plot_title, pad=30)
    # ax.axes.yaxis.set_ticklabels([])

    if max_angle is not None:
        ax.set_thetamax(max_angle)
        axis_ticks = axis_ticks[axis_ticks <= max_angle]

    if label_axis:
        if axis_ticks_unit is AngularUnits.DEGREES:
            axis_ticks = np.radians(axis_ticks)

        ax.xaxis.set_ticks(axis_ticks)
    else:
        ax.xaxis.set_ticks([])

    ax.bar(bins, data, align="edge", width=bin_width, color=colour)

    # max_data = data.max()

    # base = np.ceil(max_data / 5).astype(int)
    # offset = np.ceil(max_data / 10).astype(int)
    #
    # ax.axes.yaxis.set_major_locator(matplotlib.ticker.IndexLocator(base, offset=offset))

    ax.axes.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5, prune="both"))
    ax.axes.tick_params(
        axis="y",
        labelsize="medium",
        labelcolor="#202020",
        labeltop=True,
    )

    ax.axes.tick_params(
        axis="both",
        grid_linestyle=":",
    )

    if max_angle is None:
        ax.set_rlabel_position(10)
        ax.axes.tick_params(
            axis="y",
            labelrotation=-10,
        )

    return ax


def produce_phi_theta_polar_histogram_plots(
    phi_data: pd.DataFrame,
    theta_data: pd.DataFrame,
    zero_position_2d: CardinalDirection = CardinalDirection.NORTH,
    rotation_direction: RotationDirection = RotationDirection.CLOCKWISE,
    use_degrees: bool = True,
    use_counts: bool = False,
    plot_title: Optional[str] = None,
    fig: Optional[plt.Figure] = None,
) -> plt.Figure:
    """Produce and show the 1D polar phi and theta histograms.

    This function takes in 2D binned histogram input and shows a 2-panel
    figure containing the theta and phi polar histograms.

    Parameters
    ----------
    phi_data
        Histogram data for the phi angle.
    theta_data
        Histogram data for the theta angle.
    zero_position_2d
        The :class:`CardinalDirection` where zero should be placed in the
        1D polar histograms (default: North).
    rotation_direction
        The :class:`RotationDirection` of increasing angles in the 1D polar
        histograms (default: clockwise).
    use_degrees
        Indicate whether the values are in degrees. If ``True``, values are
        assumed to be in degrees. Otherwise, radians are assumed.
    use_counts
        Indicate whether the bar heights should reflect the counts, as
        opposed to the frequencies.
    plot_title
        Title of the overall plot (optional).
    fig
        Figure on which to produce the plots. If `None`, a new figure is
        created.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the polar histogram plots.

    See Also
    --------
    produce_polar_histogram_plot :
        Create 1D polar histograms in isolation from 1D histogram data.
    """

    if use_counts:
        histogram_key = "count"
    else:
        histogram_key = "frequency"

    phi_histogram = phi_data[histogram_key].to_numpy()
    theta_histogram = theta_data[histogram_key].to_numpy()

    phi_bins = phi_data["start"].to_numpy()
    theta_bins = theta_data["start"].to_numpy()

    # Construct the 3D plot
    fig = fig or plt.figure(figsize=(7, 5))

    # Construct the 2D plots
    # Need to convert the bins back to radians if things have been done in degrees
    if use_degrees:
        phi_bins = np.radians(phi_bins)
        theta_bins = np.radians(theta_bins)

    # Construct the theta polar plot
    ax1 = fig.add_subplot(122, projection="polar")
    ax1 = produce_polar_histogram_plot(
        ax=ax1,
        data=theta_histogram,
        bins=theta_bins,
        zero_position=zero_position_2d,
        rotation_direction=rotation_direction,
        plot_title=r"$\theta$ (Angle in $XY$)",
    )

    # Construct the phi polar plot
    # Get the angular cutoff
    max_phi = phi_data["end"].max()

    ax2 = fig.add_subplot(121, projection="polar")
    ax2 = produce_polar_histogram_plot(
        ax=ax2,
        data=phi_histogram,
        bins=phi_bins,
        zero_position=zero_position_2d,
        rotation_direction=rotation_direction,
        plot_title=r"$\phi$ (Angle from $+Z$)",
        max_angle=max_phi
    )

    # Show the plots
    fig.suptitle(plot_title, fontweight="bold", fontsize=14)
    fig.subplots_adjust(wspace=0.25)
    # fig.subplots_adjust(left=0.05, right=0.95, wspace=0.25)
    return fig


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
    """Modify a 3D Matplotlib plot to label it with spherical axes.

    Modify existing axes to add spherical phi and theta axes, as well as
    labels and a colour bar.

    Parameters
    ----------
    ax
        Axes to modify. These must be 3D axes.
    radius
        Sphere radius in the 3D plot. This value is multiplied by
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

        phi_axis_cartesian = util.convert_spherical_to_cartesian_coordinates(
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

        theta_axis_cartesian = util.convert_spherical_to_cartesian_coordinates(
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
        spherical_coordinates_of_phi_labels[
            :, util.AngularIndex.PHI
        ] = phi_label_positions
        spherical_coordinates_of_phi_labels[
            :, util.AngularIndex.THETA
        ] = theta_position_for_phi_labels

        phi_label_positions_cartesian = util.convert_spherical_to_cartesian_coordinates(
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
            :, util.AngularIndex.THETA
        ] = theta_label_positions

        spherical_coordinates_of_theta_labels[
            :, util.AngularIndex.PHI
        ] = phi_position_for_theta_labels

        theta_label_positions_cartesian = (
            util.convert_spherical_to_cartesian_coordinates(
                angular_coordinates=spherical_coordinates_of_theta_labels,
                radius=axis_tick_factor * radius,
            )
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

    if plot_colour_bar:
        if norm is None:
            norm = plt.Normalize(vmin=minimum_value, vmax=maximum_value)
        scalar_mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=colour_map)
        # print(f"Colour bar has colour map {colour_map}.")
        if colour_bar_kwargs is None:
            colour_bar_kwargs = {}

        plt.colorbar(mappable=scalar_mappable, ax=ax, **colour_bar_kwargs)

    return ax


def produce_3d_triangle_sphere_plot(
    ax: mpl_toolkits.mplot3d.axes3d.Axes3D,
    sphere: TriangleSphere,
    face_counts: pd.Series,
    colour_map: str = "viridis",
    sphere_alpha: float = 1.0,
    norm: Optional[plt.Normalize] = None,
    **kwargs: Optional[dict[str, Any]],
) -> mpl_toolkits.mplot3d.axes3d.Axes3D:
    """Produce a 3D sphere plot based on a triangle mesh.

    Using the provided axes, plot a sphere with face colours corresponding
    to the provided values. This plot is generated using Matplotlib.

    Parameters
    ----------
    ax
        Axes on which to plot the sphere.
    sphere
        Triangle sphere to plot.
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
    vectorose.triangle_sphere.TriangleSphere:
        Produce a sphere and histogram labellings to pass to this function.
    .produce_labelled_3d_plot:
        Label the axes of the 3D plot.
    .produce_3d_tregenza_sphere_plot:
        Similar function for a Tregenza sphere.
    """

    # Get the face colours
    if norm is None:
        norm = matplotlib.colors.Normalize(
            vmin=face_counts.min(), vmax=face_counts.max()
        )
    else:
        norm.autoscale_None(face_counts)

    scalar_mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=colour_map)

    face_colours = scalar_mapper.to_rgba(face_counts.to_numpy())

    # Now, prepare the sphere for plotting
    sphere_mesh = sphere.create_mesh()
    vertices: np.ndarray = sphere_mesh.points
    x_coordinates = vertices[:, 0]
    y_coordinates = vertices[:, 1]
    z_coordinates = vertices[:, 2]

    triangles = sphere_mesh.faces.reshape(-1, 4)[:, 1:]

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
    sphere_bounds = np.array(sphere_mesh.bounds)
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
    tregenza_sphere: TregenzaSphere,
    histogram_data: Optional[pd.Series] = None,
    sphere_alpha: float = 1.0,
    colour_map: str = "viridis",
    norm: Optional[plt.Normalize] = None,
    **kwargs,
) -> mpl_toolkits.mplot3d.axes3d.Axes3D:
    """Produce a 3D sphere plot based on a Tregenza sphere.

    Using the provided axes, plot a Tregenza sphere with face colours
    corresponding to the provided histogram values. This plot is generated
    using Matplotlib.

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

    # If we have a histogram, work on processing the data
    face_colours: Optional[np.ndarray]

    if histogram_data is not None:
        if norm is None:
            # Find the maximum and minimum counts
            min_face_count = histogram_data.min()
            max_face_count = histogram_data.max()
            norm = matplotlib.colors.Normalize(vmin=min_face_count, vmax=max_face_count)
        else:
            norm.autoscale_None(histogram_data)

        # Compute the colours
        scalar_mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=colour_map)

        face_colours: np.ndarray = scalar_mapper.to_rgba(histogram_data)
    else:
        face_colours = None

    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)

    # Define the patches we'll plot
    all_patch_vertices: List[np.ndarray] = []

    rings = tregenza_sphere.to_dataframe()

    phi_values_upper = rings["start"]
    phi_values_lower = rings["end"]
    theta_bin_counts = rings["bins"]

    # So, let's start with the top row by starting with the second row.
    phi_upper = phi_values_upper.iloc[1]
    number_of_bins = theta_bin_counts.iloc[1]
    thetas_second_row = np.linspace(0, 360, number_of_bins, endpoint=False)

    phi_upper_second_row = np.ones(thetas_second_row.shape) * phi_upper
    top_cap_vertices = np.stack([phi_upper_second_row, thetas_second_row], axis=-1)

    top_cap_vertices_cartesian = util.convert_spherical_to_cartesian_coordinates(
        top_cap_vertices, use_degrees=True
    )

    all_patch_vertices.append(top_cap_vertices_cartesian)

    # Now, let's go with the remaining rings
    number_of_rings = tregenza_sphere.number_of_rings

    # phi_rings = np.append(phi_values, 90)
    for i in range(1, number_of_rings - 1):
        # Get the current phi ring and the next phi ring
        upper_phi = phi_values_upper.iloc[i]
        lower_phi = phi_values_lower.iloc[i]

        # Get the current theta bounds
        number_of_bins = theta_bin_counts.iloc[i]
        current_thetas = np.linspace(0, 360, number_of_bins, endpoint=False)
        # print(f"Considering ring {i} which contains {number_of_faces + 1} faces")

        # Now, for each face, we need to construct a rectangle with
        # four vertices, which are related to the bounds.
        # current_row_faces: list[np.ndarray] = []

        for j in range(number_of_bins):
            lower_theta = current_thetas[j]
            upper_theta = current_thetas[(j + 1) % number_of_bins]

            # Define the vertices
            v1 = (upper_phi, lower_theta)
            v2 = (upper_phi, upper_theta)
            v3 = (lower_phi, upper_theta)
            v4 = (lower_phi, lower_theta)

            face_vertices = np.array([v1, v2, v3, v4])

            face_vertices_cartesian = util.convert_spherical_to_cartesian_coordinates(
                face_vertices, use_degrees=True
            )

            all_patch_vertices.append(face_vertices_cartesian)

    # And finally, the bottom patch
    phi_value = phi_values_upper.iloc[-1]
    number_of_bins = theta_bin_counts.iloc[-2]
    thetas_second_last_row = np.linspace(0, 360, number_of_bins, endpoint=False)

    phi_value_bottom_row = np.ones(thetas_second_last_row.shape) * phi_value
    bottom_cap_vertices = np.stack(
        [phi_value_bottom_row, thetas_second_last_row], axis=-1
    )

    bottom_cap_vertices_cartesian = util.convert_spherical_to_cartesian_coordinates(
        np.radians(bottom_cap_vertices)
    )

    all_patch_vertices.append(bottom_cap_vertices_cartesian)

    patch_collection = mpl_toolkits.mplot3d.art3d.Poly3DCollection(
        all_patch_vertices,
        facecolors=face_colours,
        shade=False,
        linewidths=0,
        alpha=sphere_alpha,
    )

    ax.add_collection3d(patch_collection)

    # Define the sphere radius
    sphere_radius = 1

    kwargs["radius"] = sphere_radius

    # Add the labels to the plot
    ax = produce_labelled_3d_plot(ax=ax, norm=norm, colour_map=colour_map, **kwargs)

    ax.set_aspect("equal")

    return ax


def construct_uv_sphere_vertices(
    phi_steps: int = 80, theta_steps: int = 160, radius: float = 1
) -> np.ndarray:
    """Compute the vertices for a UV sphere with rectangular faces.

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
        in a format to plot using :meth:`Axes3D.plot_surface`. This array
        will have shape ``(theta_steps + 1, phi_steps + 1, 3)`` where the
        last axis corresponds to the ``X, Y, Z`` components.

    Warnings
    --------
    This sphere should not be used to plot histograms. It is provided for
    visualisations that do not involve plotting data on the surface of the
    sphere.

    Notes
    -----
    The coordinates computed using this function can easily be used to plot
    a sphere using :meth:`Axes3D.plot_surface`. To do so, the X, Y and Z
    coordinate sheets must be separated by indexing along the last axis.

    """

    # Compute the phi and theta values
    phi = np.linspace(start=0, stop=np.pi, num=phi_steps + 1, endpoint=True)
    theta = np.linspace(start=0, stop=2 * np.pi, num=theta_steps + 1, endpoint=True)

    # Now, build the 2D spherical coordinates
    phi_grid, theta_grid = np.meshgrid(phi, theta)

    # Convert to Cartesian coordinates
    sphere_angles = np.stack([phi_grid, theta_grid], axis=-1)

    sphere_cartesian_coordinates = util.convert_spherical_to_cartesian_coordinates(
        angular_coordinates=sphere_angles, radius=radius
    )

    return sphere_cartesian_coordinates


def construct_uv_sphere_mesh(
    phi_steps: int = 80, theta_steps: int = 160, radius: float = 1
) -> pv.PolyData:
    """Construct a UV sphere mesh.

    Parameters
    ----------
    phi_steps
        Number of phi rings to construct.
    theta_steps
        Number of theta bins in each ring.
    radius
        Sphere radius.

    Returns
    -------
    pyvista.PolyData
        Mesh containing the UV sphere.
    """

    top_vertex = np.array([0, 0, radius])

    phi_step_size = 180 / phi_steps
    phi = np.linspace(phi_step_size, stop=180, num=phi_steps - 1, endpoint=False)
    theta = np.linspace(0, 360, theta_steps, endpoint=False)

    # Now, build the 2D spherical coordinates
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    # Convert to Cartesian coordinates
    sphere_angles = np.stack([phi_grid, theta_grid], axis=-1)

    sphere_cartesian_coordinates = util.convert_spherical_to_cartesian_coordinates(
        angular_coordinates=sphere_angles, radius=radius, use_degrees=True
    )

    sphere_cartesian_coordinates = sphere_cartesian_coordinates.reshape(-1, 3)

    bottom_vertex = np.array([0, 0, -radius])

    # Add the top and bottom caps
    sphere_cartesian_coordinates = np.vstack(
        [top_vertex, sphere_cartesian_coordinates, bottom_vertex]
    )

    # And now, to begin defining the faces
    cells: List[np.ndarray] = []

    # Define the top triangle fan
    face_count = 3 * np.ones(theta_steps, dtype=int)
    first_ring_starts = np.arange(1, theta_steps + 1)
    first_ring_ends = np.roll(first_ring_starts, -1)
    apex = np.zeros(theta_steps, dtype=int)
    top_fan_cells = np.stack(
        [face_count, apex, first_ring_starts, first_ring_ends], axis=-1
    )
    cells.append(top_fan_cells)

    # Define each row
    for i in range(0, phi_steps - 2):
        first_vertex_in_row = 1 + i * theta_steps
        upper_ring = np.arange(first_vertex_in_row, first_vertex_in_row + theta_steps)

        vertex_count_column = 4 * np.ones(theta_steps, dtype=int)
        top_left_corner = upper_ring
        top_right_corner = np.roll(top_left_corner, -1)
        bottom_left_corner = upper_ring + theta_steps
        bottom_right_corner = top_right_corner + theta_steps

        ring_cells = np.stack(
            [
                vertex_count_column,
                top_left_corner,
                top_right_corner,
                bottom_right_corner,
                bottom_left_corner,
            ],
            axis=-1,
        )

        cells.append(ring_cells)

    # Define the bottom triangle fan
    bottom_apex_index = len(sphere_cartesian_coordinates) - 1
    face_count = 3 * np.ones(theta_steps, dtype=int)
    last_ring_starts = np.arange(bottom_apex_index - theta_steps, bottom_apex_index)
    last_ring_ends = np.roll(last_ring_starts, -1)
    bottom_apex = bottom_apex_index * np.ones(theta_steps, dtype=int)
    bottom_fan_cells = np.stack(
        [face_count, bottom_apex, last_ring_starts, last_ring_ends], axis=-1
    )
    cells.append(bottom_fan_cells)

    cell_array = np.concatenate(cells, axis=-1)

    # Create the mesh
    uv_sphere_mesh = pv.PolyData(sphere_cartesian_coordinates, cell_array)

    return uv_sphere_mesh


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
    Iterable of mpl_toolkits.mplot3d.axes3d.Axes3D
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


def construct_confidence_cone(
    angular_radius: float,
    number_of_patches: int = 80,
    mean_orientation: Optional[np.ndarray] = None,
    two_sided_cone: bool = True,
    use_degrees: bool = False,
    **kwargs,
) -> List[mpl_toolkits.mplot3d.art3d.Poly3DCollection]:
    """Construct the patches for a confidence cone.

    Construct the triangular patches for a confidence cone with a specified
    angular radius, and optionally rotated to a specified mean direction.

    Parameters
    ----------
    angular_radius
        Angular radius for the confidence cone bounds (in radians, unless
        the parameter `use_degrees` is `True`).
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
    use_degrees
        Indicate whether the provided angular radius is in degrees.
    **kwargs
        Keyword arguments for the patch construction.
        See :class:`~mpl_toolkits.mplot3d.art3d.Poly3DCollection` for
        details.

    Returns
    -------
    list of mpl_toolkits.mplot3d.art3d.Poly3DCollection
        List of :class:`~mpl_toolkits.mplot3d.art3d.Poly3DCollection`
        representing each patch of the confidence cone. These patches are
        triangular.
    """

    # Convert to radians, if necessary
    if use_degrees:
        angular_radius = np.radians(angular_radius)

    # Create a list of patches
    patches: List[mpl_toolkits.mplot3d.art3d.Poly3DCollection] = []

    # Construct the rotation matrix
    if mean_orientation is not None:
        mean_orientation_spherical = util.compute_vector_orientation_angles(
            vectors=mean_orientation
        )

        mean_phi = mean_orientation_spherical[util.AngularIndex.PHI]
        mean_theta = mean_orientation_spherical[util.AngularIndex.THETA]

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

        ring_vertices_cartesian = util.convert_spherical_to_cartesian_coordinates(
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


def produce_3d_confidence_cone_plot(
    ax: mpl_toolkits.mplot3d.axes3d.Axes3D,
    confidence_cone_patches: List[mpl_toolkits.mplot3d.art3d.Poly3DCollection],
    sphere_vertices: np.ndarray,
    sphere_radius: float = 1,
    sphere_alpha: float = 0.5,
    sphere_colour: str = "#a8a8a8",
    **kwargs,
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
    .construct_uv_sphere_vertices :
        Generate vertices for a quad-based sphere.
    """

    # Plot the confidence cone
    for patch in confidence_cone_patches:
        ax.add_collection3d(patch)

    # Plot the sphere
    xs = sphere_vertices[..., 0]
    ys = sphere_vertices[..., 1]
    zs = sphere_vertices[..., 2]
    ax.plot_surface(xs, ys, zs, color=sphere_colour, alpha=sphere_alpha)

    # Ensure the radius is only passed once
    kwargs["radius"] = sphere_radius

    # Label the plot
    ax = produce_labelled_3d_plot(ax=ax, **kwargs)

    ax.set_aspect("equal")

    return ax
