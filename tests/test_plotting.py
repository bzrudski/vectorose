"""Test for VectoRose plotting.

Unit tests for the module :mod:`vectorose.plotting`, which includes all the
plotting functionality for VectoRose.

This test module is structured to first test the methods in the
:class:`.SpherePlotter` class, and then the assorted functions for plotting
using Matplotlib.
"""
import os

import matplotlib as mpl
import matplotlib.container
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import pytest
import vectorose as vr
import vectorose.mock_data

RANDOM_SEED = 20241220


@pytest.fixture
def setup_pyvista_environment():
    """Set up the PyVista environment for testing"""
    pv.OFF_SCREEN = True


@pytest.fixture
def vectors() -> np.ndarray:
    """Generate test vectors for the unit tests."""

    vectors = vr.mock_data.create_vonmises_fisher_vectors_single_direction(
        phi=50,
        theta=60,
        kappa=10,
        number_of_points=100_000,
        magnitude=0.5,
        magnitude_std=0.2,
        use_degrees=True,
        seed=RANDOM_SEED,
    )

    magnitudes = np.linalg.norm(vectors, axis=-1)
    vectors[magnitudes > 1] = 1

    return vectors


@pytest.fixture
def dummy_mesh(setup_pyvista_environment) -> pv.PolyData:
    """Generate a single dummy mesh for testing."""

    sphere = vr.tregenza_sphere.FineTregenzaSphere()

    sphere_mesh = sphere.create_mesh()

    return sphere_mesh


@pytest.fixture
def label_vectors(vectors):
    """Generate labelled vectors for testing the Tregenza sphere."""
    number_of_shells = 32
    sphere = vr.tregenza_sphere.FineTregenzaSphere(
        number_of_shells=number_of_shells, magnitude_range=(0, 1)
    )
    labelled_vectors, magnitude_bins = sphere.assign_histogram_bins(vectors)
    return labelled_vectors, magnitude_bins, sphere


@pytest.fixture
def label_vectors_triangulated(vectors):
    """Generate labelled vectors for testing the triangulated sphere."""
    number_of_shells = 32
    number_of_subdivisions = 5
    sphere = vr.triangle_sphere.TriangleSphere(
        number_of_subdivisions=number_of_subdivisions,
        number_of_shells=number_of_shells,
        magnitude_range=(0, 1),
    )
    labelled_vectors, magnitude_bins = sphere.assign_histogram_bins(vectors)
    return labelled_vectors, magnitude_bins, sphere


@pytest.fixture
def label_vectors_polar(vectors):
    """Generate labelled vectors for testing the polar histograms."""
    number_of_phi_bins = 18
    number_of_theta_bins = 36
    is_axial = False

    polar_discretiser = vr.polar_data.PolarDiscretiser(
        number_of_phi_bins, number_of_theta_bins, is_axial
    )

    labelled_vectors = polar_discretiser.assign_histogram_bins(vectors)

    return labelled_vectors, polar_discretiser


@pytest.fixture
def mock_nested_histogram_meshes(
    label_vectors: tuple[pd.DataFrame, np.ndarray, vr.tregenza_sphere.TregenzaSphere]
) -> list[pv.PolyData]:
    """Generate several nested dummy meshes for testing."""

    labelled_vectors, magnitude_bins, sphere = label_vectors
    hist = sphere.construct_histogram(labelled_vectors)
    sphere_meshes = sphere.create_histogram_meshes(hist, magnitude_bins)

    return sphere_meshes


def test_sphere_plotter_initialisation_one_mesh(dummy_mesh):
    """Test for creating a SpherePlotter with one mesh.

    Kick the tires to make sure that the properties set by the initialiser
    are as expected when creating a plotter with a single mesh.
    """

    plotter = vr.plotting.SpherePlotter(dummy_mesh)

    # Check the mesh-related properties
    assert len(plotter.sphere_meshes) == 1
    assert np.isclose(plotter.radius, 1)
    assert plotter.sphere_meshes[0] == dummy_mesh


def test_sphere_plotter_initialisation_many_meshes(mock_nested_histogram_meshes):
    """Test for creating a SpherePlotter with several meshes.

    Kick the tires to make sure that the properties set by the initialiser
    are as expected when creating a plotter with several meshes.
    """

    plotter = vr.plotting.SpherePlotter(mock_nested_histogram_meshes)

    # Check the mesh-related properties
    assert len(plotter.sphere_meshes) == len(mock_nested_histogram_meshes)
    assert np.isclose(plotter.radius, 1)
    assert plotter.sphere_meshes == mock_nested_histogram_meshes


#
# def test_add_spherical_axes_phi_theta(dummy_mesh):
#     """Test the spherical axis plotting functionality.
#
#     Test for :meth:`.SpherePlotter.add_spherical_axes` with both the phi
#     and the theta axes.
#     """
#
#     plotter = vr.plotting.SpherePlotter(dummy_mesh)
#
#     # Add the spherical axes
#     plotter.add_spherical_axes(plot_phi=True, plot_theta=True)
#
#     # Check that they've actually been added
#     assert False


def test_export_screenshot(mock_nested_histogram_meshes, tmp_path):
    """Test image exporting for SpherePlotter."""

    plotter = vr.plotting.SpherePlotter(mock_nested_histogram_meshes)
    plotter.produce_plot()

    export_name = os.path.join(tmp_path, "screenshot.png")

    plotter.export_screenshot(export_name)

    assert os.path.exists(export_name)


def test_export_graphic(mock_nested_histogram_meshes, tmp_path):
    """Test graphic exporting for SpherePlotter."""

    plotter = vr.plotting.SpherePlotter(mock_nested_histogram_meshes)
    plotter.produce_plot()

    export_name = os.path.join(tmp_path, "screenshot.svg")
    export_title = "Graphic"

    plotter.export_graphic(export_name, export_title)

    assert os.path.exists(export_name)


def test_export_rotating_video(mock_nested_histogram_meshes, tmp_path):
    """Test rotating video export for SpherePlotter."""

    plotter = vr.plotting.SpherePlotter(mock_nested_histogram_meshes)
    plotter.produce_plot()

    export_name = os.path.join(tmp_path, "video.mp4")

    plotter.produce_rotating_video(export_name)

    assert os.path.exists(export_name)


def test_export_shells_video(mock_nested_histogram_meshes, tmp_path):
    """Test shells video export for SpherePlotter."""

    plotter = vr.plotting.SpherePlotter(mock_nested_histogram_meshes)
    plotter.produce_plot()

    export_name = os.path.join(tmp_path, "video.mp4")

    plotter.produce_shells_video(export_name)

    assert os.path.exists(export_name)


def test_produce_1d_scalar_histogram(
    tmp_path,
    label_vectors: tuple[pd.DataFrame, np.ndarray, vr.tregenza_sphere.TregenzaSphere],
):
    """Test for plotting a 1D scalar histogram.

    Test for :func:`.plotting.produce_1d_scalar_histogram` to ensure that
    the histogram is properly constructed.
    """

    labelled_vectors, magnitude_bins, sphere = label_vectors

    magnitude_hist = sphere.construct_marginal_magnitude_histogram(labelled_vectors)

    # Plot the histogram plot
    fig = plt.figure()
    ax = plt.axes()
    ax = vr.plotting.produce_1d_scalar_histogram(magnitude_hist, magnitude_bins, ax=ax)
    fig.add_axes(ax)
    fig.savefig(os.path.join(tmp_path, "test_plot.png"))

    # Test the number of bars, like in the Py-Pkgs book
    expected_number_of_bars = len(magnitude_bins) - 1

    # Try to get the bar container from the plot
    bar_container = ax.containers[0]

    assert isinstance(bar_container, mpl.container.BarContainer), "Expected bar plot."

    assert (
        len(bar_container.datavalues) == expected_number_of_bars
    ), "Wrong number of bars plotted."

    assert np.all(
        np.isclose(bar_container.datavalues, magnitude_hist.to_numpy())
    ), "Unexpected bar heights."


def _test_produce_polar_histogram_plot(hist, tmp_path):
    """Test the polar histogram plotting."""

    data = hist["count"].to_numpy()
    bins = hist["start"].to_numpy()

    # Produce the polar plot
    fig = plt.figure()
    ax = plt.axes(projection="polar")
    ax = vr.plotting.produce_polar_histogram_plot(ax, data, bins)
    fig.add_axes(ax)
    fig.savefig(os.path.join(tmp_path, "test_plot.png"))

    # Test the number of bars, like in the Py-Pkgs book
    expected_number_of_bars = len(bins)

    # Try to get the bar container from the plot
    bar_container = ax.containers[0]
    assert isinstance(bar_container, mpl.container.BarContainer), "Expected bar plot."
    assert (
        len(bar_container.datavalues) == expected_number_of_bars
    ), "Wrong number of bars plotted."
    assert np.all(np.isclose(bar_container.datavalues, data)), "Unexpected bar heights."


def test_produce_polar_histogram_plot_phi(
    tmp_path,
    label_vectors_polar: tuple[pd.DataFrame, vr.polar_data.PolarDiscretiser],
):
    """Test for plotting a 1D polar histogram for phi angles.

    Test for :func:`.plotting.produce_polar_histogram_plot` to ensure that
    the histogram is properly constructed.
    """

    labelled_vectors, polar_discretiser = label_vectors_polar

    phi_histogram = polar_discretiser.construct_phi_histogram(labelled_vectors)

    _test_produce_polar_histogram_plot(phi_histogram, tmp_path)


def test_produce_polar_histogram_plot_theta(
    tmp_path,
    label_vectors_polar: tuple[pd.DataFrame, vr.polar_data.PolarDiscretiser],
):
    """Test for plotting a 1D polar histogram for theta angles.

    Test for :func:`.plotting.produce_polar_histogram_plot` to ensure that
    the histogram is properly constructed.
    """

    labelled_vectors, polar_discretiser = label_vectors_polar

    theta_histogram = polar_discretiser.construct_theta_histogram(labelled_vectors)

    _test_produce_polar_histogram_plot(theta_histogram, tmp_path)


def test_produce_phi_theta_polar_histogram_plots(
    label_vectors_polar: tuple[pd.DataFrame, vr.polar_data.PolarDiscretiser],
):
    """Test for plotting the phi and theta polar histograms.

    Test for :func:`.plotting.test_produce_phi_theta_polar_histogram_plots`
    to ensure that the histograms are properly constructed.
    """

    labelled_vectors, polar_discretiser = label_vectors_polar

    phi_histogram = polar_discretiser.construct_phi_histogram(labelled_vectors)
    theta_histogram = polar_discretiser.construct_theta_histogram(labelled_vectors)

    vr.plotting.produce_phi_theta_polar_histogram_plots(
        phi_histogram, theta_histogram, use_counts=False
    )

    fig = plt.gcf()

    # Make sure that there are indeed two plots
    assert len(fig.axes) == 2

    # Check the theta plot
    theta_axes = fig.axes[0]
    theta_data = theta_histogram["frequency"]
    theta_bins = theta_histogram["start"]

    # Test the number of bars, like in the Py-Pkgs book
    expected_number_of_bars = len(theta_bins)

    # Try to get the bar container from the plot
    bar_container = theta_axes.containers[0]
    assert isinstance(bar_container, mpl.container.BarContainer), "Expected bar plot."
    assert (
        len(bar_container.datavalues) == expected_number_of_bars
    ), "Wrong number of bars plotted."
    assert np.all(
        np.isclose(bar_container.datavalues, theta_data)
    ), "Unexpected bar heights."

    # Check the phi plot
    phi_axes = fig.axes[1]
    phi_data = phi_histogram["frequency"]
    phi_bins = phi_histogram["start"]

    # Test the number of bars, like in the Py-Pkgs book
    expected_number_of_bars = len(phi_bins)

    # Try to get the bar container from the plot
    bar_container = phi_axes.containers[0]
    assert isinstance(bar_container, mpl.container.BarContainer), "Expected bar plot."
    assert (
        len(bar_container.datavalues) == expected_number_of_bars
    ), "Wrong number of bars plotted."
    assert np.all(
        np.isclose(bar_container.datavalues, phi_data)
    ), "Unexpected bar heights."


def test_tregenza_plotting_mpl(
    tmp_path,
    label_vectors: tuple[pd.DataFrame, np.ndarray, vr.tregenza_sphere.TregenzaSphere],
):
    """Unit test for plotting the fine Tregenza sphere.

    Test for :func:`.plotting.produce_3d_tregenza_sphere_plot` which uses
    Matplotlib to produce a spherical histogram plot.
    """

    labelled_vectors, magnitude_bins, sphere = label_vectors

    orientation_hist = sphere.construct_marginal_orientation_histogram(labelled_vectors)

    # Plot the sphere
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax = vr.plotting.produce_3d_tregenza_sphere_plot(ax, sphere, orientation_hist)
    fig.add_axes(ax)
    fig.savefig(os.path.join(tmp_path, "test_plot.png"))

    # Get the 3D patch collections
    patches = ax.collections[0].get_paths()
    actual_number_of_patches = len(patches)

    # Get the expected number of faces in the sphere
    sphere_frame = sphere.to_dataframe()
    expected_number_of_patches = sphere_frame["bins"].sum()

    # Check to make sure the number of patches is as expected
    assert actual_number_of_patches == expected_number_of_patches

    plt.close(fig)


def test_triangulated_plotting_mpl(
    tmp_path,
    label_vectors_triangulated: tuple[
        pd.DataFrame, np.ndarray, vr.triangle_sphere.TriangleSphere
    ],
):
    """Unit test for plotting the fine Tregenza sphere.


    Test for :func:`.plotting.produce_3d_triangle_sphere_plot` which uses
    Matplotlib to produce a spherical histogram plot.
    """

    labelled_vectors, magnitude_bins, sphere = label_vectors_triangulated

    orientation_hist = sphere.construct_marginal_orientation_histogram(labelled_vectors)

    # Plot the sphere
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax = vr.plotting.produce_3d_triangle_sphere_plot(
        ax, sphere, orientation_hist, plot_colour_bar=True
    )
    fig.add_axes(ax)
    fig.savefig(os.path.join(tmp_path, "test_plot.png"))

    # Get the 3D patch collections
    patches = ax.collections[0].get_paths()
    actual_number_of_patches = len(patches)

    # Get the expected number of faces in the sphere
    sphere_frame = sphere.to_dataframe()
    expected_number_of_patches = len(sphere_frame)

    # Check to make sure the number of patches is as expected
    assert actual_number_of_patches == expected_number_of_patches

    # Check that there are two axes plotted: one for the plot, and one for
    # the colour bar.
    assert len(fig.axes) == 2

    plt.close(fig)


def test_construct_uv_sphere_vertices():
    """Test the UV sphere vertex calculation.

    Test for :func:`.plotting.construct_uv_sphere_vertices` to determine
    whether the correct number of vertices are computed and all are at a
    distance from the origin corresponding to the desired radius.
    """

    phi_steps = 18
    theta_steps = 36
    radius = 5
    expected_vertices_shape = (theta_steps + 1, phi_steps + 1, 3)

    expected_phi_increment = 180 / phi_steps
    expected_theta_increment = 360 / theta_steps

    sphere_vertices = vr.plotting.construct_uv_sphere_vertices(
        phi_steps, theta_steps, radius
    )

    # Compute the vector norms to test the radius
    vertex_list = sphere_vertices.reshape(-1, 3)
    vertex_distances = np.linalg.norm(vertex_list, axis=-1)

    # Convert the vertices to spherical coordinates and test the angles
    vertices_spherical_coordinates = vr.util.compute_vector_orientation_angles(
        vertex_list, use_degrees=True
    )

    # To compute the angular increments, we need to extract the unique phi
    # and theta angles and sort them. Rounding is performed to mitigate
    # floating point errors.
    phis = np.sort(np.unique(np.round(vertices_spherical_coordinates[:, 0], 5)))
    thetas = np.sort(np.unique(np.round(vertices_spherical_coordinates[:, 1], 5)))

    phi_increments = phis - np.roll(phis, 1)
    phi_increments = phi_increments[1:]

    theta_increments = thetas - np.roll(thetas, 1)
    theta_increments = theta_increments[1:]

    # Check the array shape
    assert sphere_vertices.shape == expected_vertices_shape

    # Check the vertex distances
    assert np.all(np.isclose(vertex_distances, radius))

    # Check the angles
    assert np.all(np.isclose(phi_increments, expected_phi_increment))
    assert np.all(np.isclose(theta_increments, expected_theta_increment))


def test_construct_uv_sphere_mesh():
    """Test UV sphere mesh construction.

    Test for :func:`.plotting.construct_uv_sphere_mesh` that ensures that
    the UV sphere mesh is properly constructed.
    """

    phi_steps = 18
    theta_steps = 36
    radius = 5

    # Compute the expected number of vertices and faces.
    # Recall that the sphere has triangle fans at the poles!
    expected_number_of_vertices = (phi_steps - 1) * theta_steps + 2
    expected_number_of_faces = phi_steps * theta_steps

    # Construct the mesh
    uv_sphere_mesh = vr.plotting.construct_uv_sphere_mesh(
        phi_steps, theta_steps, radius
    )

    # Get the vertices and the faces
    number_of_vertices = uv_sphere_mesh.n_points
    number_of_faces = uv_sphere_mesh.n_cells

    # Check everything
    assert number_of_vertices == expected_number_of_vertices
    assert number_of_faces == expected_number_of_faces


def test_animate_sphere_plot(
    tmp_path,
    label_vectors: tuple[pd.DataFrame, np.ndarray, vr.tregenza_sphere.TregenzaSphere],
):
    """Test sphere plot animation.

    Test for :func:`.plotting.animate_sphere_plot`. This test ensures that
    an animation is indeed produced with the desired number of frames.

    This test also saves the animation.
    """

    labelled_vectors, magnitude_bins, sphere = label_vectors

    orientation_hist = sphere.construct_marginal_orientation_histogram(labelled_vectors)

    # Define rotation parameters
    angle_increment = 10
    animation_delay = 200

    # Compute the expecte number of frames
    expected_number_of_frames = np.ceil(360 / angle_increment).astype(int)

    # Plot the sphere
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax = vr.plotting.produce_3d_tregenza_sphere_plot(ax, sphere, orientation_hist)
    fig.add_axes(ax)
    fig.savefig(os.path.join(tmp_path, "test_plot.png"))

    animation = vr.plotting.animate_sphere_plot(
        fig, ax, angle_increment=angle_increment, animation_delay=animation_delay
    )

    frames = list(animation.frame_seq)

    # Save the animation as a movie
    export_filename_mp4 = os.path.join(tmp_path, "animation.mp4")
    vr.io.export_mpl_animation(animation, export_filename_mp4)

    # Save the animation as a GIF
    export_filename_gif = os.path.join(tmp_path, "animation.gif")
    vr.io.export_mpl_animation(animation, export_filename_gif)

    # Check the number of frames
    assert len(frames) == expected_number_of_frames

    # Check the exports
    assert os.path.exists(export_filename_mp4)
    assert os.path.exists(export_filename_gif)


def test_construct_confidence_cone_one_sided():
    """Test the confidence cone plotting.

    Test for :func:`.plotting.construct_confidence_cone` to ensure that the
    correct number of patches is produced.

    Warnings
    --------
    Due to the interface of
    :class:`mpl_toolkits.mplot3d.art3d.Poly3DCollection`, there are limited
    options for actually testing the layout of the patches.
    """

    angular_radius = 5
    number_of_patches = 80
    mean_orientation = np.array([1, 0, 0])
    two_sided_cone = False
    use_degrees = True

    # Build the patches
    confidence_cone_patches = vr.plotting.construct_confidence_cone(
        angular_radius, number_of_patches, mean_orientation, two_sided_cone, use_degrees
    )

    # Collect all the vertices
    assert len(confidence_cone_patches) == number_of_patches


def test_construct_confidence_cone_two_sided():
    """Test the confidence cone plotting.

    Test for :func:`.plotting.construct_confidence_cone` to ensure that the
    correct number of patches is produced.

    Warnings
    --------
    Due to the interface of
    :class:`mpl_toolkits.mplot3d.art3d.Poly3DCollection`, there are limited
    options for actually testing the layout of the patches.
    """

    angular_radius = 5
    number_of_patches = 80
    mean_orientation = np.array([1, 0, 0])
    two_sided_cone = True
    use_degrees = True

    # Build the patches
    confidence_cone_patches = vr.plotting.construct_confidence_cone(
        angular_radius, number_of_patches, mean_orientation, two_sided_cone, use_degrees
    )

    # Collect all the vertices
    assert len(confidence_cone_patches) == 2 * number_of_patches


def test_produce_3d_confidence_cone_plot(tmp_path):
    """Test the function for plotting the confidence cone.

    Test of :func:`.plotting.produce_3d_confidence_cone_plot` to ensure
    that everything is plotted correctly.
    """

    angular_radius = 5
    number_of_patches = 80
    mean_orientation = np.array([1, 0, 0])
    two_sided_cone = False
    use_degrees = True

    # Build the patches
    confidence_cone_patches = vr.plotting.construct_confidence_cone(
        angular_radius, number_of_patches, mean_orientation, two_sided_cone, use_degrees
    )

    # Get the sphere vertices
    sphere_vertices = vr.plotting.construct_uv_sphere_vertices()

    # Prepare the plot
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax = vr.plotting.produce_3d_confidence_cone_plot(
        ax, confidence_cone_patches, sphere_vertices
    )
    fig.add_axes(ax)
    fig.savefig(os.path.join(tmp_path, "test_plot.png"))

    # Check that everything is plotted properly (n patches + 1 for sphere)
    assert len(ax.collections) == len(confidence_cone_patches) + 1
