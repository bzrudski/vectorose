"""Tests for VectoRose Tregenza histogram plotting.

This module contains the automated tests for the
:mod:`vectorose.tregenza_sphere` module, which provides the Tregenza-based
spherical histogram construction.
"""
import os.path

import matplotlib.pyplot as plt
import numpy as np

from vectorose import tregenza_sphere, mock_data, util

RANDOM_SEED = 20240827


def generate_test_vectors() -> np.ndarray:
    """Generate test vectors for the unit tests."""

    vectors = mock_data.create_vonmises_fisher_vectors_single_direction(
        phi=50,
        theta=60,
        kappa=10,
        number_of_points=100_000,
        magnitude=5,
        magnitude_std=1,
        use_degrees=True,
        seed=RANDOM_SEED,
    )
    return vectors


def test_construct_fine_spherical_histogram_magnitude_weight():
    """Test the spherical histogram construct, weighted by magnitude.

    Unit test for :func:`tregenza_sphere.construct_spherical_histogram` to
    check the magnitude-weighted histogram construction for the fine
    Tregenza sphere.
    """

    # Create a set of random vectors
    vectors = generate_test_vectors()

    # Construct a Tregenza sphere
    sphere = tregenza_sphere.FineTregenzaSphere()

    # Compute the vector magnitudes
    magnitudes = np.linalg.norm(vectors, axis=-1)

    # Compute the angular coordinates
    angles = util.compute_vector_orientation_angles(vectors, use_degrees=True)

    # Now, construct a magnitude-weighted histogram.
    face_values = sphere.construct_spherical_histogram(angles, magnitudes=magnitudes)

    # Check that the output face counts have the same shape as the sphere
    sphere_patch_count = sphere.patch_count
    face_value_sizes = np.array([len(fv) for fv in face_values])

    assert np.all(face_value_sizes == sphere_patch_count)

    # Now, check that the sum of the face values corresponds to the sum of
    # the vector magnitudes used to construct the histogram.
    total_face_mag_sum = np.concatenate(face_values).sum()

    total_vector_mag = magnitudes.sum()

    assert np.isclose(total_face_mag_sum, total_vector_mag)


def test_construct_fine_spherical_histogram_count_weight():
    """Test the spherical histogram construct, weighted by count.

    Unit test for :func:`tregenza_sphere.construct_spherical_histogram` to
    check the count-weighted histogram construction for the fine Tregenza
    sphere.
    """

    # Create a set of random vectors
    vectors = generate_test_vectors()

    # Construct a Tregenza sphere
    sphere = tregenza_sphere.FineTregenzaSphere()

    # Compute the angular coordinates
    angles = util.compute_vector_orientation_angles(vectors, use_degrees=True)

    # Now, construct a magnitude-weighted histogram.
    face_values = sphere.construct_spherical_histogram(angles)

    # Check that the output face counts have the same shape as the sphere
    sphere_patch_count = sphere.patch_count
    face_value_sizes = np.array([len(fv) for fv in face_values])

    assert np.all(face_value_sizes == sphere_patch_count)

    # Now, check that the sum of the counts corresponds to the number of
    # vectors used to construct the histogram.
    face_values_flat = np.concatenate(face_values)

    assert face_values_flat.sum() == len(vectors)


def test_construct_ultra_fine_spherical_histogram_magnitude_weight():
    """Test the spherical histogram construct, weighted by magnitude.

    Unit test for :func:`tregenza_sphere.construct_spherical_histogram` to
    check the magnitude-weighted histogram construction for the ultra-fine
    Tregenza sphere.
    """

    # Create a set of random vectors
    vectors = generate_test_vectors()

    # Construct a Tregenza sphere
    sphere = tregenza_sphere.UltraFineTregenzaSphere()

    # Compute the vector magnitudes
    magnitudes = np.linalg.norm(vectors, axis=-1)

    # Compute the angular coordinates
    angles = util.compute_vector_orientation_angles(vectors, use_degrees=True)

    # Now, construct a magnitude-weighted histogram.
    face_values = sphere.construct_spherical_histogram(angles, magnitudes=magnitudes)

    # Check that the output face counts have the same shape as the sphere
    sphere_patch_count = sphere.patch_count
    face_value_sizes = np.array([len(fv) for fv in face_values])

    assert np.all(face_value_sizes == sphere_patch_count)

    # Now, check that the sum of the face values corresponds to the sum of
    # the vector magnitudes used to construct the histogram.
    total_face_mag_sum = np.concatenate(face_values).sum()

    total_vector_mag = magnitudes.sum()

    assert np.isclose(total_face_mag_sum, total_vector_mag)


def test_construct_ultra_fine_spherical_histogram_count_weight():
    """Test the spherical histogram construct, weighted by count.

    Unit test for :func:`tregenza_sphere.construct_spherical_histogram` to
    check the count-weighted histogram construction for the ultra-fine
    Tregenza sphere.
    """

    # Create a set of random vectors
    vectors = generate_test_vectors()

    # Construct a Tregenza sphere
    sphere = tregenza_sphere.UltraFineTregenzaSphere()

    # Compute the angular coordinates
    angles = util.compute_vector_orientation_angles(vectors, use_degrees=True)

    # Now, construct a magnitude-weighted histogram.
    face_values = sphere.construct_spherical_histogram(angles)

    # Check that the output face counts have the same shape as the sphere
    sphere_patch_count = sphere.patch_count
    face_value_sizes = np.array([len(fv) for fv in face_values])

    assert np.all(face_value_sizes == sphere_patch_count)

    # Now, check that the sum of the counts corresponds to the number of
    # vectors used to construct the histogram.
    face_values_flat = np.concatenate(face_values)

    assert face_values_flat.sum() == len(vectors)


def test_face_area_correction_fine():
    """Test the face area correction for the fine Tregenza sphere."""

    # Create a fine Tregenza sphere
    sphere = tregenza_sphere.FineTregenzaSphere()

    # Compute the face areas
    face_areas = sphere.compute_weights()

    # Build up a histogram count with each face having its respective area.
    patch_count = sphere.patch_count

    histogram_data = []

    for i in range(sphere.number_of_rings):
        face_number = patch_count[i]
        face_area = face_areas[i]
        histogram_data.append(np.repeat(1 / face_area, face_number))

    # We'll consider those face areas to be our weights. Now, correct them.
    corrected_face_areas = sphere.correct_histogram_by_area(histogram_data)

    # Concatenate all the face values
    corrected_face_areas_flat = np.concatenate(corrected_face_areas)

    # Now, each face should be equal to 1
    assert np.all(np.isclose(corrected_face_areas_flat, 1.0))


def test_face_area_correction_ultra_fine():
    """Test the face area correction for the ultra-fine Tregenza sphere."""

    # Create a fine Tregenza sphere
    sphere = tregenza_sphere.UltraFineTregenzaSphere()

    # Compute the face areas
    face_areas = sphere.compute_weights()

    # Build up a histogram count with each face having its respective area.
    patch_count = sphere.patch_count

    histogram_data = []

    for i in range(sphere.number_of_rings):
        face_number = patch_count[i]
        face_area = face_areas[i]
        histogram_data.append(np.repeat(1 / face_area, face_number))

    # We'll consider those face areas to be our weights. Now, correct them.
    corrected_face_areas = sphere.correct_histogram_by_area(histogram_data)

    # Concatenate all the face values
    corrected_face_areas_flat = np.concatenate(corrected_face_areas)

    # Now, each face should be equal to 1
    assert np.all(np.isclose(corrected_face_areas_flat, 1.0))


def test_find_closest_patch_fine():
    """Unit test for the closest patch in the fine Tregenza sphere."""

    angles = [
        (2, 70),
        (10, 40),
        (90, 180),
        (160, 358),
        (180, 30),
    ]

    sphere = tregenza_sphere.FineTregenzaSphere()

    bins = [sphere.get_closest_face(phi, theta) for phi, theta in angles]

    expected_bins = [
        (1, 1),
        (3, 3),
        (27, 86),
        (47, 57),
        (53, 0)
    ]

    assert bins == expected_bins


def test_find_closest_patch_ultra_fine():
    """Unit test for closest patch in the ultra-fine Tregenza sphere."""

    angles = [
        (2, 70),
        (10, 40),
        (90, 180),
        (160, 358),
        (180, 30),
    ]

    sphere = tregenza_sphere.UltraFineTregenzaSphere()

    bins = [sphere.get_closest_face(phi, theta) for phi, theta in angles]

    expected_bins = [
        (2, 4),
        (7, 9),
        (62, 237),
        (109, 162),
        (123, 0)
    ]

    assert bins == expected_bins


def test_tregenza_plotting_fine(tmp_path):
    """Unit test for plotting the fine Tregenza sphere."""

    # Create the Tregenza sphere
    sphere = tregenza_sphere.FineTregenzaSphere()

    # Plot the sphere
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax = sphere.create_tregenza_plot(ax)
    fig.add_axes(ax)
    fig.savefig(os.path.join(tmp_path, "test_plot.png"))

    # Get the 3D patch collections
    patches = ax.collections[0].get_paths()

    # Check to make sure the number of patches is as expected
    assert len(patches) == sphere.patch_count.sum()

    plt.close(fig)


def test_tregenza_plotting_ultra_fine(tmp_path):
    """Unit test for plotting the ultra-fine Tregenza sphere."""

    # Create the Tregenza sphere
    sphere = tregenza_sphere.UltraFineTregenzaSphere()

    # Plot the sphere
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax = sphere.create_tregenza_plot(ax)
    fig.add_axes(ax)
    fig.savefig(os.path.join(tmp_path, "test_plot.png"))

    # Get the 3D patch collections
    patches = ax.collections[0].get_paths()

    # Check to make sure the number of patches is as expected
    assert len(patches) == sphere.patch_count.sum()

    plt.close(fig)