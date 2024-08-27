"""Tests for VectoRose triangle histogram plotting.

This module contains the automated tests for the
:mod:`vectorose.triangle_sphere` module, which provides the triangle-based
spherical histogram construction.
"""
import numpy as np

from vectorose import triangle_sphere, mock_data

RANDOM_SEED = 20240827


def generate_test_vectors() -> np.ndarray:
    """Generate test vectors for the unit tests."""

    vectors = mock_data.create_vonmises_fisher_vectors_single_direction(
        phi=50, theta=60, kappa=10, number_of_points=100_000, magnitude=5,
        magnitude_std=0, use_degrees=True, seed=RANDOM_SEED
    )
    return vectors


def test_construct_spherical_histogram_magnitude_weight():
    """Test the spherical histogram construct, weighted by magnitude.

    Unit test for :func:`triangle_sphere.construct_spherical_histogram` to
    check the magnitude-weighted histogram construction.
    """

    # Create a set of random vectors
    vectors = generate_test_vectors()

    # Now, construct a magnitude-weighted histogram.
    sphere, face_values = triangle_sphere.construct_spherical_histogram(
        vectors, weight_by_magnitude=True
    )

    # Now, check that the number of counts corresponds to the number of
    # faces in the sphere.
    assert len(sphere.faces) == len(face_values)

    # Check that the sum of counts corresponds to the sum of magnitudes
    sum_of_face_values = face_values.sum()
    total_magnitude = np.linalg.norm(vectors, axis=-1).sum()

    assert sum_of_face_values == total_magnitude


def test_construct_spherical_histogram_count_weight():
    """Test the spherical histogram construct, weighted by count.

    Unit test for :func:`triangle_sphere.construct_spherical_histogram` to
    check the count-weighted histogram construction.
    """

    # Create a set of random vectors
    vectors = generate_test_vectors()

    # Now, construct a magnitude-weighted histogram.
    sphere, face_values = triangle_sphere.construct_spherical_histogram(
        vectors, weight_by_magnitude=False
    )

    # Now, check that the number of counts corresponds to the number of
    # faces in the sphere.
    assert len(sphere.faces) == len(face_values)

    # Check that the sum of counts corresponds to the total vector count
    sum_of_face_values = face_values.sum()

    assert sum_of_face_values == len(vectors)
