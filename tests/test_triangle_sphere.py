"""Tests for VectoRose triangle histogram plotting.

This module contains the automated tests for the
:mod:`vectorose.triangle_sphere` module, which provides the triangle-based
spherical histogram construction.
"""
import numpy as np
import pandas as pd
import trimesh.primitives

import vectorose as vr

from vectorose import mock_data

RANDOM_SEED = 20240827


def generate_test_vectors() -> np.ndarray:
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
    return vectors


def generate_vectors_every_face(
    sphere: vr.triangle_sphere.TriangleSphere,
) -> np.ndarray:
    """Generate a random vector in each sphere face.

    Construct a random vector in each triangular face of the sphere.

    Parameters
    ----------
    sphere
        Triangulated sphere to use to generate the random vectors.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n, 3)`` where ``n`` is the number of triangular
        faces in the sphere.

    Notes
    -----
    This process works by assuming that each triangle is defined by two
    vectors ``v1`` and ``v2`` with the same origin. Any point along the
    line connecting the ends of these vectors can be defined as
    ``v = a * v1 + (1 - a) * v2`` where ``0 <= a <= 1``. Any point inside
    the face can be expressed as ``v = b * (a * v1 + (1 - a) * v2)`` where
    ``0 <= b <= 1``.
    """

    # Get the sphere as a dataframe.
    faces = sphere.to_dataframe()

    # Get the origin of each face
    origin = faces[["x1", "y1", "z1"]].to_numpy()
    p1 = faces[["x2", "y2", "z2"]].to_numpy()
    p2 = faces[["x3", "y3", "z3"]].to_numpy()

    v1 = p1 - origin
    v2 = p2 - origin

    # Get the number of faces
    n = len(faces)

    # Generate the ``a`` randomly for each
    a = np.random.default_rng(RANDOM_SEED).uniform(low=1e-2, high=0.99, size=(n, 1))

    # Generate the ``b`` randomly for each
    b = np.random.default_rng(RANDOM_SEED).uniform(low=1e-2, high=0.99, size=(n, 1))

    # Create the random vectors
    in_face_vectors = b * (a * v1 + (1 - a) * v2)

    # Add these displacements to the origins
    vectors = origin + in_face_vectors

    # Compute unit vectors
    vectors, _ = vr.util.normalise_vectors(vectors)

    return vectors


def _test_assign_bins(subdivisions: int, number_of_shells: int):
    """Test the histogram bin assignment.

    Test of :meth:`TriangleSphere.assign_histogram_bins` to study that the
    bins are assigned correctly and that the output has the proper shape.

    Parameters
    ----------
    subdivisions
        Number of subdivisions for creating the sphere.
    number_of_shells
        Number of magnitude shells to consider.
    """

    # Generate sphere
    sphere = vr.triangle_sphere.TriangleSphere(subdivisions, number_of_shells)

    # Generate random vectors
    vectors = generate_vectors_every_face(sphere)
    number_of_vectors = len(vectors)

    # Generate random magnitudes
    magnitudes = np.random.default_rng(RANDOM_SEED).uniform(
        1e-6, 1, size=(number_of_vectors, 1)
    )
    vectors = magnitudes * vectors

    # Label the vectors
    labelled_vectors, magnitude_bin_edges = sphere.assign_histogram_bins(vectors)

    # Check the shape of the labelled bins
    assert len(labelled_vectors) == number_of_vectors

    # Check the shape of the bin edges
    assert len(magnitude_bin_edges) == number_of_shells + 1

    # Check the orientation assignments
    face_indices = labelled_vectors["face"].to_numpy()
    expected_indices = np.arange(number_of_vectors)

    assert np.all(face_indices == expected_indices)

    # Check the magnitude bin assignments
    min_shell = labelled_vectors["shell"].min()
    max_shell = labelled_vectors["shell"].max()

    assert min_shell <= max_shell < number_of_shells


def test_assign_bins_5_subdivisions_many_shells():
    """Test the orientation bin assignment.

    Test of :meth:`TriangleSphere.assign_histogram_bins` with five
    subdivisions and multiple shells.
    """
    subdivisions = 5
    number_of_shells = 32

    _test_assign_bins(subdivisions, number_of_shells)


def test_assign_bins_5_subdivisions_one_shell():
    """Test the orientation bin assignment.

    Test of :meth:`TriangleSphere.assign_histogram_bins` with five
    subdivisions and one shell.
    """
    subdivisions = 5
    number_of_shells = 1

    _test_assign_bins(subdivisions, number_of_shells)


def test_assign_bins_3_subdivisions_many_shells():
    """Test the orientation bin assignment.

    Test of :meth:`TriangleSphere.assign_histogram_bins` with three
    subdivisions and multiple shells.
    """
    subdivisions = 3
    number_of_shells = 32

    _test_assign_bins(subdivisions, number_of_shells)


def test_assign_bins_3_subdivisions_one_shell():
    """Test the orientation bin assignment.

    Test of :meth:`TriangleSphere.assign_histogram_bins` with three
    subdivisions and one shell.
    """
    subdivisions = 3
    number_of_shells = 1

    _test_assign_bins(subdivisions, number_of_shells)


def _test_create_mesh(subdivisions: int):
    """Test mesh creation.

    Test for :meth:`TriangleSphere.create_mesh`. Makes sure that the
    correct number of faces and vertices are included.

    Parameters
    ----------
    subdivisions
        Number of subdivisions to use when constructing the triangulated
        sphere.
    """

    sphere = vr.triangle_sphere.TriangleSphere(subdivisions)

    sphere_mesh = sphere.create_mesh()

    original_mesh = trimesh.primitives.Sphere(subdivisions=subdivisions)

    # Check the number of faces
    assert sphere_mesh.n_cells == len(original_mesh.faces)

    # Check the number of vertices
    assert sphere_mesh.n_points == len(original_mesh.vertices)

    # Get the degree of each face
    face_definitions = sphere_mesh.faces.reshape(-1, 4)

    assert np.all(face_definitions[:, 0] == 3)


def test_create_mesh_subdivision_3():
    """Test mesh creation for 3 subdivisions.

    Test for :meth:`TriangleSphere.create_mesh`. Makes sure that the
    correct number of faces and vertices are included.
    """
    subdivisions = 3

    _test_create_mesh(subdivisions)


def test_create_mesh_subdivision_5():
    """Test mesh creation for 5 subdivisions.

    Test for :meth:`TriangleSphere.create_mesh`. Makes sure that the
    correct number of faces and vertices are included.
    """
    subdivisions = 5

    _test_create_mesh(subdivisions)


def _test_construct_bivariate_histogram(
    subdivisions: int, number_of_shells: int, return_fraction: bool
):
    """Test the bivariate histogram construction.

    Test of :meth:`TriangleSphere.construct_histogram` to
    check the histogram construction for the triangulated sphere.

    Parameters
    ----------
    subdivisions
        Number of subdivisions to construct the triangulated sphere.
    number_of_shells
        Number of spherical shells to consider.
    return_fraction
        Indicate whether to consider frequencies or counts.
    """

    # Create a set of random vectors
    vectors = generate_test_vectors()
    number_of_vectors = len(vectors)

    # Construct a Tregenza sphere
    sphere = vr.triangle_sphere.TriangleSphere(
        number_of_subdivisions=subdivisions, number_of_shells=number_of_shells
    )

    # Compute the bin assignments
    labelled_vectors, magnitude_bin_edges = sphere.assign_histogram_bins(vectors)

    # Construct the histogram
    bivariate_histogram = sphere.construct_histogram(labelled_vectors, return_fraction)

    # Sum all the values
    frequency_sum = bivariate_histogram.sum()

    if return_fraction:
        expected_result = 1
        assert np.isclose(frequency_sum, expected_result)
    else:
        expected_result = number_of_vectors
        assert frequency_sum == expected_result


def test_construct_bivariate_histogram_counts_many_shells():
    """Test the bivariate histogram construction using counts.

    Test of :meth:`TriangleSphere.construct_histogram` to
    check the histogram construction for the triangulated sphere using
    count values and multiple shells. Five subdivisions considered.
    """

    number_of_subdivisions = 5
    number_of_shells = 32
    return_fraction = False

    _test_construct_bivariate_histogram(
        number_of_subdivisions, number_of_shells, return_fraction
    )


def test_construct_bivariate_histogram_frequencies_many_shells():
    """Test the bivariate histogram construction using frequencies.

    Test of :meth:`TriangleSphere.construct_histogram` to
    check the histogram construction for the triangulated sphere using
    frequency values and multiple shells. Five subdivisions considered.
    """

    number_of_subdivisions = 5
    number_of_shells = 32
    return_fraction = True

    _test_construct_bivariate_histogram(
        number_of_subdivisions, number_of_shells, return_fraction
    )


def test_construct_bivariate_histogram_counts_single_shell():
    """Test the bivariate histogram construction using counts.

    Test of :meth:`TriangleSphere.construct_histogram` to
    check the histogram construction for the triangulated sphere using
    count values and a single shell. Five subdivisions considered.
    """

    number_of_subdivisions = 5
    number_of_shells = 1
    return_fraction = False

    _test_construct_bivariate_histogram(
        number_of_subdivisions, number_of_shells, return_fraction
    )


def test_construct_bivariate_histogram_frequencies_single_shell():
    """Test the bivariate histogram construction using frequencies.

    Test of :meth:`TriangleSphere.construct_histogram` to
    check the histogram construction for the triangulated sphere using
    frequency values and a single shell. Five subdivisions considered.
    """

    number_of_subdivisions = 5
    number_of_shells = 1
    return_fraction = True

    _test_construct_bivariate_histogram(
        number_of_subdivisions, number_of_shells, return_fraction
    )


def _test_construct_marginal_magnitude_histogram(
    number_of_subdivisions: int, number_of_shells: int, return_fraction: bool
):
    """Test the magnitude histogram construction.

    Test of
    :meth:`TriangleSphere.construct_marginal_magnitude_histogram` to
    check the histogram construction for the triangulated sphere.

    Parameters
    ----------
    number_of_subdivisions
        Number of subdivisions to use when constructing the sphere.
    number_of_shells
        Number of spherical shells to consider.
    return_fraction
        Indicate whether to consider frequencies or counts.
    """

    # Create a set of random vectors
    vectors = generate_test_vectors()
    number_of_vectors = len(vectors)

    # Construct a Tregenza sphere
    sphere = vr.triangle_sphere.TriangleSphere(
        number_of_subdivisions=number_of_subdivisions, number_of_shells=number_of_shells
    )

    # Compute the bin assignments
    labelled_vectors, magnitude_bin_edges = sphere.assign_histogram_bins(vectors)

    # Construct the histogram
    magnitude_histogram = sphere.construct_marginal_magnitude_histogram(
        labelled_vectors, return_fraction
    )

    # Sum all the values
    frequency_sum = magnitude_histogram.sum()

    if return_fraction:
        expected_result = 1
        assert frequency_sum == expected_result
    else:
        expected_result = number_of_vectors
        assert np.isclose(frequency_sum, expected_result)


def test_construct_marginal_magnitude_histogram_counts_many_shells():
    """Test the magnitude histogram construction using counts.

    Test of
    :meth:`TriangleSphere.construct_marginal_magnitude_histogram` to
    check the histogram construction for the triangulated sphere.

    This test considers count values for a histogram with multiple shells.
    """

    number_of_subdivisions = 5
    number_of_shells = 32
    return_fraction = False

    _test_construct_marginal_magnitude_histogram(
        number_of_subdivisions, number_of_shells, return_fraction
    )


def test_construct_marginal_magnitude_histogram_frequencies_many_shells():
    """Test the magnitude histogram construction using frequencies.

    Test of
    :meth:`TriangleSphere.construct_marginal_magnitude_histogram` to
    check the histogram construction for the triangulated sphere.

    This test considers frequency values for a histogram with multiple
    shells.
    """

    number_of_subdivisions = 5
    number_of_shells = 32
    return_fraction = True

    _test_construct_marginal_magnitude_histogram(
        number_of_subdivisions, number_of_shells, return_fraction
    )


def test_construct_marginal_magnitude_histogram_counts_single_shell():
    """Test the magnitude histogram construction using counts.

    Test of
    :meth:`TriangleSphere.construct_marginal_magnitude_histogram` to
    check the histogram construction for the triangulated sphere.

    This test considers count values for a histogram with a single shell.
    """

    number_of_subdivisions = 5
    number_of_shells = 1
    return_fraction = False

    _test_construct_marginal_magnitude_histogram(
        number_of_subdivisions, number_of_shells, return_fraction
    )


def test_construct_marginal_magnitude_histogram_frequencies_single_shell():
    """Test the magnitude histogram construction using frequencies.

    Test of
    :meth:`TriangleSphere.construct_marginal_magnitude_histogram` to
    check the histogram construction for the triangulated sphere.

    This test considers frequency values for a histogram with a single
    shell.
    """

    number_of_subdivisions = 5
    number_of_shells = 1
    return_fraction = True

    _test_construct_marginal_magnitude_histogram(
        number_of_subdivisions, number_of_shells, return_fraction
    )


def _test_construct_marginal_orientation_histogram(
    number_of_subdivisions: int, number_of_shells: int, return_fraction: bool
):
    """Test the orientation histogram construction using counts.

    Test of
    :meth:`TriangleSphere.construct_marginal_orientation_histogram` to
    check the histogram construction for the triangulated sphere.

    Parameters
    ----------
    number_of_subdivisions
        Number of subdivisions to use when constructing the sphere.
    number_of_shells
        Number of spherical shells to consider.
    return_fraction
        Indicate whether to consider frequencies or counts.
    """

    # Create a set of random vectors
    vectors = generate_test_vectors()
    number_of_vectors = len(vectors)

    # Construct a Tregenza sphere
    sphere = vr.triangle_sphere.TriangleSphere(
        number_of_subdivisions=number_of_subdivisions, number_of_shells=number_of_shells
    )

    # Compute the bin assignments
    labelled_vectors, orientation_bin_edges = sphere.assign_histogram_bins(vectors)

    # Construct the histogram
    orientation_histogram = sphere.construct_marginal_orientation_histogram(
        labelled_vectors, return_fraction
    )

    # Sum all the values
    frequency_sum = orientation_histogram.sum()

    if return_fraction:
        expected_result = 1
        assert np.isclose(frequency_sum, expected_result)
    else:
        expected_result = number_of_vectors
        assert frequency_sum == expected_result


def test_construct_marginal_orientation_histogram_counts_many_shells():
    """Test the orientation histogram construction using counts.

    Test of
    :meth:`TriangleSphere.construct_marginal_orientation_histogram` to
    check the histogram construction for the triangulated sphere.

    This test considers count values and multiple histogram shells.
    """

    number_of_subdivisions = 5
    number_of_shells = 32
    return_fraction = False

    _test_construct_marginal_orientation_histogram(
        number_of_subdivisions, number_of_shells, return_fraction
    )


def test_construct_marginal_orientation_histogram_frequencies_many_shells():
    """Test the orientation histogram construction using frequencies.

    Test of
    :meth:`TriangleSphere.construct_marginal_orientation_histogram` to
    check the histogram construction for the triangulated sphere.

    This test considers frequency values and multiple histogram shells.
    """

    number_of_subdivisions = 5
    number_of_shells = 32
    return_fraction = True

    _test_construct_marginal_orientation_histogram(
        number_of_subdivisions, number_of_shells, return_fraction
    )


def test_construct_marginal_orientation_histogram_counts_single_shell():
    """Test the orientation histogram construction using counts.

    Test of
    :meth:`TriangleSphere.construct_marginal_orientation_histogram` to
    check the histogram construction for the triangulated sphere.

    This test considers count values and a single histogram shell.
    """

    number_of_subdivisions = 5
    number_of_shells = 1
    return_fraction = False

    _test_construct_marginal_orientation_histogram(
        number_of_subdivisions, number_of_shells, return_fraction
    )


def test_construct_marginal_orientation_histogram_frequencies_single_shell():
    """Test the orientation histogram construction using frequencies.

    Test of
    :meth:`TriangleSphere.construct_marginal_orientation_histogram` to
    check the histogram construction for the triangulated sphere.

    This test considers frequency values and a single histogram shell.
    """

    number_of_subdivisions = 5
    number_of_shells = 1
    return_fraction = True

    _test_construct_marginal_orientation_histogram(
        number_of_subdivisions, number_of_shells, return_fraction
    )


def _test_construct_conditional_orientation_histogram(
    number_of_subdivisions: int, number_of_shells: int
):
    """Test the conditional orientation histogram construction.

    Test of
    :meth:`TriangleSphere.construct_conditional_orientation_histogram` to
    check the histogram construction for the triangulated sphere.

    Parameters
    ----------
    number_of_subdivisions
        The number of subdivisions to perform when constructing the
        triangulated sphere.
    number_of_shells
        The number of histogram shells to consider.
    """

    # Create a set of random vectors
    vectors = generate_test_vectors()

    # Construct a Tregenza sphere
    sphere = vr.triangle_sphere.TriangleSphere(
        number_of_subdivisions=number_of_subdivisions, number_of_shells=number_of_shells
    )

    # Compute the bin assignments
    labelled_vectors, orientation_bin_edges = sphere.assign_histogram_bins(vectors)

    # Construct the histogram
    orientation_histogram = sphere.construct_conditional_orientation_histogram(
        labelled_vectors
    )

    # Sum all the values
    frequency_sum = orientation_histogram.groupby("shell").sum()

    # Some shells may have no vectors!
    shell_is_zero = frequency_sum == 0
    shell_is_approx_one = np.isclose(frequency_sum, 1)
    assert np.all(np.logical_or(shell_is_zero, shell_is_approx_one))


def test_construct_conditional_orientation_histogram_many_shells():
    """Test the conditional orientation histogram construction.

    Test of
    :meth:`TriangleSphere.construct_conditional_orientation_histogram` to
    check the histogram construction for the triangulated sphere.

    This test considers multiple orientation shells.
    """

    number_of_subdivisions = 5
    number_of_shells = 32

    _test_construct_conditional_orientation_histogram(
        number_of_subdivisions, number_of_shells
    )


def test_construct_conditional_orientation_histogram_single_shell():
    """Test the conditional orientation histogram construction.

    Test of
    :meth:`TriangleSphere.construct_conditional_orientation_histogram` to
    check the histogram construction for the triangulated sphere.

    This test considers a single orientation shell.
    """

    number_of_subdivisions = 5
    number_of_shells = 1

    _test_construct_conditional_orientation_histogram(
        number_of_subdivisions, number_of_shells
    )


def _test_construct_conditional_magnitude_histogram(
    number_of_subdivisions: int, number_of_shells: int
):
    """Test the conditional magnitude histogram construction.

    Test of
    :meth:`TriangleSphere.construct_conditional_magnitude_histogram` to
    check the histogram construction for the triangulated sphere.

    Parameters
    ----------
    number_of_subdivisions
        Number of subdivisions to use the construct the triangulated
        sphere.
    number_of_shells
        Number of histogram shells to construct.
    """

    # Create a set of random vectors
    vectors = generate_test_vectors()

    # Construct a Tregenza sphere
    sphere = vr.triangle_sphere.TriangleSphere(
        number_of_subdivisions=number_of_subdivisions, number_of_shells=number_of_shells
    )

    # Compute the bin assignments
    labelled_vectors, magnitude_bin_edges = sphere.assign_histogram_bins(vectors)

    # Construct the histogram
    magnitude_histogram = sphere.construct_conditional_magnitude_histogram(
        labelled_vectors
    )

    # Sum all the values
    frequency_sum = magnitude_histogram.groupby(["face"]).sum()

    # Some shells may have no vectors!
    bin_is_zero = frequency_sum == 0
    bin_is_approx_one = np.isclose(frequency_sum, 1)

    assert np.all(np.logical_or(bin_is_zero, bin_is_approx_one))


def test_construct_conditional_magnitude_histogram_many_shells():
    """Test the conditional magnitude histogram construction.

    Test of
    :meth:`TriangleSphere.construct_conditional_magnitude_histogram` to
    check the histogram construction for the triangulated sphere.

    This test considers many histogram shells.
    """

    number_of_subdivisions = 5
    number_of_shells = 32

    _test_construct_conditional_magnitude_histogram(
        number_of_subdivisions, number_of_shells
    )


def test_construct_conditional_magnitude_histogram_single_shell():
    """Test the conditional magnitude histogram construction.

    Test of
    :meth:`TriangleSphere.construct_conditional_magnitude_histogram` to
    check the histogram construction for the triangulated sphere.

    This test considers a single histogram shell.
    """

    number_of_subdivisions = 5
    number_of_shells = 1

    _test_construct_conditional_magnitude_histogram(
        number_of_subdivisions, number_of_shells
    )


def _test_create_histogram_meshes(
    number_of_subdivisions: int,
    number_of_shells: int,
    use_constant_radius: bool,
    normalise_by_shell: bool,
    use_frequencies: bool,
):
    """Test the histogram mesh creation.

    Test for :meth:`TriangleSphere.create_histogram_meshes` to assess
    whether the meshes are generated properly. A number of parameters can
    be varied to test for the number of shells, the radii, normalisation by
    shell and the use of counts vs frequencies.

    Parameters
    ----------
    number_of_subdivisions
        Number of subdivisions to use when constructing the triangulated
        sphere.
    number_of_shells
        Number of histogram shells to construct.
    use_constant_radius
        Indicate whether the shell meshes should have a consistent radius
        of 1.
    normalise_by_shell
        Indicate whether to normalise the shell intensities to the
        respective maximum values.
    use_frequencies
        Indicate whether to consider frequency data as opposed to count
        values. Ignored if `normalise_by_shell` is set to ``True``.
    """

    # Create a set of random vectors
    vectors = generate_test_vectors()

    # Construct a Tregenza sphere
    sphere = vr.triangle_sphere.TriangleSphere(
        number_of_subdivisions=number_of_subdivisions, number_of_shells=number_of_shells
    )

    # Compute the bin assignments
    labelled_vectors, magnitude_bin_edges = sphere.assign_histogram_bins(vectors)

    # Construct the histogram
    bivariate_histogram = sphere.construct_histogram(labelled_vectors, use_frequencies)

    if use_constant_radius:
        magnitude_bin_edges = None
        expected_radii = 1
    else:
        expected_radii = magnitude_bin_edges[1:]

    # Construct the shell meshes
    shell_meshes = sphere.create_histogram_meshes(
        histogram_data=bivariate_histogram,
        magnitude_bins=magnitude_bin_edges,
        normalise_by_shell=normalise_by_shell,
    )

    assert len(shell_meshes) == number_of_shells

    if normalise_by_shell:
        maxima = np.array([m.cell_data["frequency"].max() for m in shell_meshes])
        minima = np.array([m.cell_data["frequency"].min() for m in shell_meshes])

        assert np.all(np.logical_or(pd.isna(maxima), maxima <= 1))
        assert np.all(np.logical_or(pd.isna(maxima), maxima >= 0))
        assert np.all(np.logical_or(pd.isna(minima), minima >= 0))
        assert np.all(np.logical_or(pd.isna(minima), minima <= 0))
        assert np.all(np.logical_or(pd.isna(maxima), minima <= maxima))
    else:
        scalar_sum = np.sum([m.cell_data["frequency"].sum() for m in shell_meshes])

        if use_frequencies:
            assert np.isclose(scalar_sum, 1)
        else:
            number_of_vectors = len(vectors)
            assert scalar_sum == number_of_vectors

    # Check the radii
    radii = [np.max(m.bounds) for m in shell_meshes]

    assert np.all(np.isclose(radii, expected_radii))


def test_create_histogram_meshes_many_shells_no_norm_diff_radius_frequency():
    """Test the histogram mesh creation.

    Test for :meth:`TriangleSphere.create_histogram_meshes` to assess
    whether the meshes are generated properly.

    This test considers many shells with different radius, without shell
    normalisation, while considering frequency values.
    """

    number_of_subdivisions = 5
    number_of_shells = 32
    use_constant_radius = False
    normalise_by_shell = False
    use_frequencies = True

    _test_create_histogram_meshes(
        number_of_subdivisions,
        number_of_shells,
        use_constant_radius,
        normalise_by_shell,
        use_frequencies,
    )


def test_create_histogram_meshes_many_shells_no_norm_diff_radius_counts():
    """Test the histogram mesh creation.

    Test for :meth:`TriangleSphere.create_histogram_meshes` to assess
    whether the meshes are generated properly.

    This test considers many shells with different radius, without shell
    normalisation, while considering count values.
    """

    number_of_subdivisions = 5
    number_of_shells = 32
    use_constant_radius = False
    normalise_by_shell = False
    use_frequencies = False

    _test_create_histogram_meshes(
        number_of_subdivisions,
        number_of_shells,
        use_constant_radius,
        normalise_by_shell,
        use_frequencies,
    )


def test_create_histogram_meshes_many_shells_norm_diff_radius():
    """Test the histogram mesh creation.

    Test for :meth:`TriangleSphere.create_histogram_meshes` to assess
    whether the meshes are generated properly.

    This test considers many shells with different radius, with shell
    normalisation.
    """

    number_of_subdivisions = 5
    number_of_shells = 32
    use_constant_radius = False
    normalise_by_shell = True
    use_frequencies = False

    _test_create_histogram_meshes(
        number_of_subdivisions,
        number_of_shells,
        use_constant_radius,
        normalise_by_shell,
        use_frequencies,
    )


def test_create_histogram_meshes_many_shells_norm_same_radius():
    """Test the histogram mesh creation.

    Test for :meth:`TriangleSphere.create_histogram_meshes` to assess
    whether the meshes are generated properly.

    This test considers many shells with different radius, with shell
    normalisation.
    """

    number_of_subdivisions = 5
    number_of_shells = 32
    use_constant_radius = True
    normalise_by_shell = True
    use_frequencies = False

    _test_create_histogram_meshes(
        number_of_subdivisions,
        number_of_shells,
        use_constant_radius,
        normalise_by_shell,
        use_frequencies,
    )


def test_create_histogram_meshes_single_shell_no_norm_diff_radius_counts():
    """Test the histogram mesh creation.

    Test for :meth:`TriangleSphere.create_histogram_meshes` to assess
    whether the meshes are generated properly.

    This test considers a single shell without shell normalisation, while
    considering count values. The radius is not set to constant, and it
    will thus reflect the maximum value present in the dataset, not
    necessarily 1.
    """

    number_of_subdivisions = 5
    number_of_shells = 1
    use_constant_radius = False
    normalise_by_shell = False
    use_frequencies = False

    _test_create_histogram_meshes(
        number_of_subdivisions,
        number_of_shells,
        use_constant_radius,
        normalise_by_shell,
        use_frequencies,
    )


def test_create_histogram_meshes_single_shell_norm_diff_radius():
    """Test the histogram mesh creation.

    Test for :meth:`TriangleSphere.create_histogram_meshes` to assess
    whether the meshes are generated properly.

    This test considers a single shell with radius set to constant, with
    shell normalisation.
    """

    number_of_subdivisions = 5
    number_of_shells = 1
    use_constant_radius = True
    normalise_by_shell = True
    use_frequencies = False

    _test_create_histogram_meshes(
        number_of_subdivisions,
        number_of_shells,
        use_constant_radius,
        normalise_by_shell,
        use_frequencies,
    )


def test_convert_vectors_to_cartesian_array_non_unit():
    """Test the vector conversion to Cartesian array.

    Test for :meth:`TriangleSphere.convert_vectors_to_cartesian_array`
    which converts unit vectors and separate magnitudes coordinates to unit
    or non-unit vectors in Cartesian coordinates. In this test, the vectors
    are not normalised to unit length.
    """

    # Generate a collection of non-unit vectors
    vectors = generate_test_vectors()

    # Create a sphere
    sphere = vr.triangle_sphere.TriangleSphere(
        number_of_subdivisions=5, number_of_shells=32
    )

    # Perform the bin assignment on these vectors
    labelled_vectors, _ = sphere.assign_histogram_bins(vectors)

    # And now, convert the vectors back to Cartesian coordinates
    cartesian_coordinates = sphere.convert_vectors_to_cartesian_array(
        labelled_vectors, False
    )

    # Check to see if they are close enough to the originals
    assert np.all(np.isclose(cartesian_coordinates, vectors))


def test_convert_vectors_to_cartesian_array_normalised():
    """Test the vector conversion to Cartesian array.

    Test for :meth:`TriangleSphere.convert_vectors_to_cartesian_array`
    which converts unit vectors and separate magnitudes coordinates to unit
    or non-unit vectors in Cartesian coordinates. In this test, the vectors
    are normalised to unit length.
    """

    # Generate a collection of non-unit vectors
    vectors = generate_test_vectors()

    # Create a sphere
    sphere = vr.triangle_sphere.TriangleSphere(
        number_of_subdivisions=5, number_of_shells=32
    )

    # Perform the bin assignment on these vectors
    labelled_vectors, _ = sphere.assign_histogram_bins(vectors)

    # And now, convert the vectors back to Cartesian coordinates
    cartesian_coordinates = sphere.convert_vectors_to_cartesian_array(
        labelled_vectors, True
    )

    # Normalise the original vectors
    unit_vectors, _ = vr.util.normalise_vectors(vectors)

    # Check to see if they are close enough to the originals
    assert np.all(np.isclose(cartesian_coordinates, unit_vectors))
