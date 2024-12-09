"""Tests for VectoRose Tregenza histogram plotting.

This module contains the automated tests for the
:mod:`vectorose.tregenza_sphere` module, which provides the Tregenza-based
spherical histogram construction.
"""

import numpy as np
import pandas as pd

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


# Test the geometry of the spheres
def test_coarse_tregenza_sphere_mesh():
    """Test the coarse Tregenza sphere mesh.

    Test of :meth:`CoarseTregenzaSphere.create_mesh`. Ensures that the
    number of faces indeed corresponds with the expected count.
    """

    sphere = vr.tregenza_sphere.CoarseTregenzaSphere()
    mesh = sphere.create_mesh()
    rings = sphere.to_dataframe()

    expected_total_number_of_patches = rings["bins"].sum()

    # Test the number of faces
    assert mesh.n_cells == expected_total_number_of_patches == 520


def test_fine_tregenza_sphere_mesh():
    """Test the fine Tregenza sphere mesh.

    Test of :meth:`FineTregenzaSphere.create_mesh`. Ensures that the number
    of faces indeed corresponds with the expected count.
    """

    sphere = vr.tregenza_sphere.FineTregenzaSphere()
    mesh = sphere.create_mesh()
    rings = sphere.to_dataframe()

    expected_total_number_of_patches = rings["bins"].sum()

    # Test the number of faces
    assert mesh.n_cells == expected_total_number_of_patches == 5806


def test_ultra_fine_tregenza_sphere_mesh():
    """Test the ultra-fine Tregenza sphere mesh.

    Test of :meth:`UltraFineTregenzaSphere.create_mesh`. Ensures that the
    number of faces indeed corresponds with the expected count.
    """

    sphere = vr.tregenza_sphere.UltraFineTregenzaSphere()
    mesh = sphere.create_mesh()
    rings = sphere.to_dataframe()

    expected_total_number_of_patches = rings["bins"].sum()

    # Test the number of faces
    assert mesh.n_cells == expected_total_number_of_patches == 36956


# Test the orientation bin assignment
def test_get_closest_faces_coarse():
    """Test orientation bin assignment in the Coarse Tregenza sphere.

    Test of :meth:`CoarseTregenzaSphere.get_closest_faces`. Ensures that
    the sphere indeed assigns the vectors to the appropriate bin.
    """

    angles = [
        (2, 70),
        (10, 40),
        (90, 180),
        (160, 358),
        (180, 30),
    ]

    spherical_coordinates = pd.DataFrame(angles, columns=["phi", "theta"])

    sphere = vr.tregenza_sphere.CoarseTregenzaSphere()

    bins = sphere.get_closest_faces(spherical_coordinates)

    expected_bins = pd.DataFrame(
        [(0, 0), (1, 0), (9, 25), (15, 14), (17, 0)], columns=["ring", "bin"]
    )

    assert np.all(bins == expected_bins)


def test_get_closest_faces_fine():
    """Test orientation bin assignment in the Fine Tregenza sphere.

    Test of :meth:`FineTregenzaSphere.get_closest_faces`. Ensures that the
    sphere indeed assigns the vectors to the appropriate bin.
    """

    angles = [
        (2, 70),
        (10, 40),
        (90, 180),
        (160, 358),
        (180, 30),
    ]

    spherical_coordinates = pd.DataFrame(angles, columns=["phi", "theta"])

    sphere = vr.tregenza_sphere.FineTregenzaSphere()

    bins = sphere.get_closest_faces(spherical_coordinates)

    expected_bins = pd.DataFrame(
        [(1, 1), (3, 3), (27, 86), (47, 57), (53, 0)], columns=["ring", "bin"]
    )

    assert np.all(bins == expected_bins)


def test_get_closest_faces_ultra_fine():
    """Test orientation bin assignment in the Ultra-Fine Tregenza sphere.

    Test of :meth:`UltraFineTregenzaSphere.get_closest_faces`. Ensures that
    the sphere indeed assigns the vectors to the appropriate bin.
    """

    angles = [
        (2, 70),
        (10, 40),
        (90, 180),
        (160, 358),
        (180, 30),
    ]

    spherical_coordinates = pd.DataFrame(angles, columns=["phi", "theta"])

    sphere = vr.tregenza_sphere.UltraFineTregenzaSphere()

    bins = sphere.get_closest_faces(spherical_coordinates)

    expected_bins = pd.DataFrame(
        [(2, 4), (7, 9), (62, 237), (109, 162), (123, 0)], columns=["ring", "bin"]
    )

    assert np.all(bins == expected_bins)


def _test_assign_bins(number_of_shells: int):
    """Test the histogram bin assignment.

    Test of :meth:`TregenzaSphere.assign_histogram_bins` to study the
    shape of the output.

    Parameters
    ----------
    number_of_shells
        The number of histogram shells to consider.
    """
    # Create the vectors
    vectors = generate_test_vectors()
    number_of_vectors = len(vectors)

    # Create the sphere
    sphere = vr.tregenza_sphere.FineTregenzaSphere(number_of_shells=number_of_shells)

    # Assign the histogram bins
    labelled_vectors, magnitude_bin_edges = sphere.assign_histogram_bins(vectors)

    # Check the geometry of the labelled vectors
    assert len(labelled_vectors) == number_of_vectors

    # Check the assigned labels to make sure they're in a valid range
    min_shell = labelled_vectors["shell"].min()
    max_shell = labelled_vectors["shell"].max()
    assert min_shell <= max_shell < number_of_shells

    # And now check the rings
    number_of_rings = sphere.number_of_rings
    min_ring = labelled_vectors["ring"].min()
    max_ring = labelled_vectors["ring"].max()
    assert min_ring <= max_ring < number_of_rings

    # And now, check each ring
    rings = sphere.to_dataframe()
    number_of_bins = rings["bins"]
    for i in range(number_of_rings):
        bin_count = number_of_bins.iloc[i]
        bins_for_ring = labelled_vectors[labelled_vectors["ring"] == i]

        if len(bins_for_ring) == 0:
            continue

        min_bin = bins_for_ring["bin"].min()
        max_bin = bins_for_ring["bin"].max()

        assert min_bin <= max_bin < bin_count

    # Check the number of magnitude bin edges
    assert len(magnitude_bin_edges) == number_of_shells + 1


def test_assign_bins_many_bins():
    """Test the histogram bin assignment.

    Test of :meth:`TregenzaSphere.assign_histogram_bins` to study the
    shape of the output. This test considers multiple histogram shells.
    """
    number_of_shells = 32

    _test_assign_bins(number_of_shells)


def test_assign_bins_single_bins():
    """Test the histogram bin assignment.

    Test of :meth:`TregenzaSphere.assign_histogram_bins` to study the
    shape of the output. This test considers only one histogram shell.
    """
    number_of_shells = 1

    _test_assign_bins(number_of_shells)


def _test_construct_bivariate_histogram(number_of_shells: int, return_fraction: bool):
    """Test the bivariate histogram construction.

    Test of :meth:`TregenzaSphere.construct_histogram` to
    check the histogram construction for the fine Tregenza sphere.

    Parameters
    ----------
    number_of_shells
        Number of spherical shells to consider.
    return_fraction
        Indicate whether to consider frequencies or counts.
    """

    # Create a set of random vectors
    vectors = generate_test_vectors()
    number_of_vectors = len(vectors)

    # Construct a Tregenza sphere
    sphere = vr.tregenza_sphere.FineTregenzaSphere(number_of_shells=number_of_shells)

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

    Test of :meth:`TregenzaSphere.construct_histogram` to
    check the histogram construction for the fine Tregenza sphere using
    count values and multiple shells.
    """

    number_of_shells = 32
    return_fraction = False

    _test_construct_bivariate_histogram(number_of_shells, return_fraction)


def test_construct_bivariate_histogram_frequencies_many_shells():
    """Test the bivariate histogram construction using frequencies.

    Test of :meth:`TregenzaSphere.construct_histogram` to
    check the histogram construction for the fine Tregenza sphere using
    frequency values and multiple shells.
    """

    number_of_shells = 32
    return_fraction = True

    _test_construct_bivariate_histogram(number_of_shells, return_fraction)


def test_construct_bivariate_histogram_counts_single_shell():
    """Test the bivariate histogram construction using counts.

    Test of :meth:`TregenzaSphere.construct_histogram` to
    check the histogram construction for the fine Tregenza sphere using
    count values and a single shell.
    """

    number_of_shells = 1
    return_fraction = False

    _test_construct_bivariate_histogram(number_of_shells, return_fraction)


def test_construct_bivariate_histogram_frequencies_single_shell():
    """Test the bivariate histogram construction using frequencies.

    Test of :meth:`TregenzaSphere.construct_histogram` to
    check the histogram construction for the fine Tregenza sphere using
    frequency values and a single shell.
    """

    number_of_shells = 1
    return_fraction = True

    _test_construct_bivariate_histogram(number_of_shells, return_fraction)


def _test_construct_marginal_magnitude_histogram(
    number_of_shells: int, return_fraction: bool
):
    """Test the magnitude histogram construction.

    Test of
    :meth:`TregenzaSphere.construct_marginal_magnitude_histogram` to
    check the histogram construction for the fine Tregenza sphere.

    Parameters
    ----------
    number_of_shells
        Number of spherical shells to consider.
    return_fraction
        Indicate whether to consider frequencies or counts.
    """

    # Create a set of random vectors
    vectors = generate_test_vectors()
    number_of_vectors = len(vectors)

    # Construct a Tregenza sphere
    sphere = vr.tregenza_sphere.FineTregenzaSphere(number_of_shells=number_of_shells)

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
    :meth:`TregenzaSphere.construct_marginal_magnitude_histogram` to
    check the histogram construction for the fine Tregenza sphere.

    This test considers count values for a histogram with multiple shells.
    """

    number_of_shells = 32
    return_fraction = False

    _test_construct_marginal_magnitude_histogram(number_of_shells, return_fraction)


def test_construct_marginal_magnitude_histogram_frequencies_many_shells():
    """Test the magnitude histogram construction using frequencies.

    Test of
    :meth:`TregenzaSphere.construct_marginal_magnitude_histogram` to
    check the histogram construction for the fine Tregenza sphere.

    This test considers frequency values for a histogram with multiple
    shells.
    """

    number_of_shells = 32
    return_fraction = True

    _test_construct_marginal_magnitude_histogram(number_of_shells, return_fraction)


def test_construct_marginal_magnitude_histogram_counts_single_shell():
    """Test the magnitude histogram construction using counts.

    Test of
    :meth:`TregenzaSphere.construct_marginal_magnitude_histogram` to
    check the histogram construction for the fine Tregenza sphere.

    This test considers count values for a histogram with a single shell.
    """

    number_of_shells = 1
    return_fraction = False

    _test_construct_marginal_magnitude_histogram(number_of_shells, return_fraction)


def test_construct_marginal_magnitude_histogram_frequencies_single_shell():
    """Test the magnitude histogram construction using frequencies.

    Test of
    :meth:`TregenzaSphere.construct_marginal_magnitude_histogram` to
    check the histogram construction for the fine Tregenza sphere.

    This test considers frequency values for a histogram with a single
    shell.
    """

    number_of_shells = 1
    return_fraction = True

    _test_construct_marginal_magnitude_histogram(number_of_shells, return_fraction)


def _test_construct_marginal_orientation_histogram(
    number_of_shells: int, return_fraction: bool
):
    """Test the orientation histogram construction using counts.

    Test of
    :meth:`TregenzaSphere.construct_marginal_orientation_histogram` to
    check the histogram construction for the fine Tregenza sphere."""

    # Create a set of random vectors
    vectors = generate_test_vectors()
    number_of_vectors = len(vectors)

    # Construct a Tregenza sphere
    sphere = vr.tregenza_sphere.FineTregenzaSphere(number_of_shells=number_of_shells)

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
    :meth:`TregenzaSphere.construct_marginal_orientation_histogram` to
    check the histogram construction for the fine Tregenza sphere.

    This test considers count values and multiple histogram shells.
    """

    number_of_shells = 32
    return_fraction = False

    _test_construct_marginal_orientation_histogram(number_of_shells, return_fraction)


def test_construct_marginal_orientation_histogram_frequencies_many_shells():
    """Test the orientation histogram construction using frequencies.

    Test of
    :meth:`TregenzaSphere.construct_marginal_orientation_histogram` to
    check the histogram construction for the fine Tregenza sphere.

    This test considers frequency values and multiple histogram shells.
    """

    number_of_shells = 32
    return_fraction = True

    _test_construct_marginal_orientation_histogram(number_of_shells, return_fraction)


def test_construct_marginal_orientation_histogram_counts_single_shell():
    """Test the orientation histogram construction using counts.

    Test of
    :meth:`TregenzaSphere.construct_marginal_orientation_histogram` to
    check the histogram construction for the fine Tregenza sphere.

    This test considers count values and a single histogram shell.
    """

    number_of_shells = 1
    return_fraction = False

    _test_construct_marginal_orientation_histogram(number_of_shells, return_fraction)


def test_construct_marginal_orientation_histogram_frequencies_single_shell():
    """Test the orientation histogram construction using frequencies.

    Test of
    :meth:`TregenzaSphere.construct_marginal_orientation_histogram` to
    check the histogram construction for the fine Tregenza sphere.

    This test considers frequency values and a single histogram shell.
    """

    number_of_shells = 1
    return_fraction = True

    _test_construct_marginal_orientation_histogram(number_of_shells, return_fraction)


def _test_construct_conditional_orientation_histogram(number_of_shells: int):
    """Test the conditional orientation histogram construction.

    Test of
    :meth:`TregenzaSphere.construct_conditional_orientation_histogram` to
    check the histogram construction for the fine Tregenza sphere.

    Parameters
    ----------
    number_of_shells
        The number of histogram shells to consider.
    """

    # Create a set of random vectors
    vectors = generate_test_vectors()

    # Construct a Tregenza sphere
    sphere = vr.tregenza_sphere.FineTregenzaSphere(number_of_shells=number_of_shells)

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
    :meth:`TregenzaSphere.construct_conditional_orientation_histogram` to
    check the histogram construction for the fine Tregenza sphere.

    This test considers multiple orientation shells.
    """

    number_of_shells = 32

    _test_construct_conditional_orientation_histogram(number_of_shells)


def test_construct_conditional_orientation_histogram_single_shell():
    """Test the conditional orientation histogram construction.

    Test of
    :meth:`TregenzaSphere.construct_conditional_orientation_histogram` to
    check the histogram construction for the fine Tregenza sphere.

    This test considers a single orientation shell.
    """

    number_of_shells = 1

    _test_construct_conditional_orientation_histogram(number_of_shells)


def _test_construct_conditional_magnitude_histogram(number_of_shells: int):
    """Test the conditional magnitude histogram construction.

    Test of
    :meth:`TregenzaSphere.construct_conditional_magnitude_histogram` to
    check the histogram construction for the fine Tregenza sphere.

    Parameters
    ----------
    number_of_shells
        Number of histogram shells to construct.
    """

    # Create a set of random vectors
    vectors = generate_test_vectors()

    # Construct a Tregenza sphere
    sphere = vr.tregenza_sphere.FineTregenzaSphere(number_of_shells=number_of_shells)

    # Compute the bin assignments
    labelled_vectors, magnitude_bin_edges = sphere.assign_histogram_bins(vectors)

    # Construct the histogram
    magnitude_histogram = sphere.construct_conditional_magnitude_histogram(
        labelled_vectors
    )

    # Sum all the values
    frequency_sum = magnitude_histogram.groupby(["ring", "bin"]).sum()

    # Some shells may have no vectors!
    bin_is_zero = frequency_sum == 0
    bin_is_approx_one = np.isclose(frequency_sum, 1)

    assert np.all(np.logical_or(bin_is_zero, bin_is_approx_one))


def test_construct_conditional_magnitude_histogram_many_shells():
    """Test the conditional magnitude histogram construction.

    Test of
    :meth:`TregenzaSphere.construct_conditional_magnitude_histogram` to
    check the histogram construction for the fine Tregenza sphere.

    This test considers many histogram shells.
    """

    number_of_shells = 32

    _test_construct_conditional_magnitude_histogram(number_of_shells)


def test_construct_conditional_magnitude_histogram_single_shell():
    """Test the conditional magnitude histogram construction.

    Test of
    :meth:`TregenzaSphere.construct_conditional_magnitude_histogram` to
    check the histogram construction for the fine Tregenza sphere.

    This test considers a single histogram shell.
    """

    number_of_shells = 1

    _test_construct_conditional_magnitude_histogram(number_of_shells)


def _test_create_histogram_meshes(
    number_of_shells: int,
    use_constant_radius: bool,
    normalise_by_shell: bool,
    use_frequencies: bool,
):
    """Test the histogram mesh creation.

    Test for :meth:`TregenzaSphere.create_histogram_meshes` to assess
    whether the meshes are generated properly. A number of parameters can
    be varied to test for the number of shells, the radii, normalisation by
    shell and the use of counts vs frequencies.

    Parameters
    ----------
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
    sphere = vr.tregenza_sphere.FineTregenzaSphere(number_of_shells=number_of_shells)

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

        assert np.all(maxima <= 1)
        assert np.all(maxima >= 0)
        assert np.all(minima >= 0)
        assert np.all(minima <= 0)
        assert np.all(minima <= maxima)
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

    Test for :meth:`TregenzaSphere.create_histogram_meshes` to assess
    whether the meshes are generated properly.

    This test considers many shells with different radius, without shell
    normalisation, while considering frequency values.
    """

    number_of_shells = 32
    use_constant_radius = False
    normalise_by_shell = False
    use_frequencies = True

    _test_create_histogram_meshes(
        number_of_shells, use_constant_radius, normalise_by_shell, use_frequencies
    )


def test_create_histogram_meshes_many_shells_no_norm_diff_radius_counts():
    """Test the histogram mesh creation.

    Test for :meth:`TregenzaSphere.create_histogram_meshes` to assess
    whether the meshes are generated properly.

    This test considers many shells with different radius, without shell
    normalisation, while considering count values.
    """

    number_of_shells = 32
    use_constant_radius = False
    normalise_by_shell = False
    use_frequencies = False

    _test_create_histogram_meshes(
        number_of_shells, use_constant_radius, normalise_by_shell, use_frequencies
    )


def test_create_histogram_meshes_many_shells_norm_diff_radius():
    """Test the histogram mesh creation.

    Test for :meth:`TregenzaSphere.create_histogram_meshes` to assess
    whether the meshes are generated properly.

    This test considers many shells with different radius, with shell
    normalisation.
    """

    number_of_shells = 32
    use_constant_radius = False
    normalise_by_shell = True
    use_frequencies = False

    _test_create_histogram_meshes(
        number_of_shells, use_constant_radius, normalise_by_shell, use_frequencies
    )


def test_create_histogram_meshes_many_shells_norm_same_radius():
    """Test the histogram mesh creation.

    Test for :meth:`TregenzaSphere.create_histogram_meshes` to assess
    whether the meshes are generated properly.

    This test considers many shells with different radius, with shell
    normalisation.
    """

    number_of_shells = 32
    use_constant_radius = True
    normalise_by_shell = True
    use_frequencies = False

    _test_create_histogram_meshes(
        number_of_shells, use_constant_radius, normalise_by_shell, use_frequencies
    )


def test_create_histogram_meshes_single_shell_no_norm_diff_radius_counts():
    """Test the histogram mesh creation.

    Test for :meth:`TregenzaSphere.create_histogram_meshes` to assess
    whether the meshes are generated properly.

    This test considers a single shell without shell normalisation, while
    considering count values. The radius is not set to constant, and it
    will thus reflect the maximum value present in the dataset, not
    necessarily 1.
    """

    number_of_shells = 1
    use_constant_radius = False
    normalise_by_shell = False
    use_frequencies = False

    _test_create_histogram_meshes(
        number_of_shells, use_constant_radius, normalise_by_shell, use_frequencies
    )


def test_create_histogram_meshes_single_shell_norm_diff_radius():
    """Test the histogram mesh creation.

    Test for :meth:`TregenzaSphere.create_histogram_meshes` to assess
    whether the meshes are generated properly.

    This test considers a single shell with radius set to constant, with
    shell normalisation.
    """

    number_of_shells = 1
    use_constant_radius = True
    normalise_by_shell = True
    use_frequencies = False

    _test_create_histogram_meshes(
        number_of_shells, use_constant_radius, normalise_by_shell, use_frequencies
    )


def test_convert_vectors_to_cartesian_array_non_unit():
    """Test the vector conversion to Cartesian array.

    Test for :meth:`TregenzaSphere.convert_vectors_to_cartesian_array`
    which converts vectors in spherical coordinates to vectors in Cartesian
    coordinates. In this test, the vectors are not normalised to unit
    length.
    """

    # Generate a collection of non-unit vectors
    vectors = generate_test_vectors()

    # Create a sphere
    sphere = vr.tregenza_sphere.FineTregenzaSphere(number_of_shells=32)

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

    Test for :meth:`TregenzaSphere.convert_vectors_to_cartesian_array`
    which converts vectors in spherical coordinates to vectors in Cartesian
    coordinates. In this test, the vectors are normalised to unit length.
    """

    # Generate a collection of non-unit vectors
    vectors = generate_test_vectors()

    # Create a sphere
    sphere = vr.tregenza_sphere.FineTregenzaSphere(number_of_shells=32)

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


def test_correct_histogram_by_area():
    """Test the Tregenza Sphere face area correction.

    Test for :meth:`TregenzaSphere.correct_histogram_by_area` which
    corrects the computed histogram result by a weighting associated with
    the face areas.
    """

    # Create vectors
    vectors = generate_test_vectors()

    # Create a sphere
    number_of_shells = 32
    sphere = vr.tregenza_sphere.FineTregenzaSphere(number_of_shells)

    # Get the face area weights
    sphere_dataframe = sphere.to_dataframe()
    weights = sphere_dataframe["weight"]

    # Make sure the weights make sense
    assert weights.min() >= 0
    assert weights.max() <= 1

    # Assign the bins
    labelled_vectors, _ = sphere.assign_histogram_bins(vectors)

    # Construct the bivariate histogram
    bivariate_histogram = sphere.construct_histogram(labelled_vectors, False)

    # Now, perform the correction
    corrected_histogram = sphere.correct_histogram_by_area(bivariate_histogram)

    # Make sure that the values are no larger than they were before
    assert np.all(corrected_histogram <= bivariate_histogram)

    # Make sure that the values are indeed corrected by the weight
    for shell in range(number_of_shells):
        for ring in range(sphere.number_of_rings):
            original_histogram_values = bivariate_histogram[shell, ring]
            corrected_histogram_values = corrected_histogram[shell, ring]
            ring_weight = weights.iloc[ring]

            ratio = corrected_histogram_values / original_histogram_values

            if np.all(original_histogram_values == 0):
                continue

            assert np.all(np.logical_or(pd.isna(ratio), np.isclose(ratio, ring_weight)))

