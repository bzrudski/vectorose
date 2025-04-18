"""Tests for import/output operations.

This module contains the automated tests for the :mod:`vectorose.io` module
dedicated to import and export of vector field data.
"""
import os.path

import numpy as np
import pandas as pd
import pytest
import scipy as sp
import vectorose as vr

RANDOM_SEED = 20241205


def generate_random_vectors(l: int, w: int, d: int, flat: bool = True) -> np.ndarray:
    """Generate a collection of random vectors.

    Generate a collection of uniform vectors situated on an integer grid.

    Parameters
    ----------
    l
        Length of the grid.
    w
        Width of the grid.
    d
        Depth of the grid.
    flat
        Indicate whether to return a flat 2D array. Otherwise, a 4D array
        is returned, with the last axis corresponding to the components.

    Returns
    -------
    numpy.ndarray
        Array containing the generated vectors. If ``flat`` is ``True``,
        an array of shape ``(l * w * d, 6)`` is produced containing the
        spatial coordinates and the vector components. Otherwise, an array
        of shape ``(l, w, d, 3)`` is produced.
    """

    n = l * w * d

    # Create random vectors
    my_random_vectors = sp.stats.uniform_direction(dim=3, seed=RANDOM_SEED).rvs(size=n)

    # Define spatial locations
    my_spatial_locations = np.moveaxis(np.indices([l, w, d]).reshape(3, -1), 0, -1)

    if flat:
        # Concatenate the two together
        my_vector_array = np.concatenate(
            [my_spatial_locations, my_random_vectors], axis=-1
        )
    else:
        my_vector_array = my_random_vectors.reshape((l, w, d, 3))

    return my_vector_array

@pytest.fixture
def random_vectors_flat() -> np.ndarray:
    """Generate random vectors with locations for unit tests."""

    # Define the number of vectors
    l = 20
    w = 30
    d = 50

    my_vector_array = generate_random_vectors(l, w, d, flat=True)

    return my_vector_array

@pytest.fixture
def random_vectors_non_flat() -> np.ndarray:
    """Generate random vectors with locations for unit tests."""

    # Define the number of vectors
    l = 20
    w = 30
    d = 50

    my_vector_array = generate_random_vectors(l, w, d, flat=False)

    return my_vector_array

@pytest.fixture
def random_vectors_flat_nan(random_vectors_flat) -> np.ndarray:
    """Generate random vectors with a NaN entry."""
    # Add a new NaN vector
    nan_vector = np.array([0, 0, 0, 0, 0, np.nan])

    my_vector_array = np.vstack([random_vectors_flat, nan_vector])

    return my_vector_array


def test_vector_import_numpy_spatial_locations_flat(tmp_path, random_vectors_flat):
    """Test loading a vector field from a NumPy array."""

    # Save the random vectors
    vector_path = os.path.join(tmp_path, "my_vectors.npy")
    np.save(vector_path, random_vectors_flat)

    # Load the vectors
    my_loaded_vectors = vr.io.import_vector_field(
        vector_path, location_columns=[0, 1, 2], component_columns=[-3, -2, -1]
    )

    # Check the shape
    assert my_loaded_vectors.shape == random_vectors_flat.shape
    assert np.all(my_loaded_vectors == random_vectors_flat)


def test_vector_import_numpy_only_components_flat(tmp_path, random_vectors_flat):
    """Test loading a vector field from a NumPy array."""

    # Save the random vectors
    vector_path = os.path.join(tmp_path, "my_vectors.npy")
    np.save(vector_path, random_vectors_flat)

    # Get the original components
    my_vector_components = random_vectors_flat[:, -3:]

    # Load the vectors
    my_loaded_vectors = vr.io.import_vector_field(
        vector_path, location_columns=None, component_columns=[-3, -2, -1]
    )

    # Check the shape
    assert my_loaded_vectors.shape == my_vector_components.shape
    assert np.all(my_loaded_vectors == my_vector_components)


def test_vector_import_numpy_only_components_flat_with_nan(tmp_path, random_vectors_flat_nan):
    """Test loading a vector field from a NumPy array with NaN values."""

    # Save the random vectors
    vector_path = os.path.join(tmp_path, "my_vectors.npy")
    np.save(vector_path, random_vectors_flat_nan)

    # Get the non-NaN vectors and locations
    my_non_nan_vectors = random_vectors_flat_nan[:-1]

    # Load the vectors
    my_loaded_vectors = vr.io.import_vector_field(
        vector_path,
    )

    # Check the shape
    assert my_loaded_vectors.shape == my_non_nan_vectors.shape
    assert np.all(my_loaded_vectors == my_non_nan_vectors)

    # Make sure that none of the entries is NaN
    assert np.all(~np.isnan(my_loaded_vectors))


def test_vector_import_numpy_4d_array(tmp_path, random_vectors_non_flat):
    """Test loading a vector field from a 4D NumPy array."""

    # Save the random vectors
    vector_path = os.path.join(tmp_path, "my_vectors.npy")
    np.save(vector_path, random_vectors_non_flat)

    # Get the original components
    my_vector_components = random_vectors_non_flat.reshape(-1, 3)

    # Load the vectors
    my_loaded_vectors = vr.io.import_vector_field(vector_path)

    # Check the shape
    assert my_loaded_vectors.shape == my_vector_components.shape
    assert np.all(my_loaded_vectors == my_vector_components)


def test_vector_import_csv_comma_spatial_locations(tmp_path, random_vectors_flat):
    """Test loading a vector field from a CSV file with locations."""

    # Create a data frame with the vectors
    my_vector_data = pd.DataFrame(
        random_vectors_flat, columns=["x", "y", "z", "vx", "vy", "vz"]
    )

    # Save the random vectors
    vector_path = os.path.join(tmp_path, "my_vectors.csv")
    my_vector_data.to_csv(
        vector_path, sep=",", index=True, columns=["vy", "vz", "vx", "z", "x", "y"]
    )

    # Load the vectors
    my_loaded_vectors = vr.io.import_vector_field(
        vector_path,
        location_columns=[5, 6, 4],
        component_columns=[3, 1, 2],
        separator=",",
    )

    # Check the shape
    assert my_loaded_vectors.shape == random_vectors_flat.shape
    assert np.all(np.isclose(my_loaded_vectors, random_vectors_flat))


def test_vector_import_csv_comma_spatial_locations_nan(tmp_path, random_vectors_flat_nan):
    """Test loading from a CSV file with locations and NaN."""

    # Create a data frame with the vectors
    my_vector_data = pd.DataFrame(
        random_vectors_flat_nan, columns=["x", "y", "z", "vx", "vy", "vz"]
    )

    # Save the random vectors
    vector_path = os.path.join(tmp_path, "my_vectors.csv")
    my_vector_data.to_csv(
        vector_path, sep=",", index=True, columns=["vy", "vz", "vx", "z", "x", "y"]
    )

    # Load the vectors
    my_loaded_vectors = vr.io.import_vector_field(
        vector_path,
        location_columns=[5, 6, 4],
        component_columns=[3, 1, 2],
        separator=",",
    )

    # Check that there are no NaN values
    assert np.all(~np.isnan(my_loaded_vectors))


def test_vector_import_excel_only_components(tmp_path, random_vectors_flat):
    """Test loading a vector field from a CSV file with locations."""

    # Extract the components
    my_vector_components = random_vectors_flat[:, -3:]

    # Create a data frame with the vectors
    my_vector_data = pd.DataFrame(
        random_vectors_flat, columns=["x", "y", "z", "vx", "vy", "vz"]
    )

    # Save the random vectors
    vector_path = os.path.join(tmp_path, "my_vectors.xlsx")
    my_vector_data.to_excel(
        vector_path,
        sheet_name="VectorSheet",
        columns=["vx", "vy", "vz"],
        index=True,
        header=True,
    )

    # Load the vectors
    my_loaded_vectors = vr.io.import_vector_field(
        vector_path,
        component_columns=[1, 2, 3],
        contains_headers=True,
        sheet="VectorSheet",
    )

    # Check the shape
    assert my_loaded_vectors.shape == my_vector_components.shape
    assert np.all(np.isclose(my_loaded_vectors, my_vector_components))


def test_vector_import_excel_components_and_spatial(tmp_path, random_vectors_flat):
    """Test loading a vector field from a CSV file with locations."""

    # Create a data frame with the vectors
    my_vector_data = pd.DataFrame(
        random_vectors_flat, columns=["x", "y", "z", "vx", "vy", "vz"]
    )

    # Save the random vectors
    vector_path = os.path.join(tmp_path, "my_vectors.xlsx")
    my_vector_data.to_excel(
        vector_path,
        sheet_name="VectorSheet",
        index=False,
        header=True,
    )

    # Load the vectors
    my_loaded_vectors = vr.io.import_vector_field(
        vector_path,
        location_columns=[0, 1, 2],
        component_columns=[-3, -2, -1],
        contains_headers=True,
        sheet="VectorSheet",
    )

    # Check the shape
    assert my_loaded_vectors.shape == random_vectors_flat.shape
    assert np.all(np.isclose(my_loaded_vectors, random_vectors_flat))


def test_vector_import_excel_components_and_spatial_nan(tmp_path, random_vectors_flat_nan):
    """Test loading a vector field from a CSV file with locations."""

    # Create a data frame with the vectors
    my_vector_data = pd.DataFrame(
        random_vectors_flat_nan, columns=["x", "y", "z", "vx", "vy", "vz"]
    )

    # Save the random vectors
    vector_path = os.path.join(tmp_path, "my_vectors.xlsx")
    my_vector_data.to_excel(
        vector_path,
        sheet_name="VectorSheet",
        index=False,
        header=True,
    )

    # Load the vectors
    my_loaded_vectors = vr.io.import_vector_field(
        vector_path,
        location_columns=[0, 1, 2],
        component_columns=[-3, -2, -1],
        contains_headers=True,
        sheet="VectorSheet",
    )

    # Check the shape
    assert np.all(~np.isnan(my_loaded_vectors))
