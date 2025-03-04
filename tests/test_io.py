"""Tests for import/output operations.

This module contains the automated tests for the :mod:`vectorose.io` module
dedicated to import and export of vector field data.
"""
import os.path

import numpy as np
import pandas as pd
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


def test_vector_import_numpy_spatial_locations_flat(tmp_path):
    """Test loading a vector field from a NumPy array."""

    # Define the number of vectors
    l = w = d = 100

    my_vector_array = generate_random_vectors(l, w, d, flat=True)

    # Save the random vectors
    vector_path = os.path.join(tmp_path, "my_vectors.npy")
    np.save(vector_path, my_vector_array)

    # Load the vectors
    my_loaded_vectors = vr.io.import_vector_field(
        vector_path, location_columns=[0, 1, 2], component_columns=[-3, -2, -1]
    )

    # Check the shape
    assert my_loaded_vectors.shape == my_vector_array.shape
    assert np.all(my_loaded_vectors == my_vector_array)


def test_vector_import_numpy_only_components_flat(tmp_path):
    """Test loading a vector field from a NumPy array."""

    # Define the number of vectors
    l = w = d = 100

    my_vector_array = generate_random_vectors(l, w, d, flat=True)

    # Save the random vectors
    vector_path = os.path.join(tmp_path, "my_vectors.npy")
    np.save(vector_path, my_vector_array)

    # Get the original components
    my_vector_components = my_vector_array[:, -3:]

    # Load the vectors
    my_loaded_vectors = vr.io.import_vector_field(
        vector_path, location_columns=None, component_columns=[-3, -2, -1]
    )

    # Check the shape
    assert my_loaded_vectors.shape == my_vector_components.shape
    assert np.all(my_loaded_vectors == my_vector_components)


def test_vector_import_numpy_4d_array(tmp_path):
    """Test loading a vector field from a 4D NumPy array."""

    # Define the number of vectors
    l = 100
    w = 120
    d = 80

    my_vector_array = generate_random_vectors(l, w, d, flat=False)

    # Save the random vectors
    vector_path = os.path.join(tmp_path, "my_vectors.npy")
    np.save(vector_path, my_vector_array)

    # Get the original components
    my_vector_components = my_vector_array.reshape(-1, 3)

    # Load the vectors
    my_loaded_vectors = vr.io.import_vector_field(vector_path)

    # Check the shape
    assert my_loaded_vectors.shape == my_vector_components.shape
    assert np.all(my_loaded_vectors == my_vector_components)


def test_vector_import_csv_comma_spatial_locations(tmp_path):
    """Test loading a vector field from a CSV file with locations."""

    # Define the number of vectors
    l = w = d = 100

    my_vector_array = generate_random_vectors(l, w, d, flat=True)

    # Create a data frame with the vectors
    my_vector_data = pd.DataFrame(
        my_vector_array, columns=["x", "y", "z", "vx", "vy", "vz"]
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
    assert my_loaded_vectors.shape == my_vector_array.shape
    assert np.all(np.isclose(my_loaded_vectors, my_vector_array))


def test_vector_import_excel_only_components(tmp_path):
    """Test loading a vector field from a CSV file with locations."""

    # Define the number of vectors
    l = 20
    w = 30
    d = 50

    my_vector_array = generate_random_vectors(l, w, d, flat=True)

    # Extract the components
    my_vector_components = my_vector_array[:, -3:]

    # Create a data frame with the vectors
    my_vector_data = pd.DataFrame(
        my_vector_array, columns=["x", "y", "z", "vx", "vy", "vz"]
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
