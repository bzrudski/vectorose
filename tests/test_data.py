"""Tests for sample data loading.

This module contains unit tests for loading sample data files using the
module :mod:`vectorose.data`.
"""
import numpy as np
import pytest

import vectorose as vr
import vectorose.data

@pytest.mark.parametrize(
    ["dataset", "n_cols", "n_vectors"],
    [
        ("cluster_girdle", 3, 150_000),
        ("two_clusters", 3, 200_000),
        ("twisted_blocks", 6, 857_375)
    ]
)
def test_load_cluster_girdle(dataset, n_cols, n_vectors):
    """Test loading the sample datasets."""

    # Load the dataset
    sample_data = vr.data.SampleData(dataset).load()

    # Ensure that it has loaded properly
    assert isinstance(sample_data, np.ndarray)
    loaded_vectors, loaded_columns = sample_data.shape
    assert loaded_vectors == n_vectors
    assert loaded_columns == n_cols
