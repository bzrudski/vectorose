# Copyright (c) 2024-, Benjamin Rudski, Joseph Deering
#
# This code is licensed under the MIT License. See the `LICENSE` file for
# more details about copying.

"""Triangle-based Sphere Plotting.

This module provides the functions necessary to produce a triangle mesh of
a sphere, with face colours corresponding to the point count in each face.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import trimesh

from . import util


class TriangleSphere:
    """Representation of a sphere constructed using equal-area triangles.

    Compute and visualise histograms using a sphere composed of equal-area
    triangles.
    """

    # Attributes
    _sphere: trimesh.primitives.Sphere
    """Sphere mesh used to compute and visualise the histogram."""

    _faces: pd.DataFrame
    """Data frame containing information about the mesh faces."""

    number_of_shells: int
    """Number of shells to consider for bivariate vector histograms."""

    magnitude_range: Optional[Tuple[float, float]]
    """Range for the magnitude values.
    
    Maximum and minimum values to consider for the magnitude. If ``None``,
    then the maximum and minimum values are computed from the provided
    vectors.
    """

    magnitude_precision: Optional[int] = 8
    """Precision with which to round the magnitudes when binning.

    To avoid floating point errors, the vector magnitudes may be rounded
    before binning. This option allows the precision of the rounding to be
    set. If ``None``, then no rounding is performed.
    """

    def __init__(
        self,
        number_of_subdivisions: int = 3,
        number_of_shells: int = 1,
        magnitude_range: Optional[Tuple[float, float]] = None
    ):
        # Create the sphere
        sphere = trimesh.primitives.Sphere(
            radius=1,
            subdivisions=number_of_subdivisions,
            mutable=False
        )

        self._sphere = sphere

        # Get the data frame containing the faces
        face_ids = np.arange(len(sphere.faces))
        vertex_coordinates = sphere.vertices[sphere.faces].reshape(-1, 9)

        faces_dataframe = pd.DataFrame(
            vertex_coordinates,
            index=face_ids,
            columns=["x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3", "z3"]
        )

        self._faces = faces_dataframe

        # Assign the number of shells and the magnitude range
        self.number_of_shells = number_of_shells
        self.magnitude_range = magnitude_range

    def assign_histogram_bins(self, vectors: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        """Assign histogram bins using the triangle-based sphere.

        Parameters
        ----------
        vectors
            Array of shape ``(n, 3)`` containing the Cartesian components
            of the vectors from which to construct the histogram.

        Returns
        -------
        pandas.DataFrame
            List of all vectors, containing two additional columns,
            representing the magnitude ``shell`` and the nearest ``face``.
        numpy.ndarray
            Histogram bin edges for the magnitude shells.
        """

        # Create the histogram
        histogram = pd.DataFrame(vectors, columns=["x", "y", "z"])

        # Normalise the vectors and compute their magnitudes
        unit_vectors, magnitudes = util.normalise_vectors(vectors)

        # Now, perform the magnitude histogram binning
        if self.number_of_shells > 1:
            magnitude_bin_edges = np.histogram_bin_edges(
                magnitudes, bins=self.number_of_shells, range=self.magnitude_range
            )

            # Don't consider the initial bin edge.
            internal_bin_edges = magnitude_bin_edges[1:]

            # Round the magnitudes, if requested
            if self.magnitude_precision is not None:
                magnitudes = np.round(magnitudes, self.magnitude_precision)

            # Assign the vectors the correct bins
            magnitude_histogram_bins = np.digitize(
                magnitudes, internal_bin_edges, right=True
            )
        else:
            magnitude_histogram_bins = np.zeros(len(magnitudes), dtype=int)
            magnitude_bin_edges = np.array([0])

        histogram["shell"] = magnitude_histogram_bins

        # And now, deal with the triangular faces
        proximity_query = trimesh.proximity.ProximityQuery(self._sphere)
        _, _, face_indices = proximity_query.on_surface(unit_vectors)

        histogram["face"] = face_indices

        return histogram, magnitude_bin_edges


    def construct_histogram(
        self,
        binned_data: pd.DataFrame,
        return_fraction: bool = True
    ) -> pd.Series:
        """Construct a histogram based on the triangulated sphere.

        Parameters
        ----------
        binned_data
            All vectors, along with their respective shells, indicated in
            the ``shell`` column, and face, indicated in the ``face``
            column (case-sensitive).
        return_fraction
            Indicate whether the proportion or count of vectors in each
            face bin should be returned.

        Returns
        -------
        pandas.Series
            The frequency or count corresponding to each sphere face,
            grouped by shell.
        """

        # Get the total number of vectors
        number_of_vectors = len(binned_data)

        # Use groupby to perform the grouping
        original_histogram = binned_data.groupby(["shell", "face"]).apply(len)

        # Modify the index to account for any missing bins.
        multi_index = self._construct_histogram_index()

        filled_histogram = original_histogram.reindex(index=multi_index, fill_value=0)

        if return_fraction:
            filled_histogram /= number_of_vectors

        return filled_histogram

    def _construct_histogram_index(self) -> pd.MultiIndex:
        """Get the index for the current triangulated sphere.

        Produce the full histogram index for the current triangulated
        sphere, containing all face indices for each requested shell.

        Returns
        -------
        pandas.MultiIndex
            Index containing all possible values of ``shell`` and ``face``.
        """

        # Get the number of faces
        number_of_faces = len(self._faces)

        # Get the face indices
        face_indices = np.tile(np.arange(number_of_faces), self.number_of_shells)

        # And now, get the shell indices
        shell_indices = np.repeat(np.arange(self.number_of_shells), number_of_faces)

        # Construct the MultiIndex using these arrays
        multi_index = pd.MultiIndex.from_arrays(
            [shell_indices, face_indices],
            names=["shell", "face"]
        )

        return multi_index



def construct_spherical_histogram(
    vectors: np.ndarray,
    radius: float = 1,
    subdivisions: int = 3,
    weight_by_magnitude: bool = False,
) -> Tuple[trimesh.primitives.Sphere, np.ndarray]:
    """Construct a spherical histogram.

    Construct a spherical histogram based on the specified vectors. The
    number of points contained in each face will be counted and returned as
    an array, along with the actual sphere.

    Parameters
    ----------
    vectors
        NumPy array containing the 3D vector components in the order
        ``(x, y, z)``. This array should have shape ``(n, 3)`` where ``n``
        is the number of vectors.
    radius
        Radius of the sphere. The vector components are multiplied to
        ensure that they fit the sphere with this radius.
    subdivisions
        Number of subdivisions when constructing the icosphere.
    weight_by_magnitude
        Indicate whether to weight the histogram by 3D magnitude.

    Returns
    -------
    sphere : trimesh.primitives.Sphere
        The sphere on which the histogram is constructed.
    face_counts : numpy.ndarray
        Array containing the counts at each face. The order corresponds to
        the order of the faces defined by the mesh.
    """

    # First, normalise the vectors and compute their magnitudes
    normalised_vectors, magnitudes = util.normalise_vectors(vectors)

    # Now, multiply by the radius
    normalised_vectors *= radius

    # Construct the sphere
    sphere = trimesh.primitives.Sphere(radius=radius, subdivisions=subdivisions)

    # Construct a proximity query and get the closest points on the sphere
    proximity_query = trimesh.proximity.ProximityQuery(sphere)
    _, _, face_indices = proximity_query.on_surface(normalised_vectors)

    # And now, we need to construct the counts array
    number_of_faces = len(sphere.faces)
    counts_array = np.zeros(number_of_faces)

    # We need to populate this array with the counts for each face.
    for vector_index, face_index in enumerate(face_indices):
        if weight_by_magnitude:
            counts_array[face_index] += magnitudes[vector_index]
        else:
            counts_array[face_index] += 1

    # Finally, we return both the sphere and the array of counts
    return sphere, counts_array


def run_spherical_histogram_pipeline(
    vector_field: np.ndarray,
    radius: float = 1,
    subdivisions: int = 3,
    weight_by_magnitude: bool = False,
    is_axial_data: bool = False,
) -> Tuple[trimesh.primitives.Sphere, np.ndarray]:
    """Run the complete triangle-based spherical histogram construction.

    Construct a spherical histogram using a triangle-based mesh. This
    function performs preprocessing, as well as the sphere construction.

    Parameters
    ----------
    vector_field
        NumPy array of shape ``(n, 3)`` or ``(n, 6)`` containing the
        vectors to use to construct the histogram. If the array contains 6
        columns, the last 3 are assumed to be the vector components.
    radius
        Sphere radius, by default 1.
    subdivisions
        Sphere subdivision number, by default 3. Increasing this value
        provides more faces, and thus more detail, at a higher
        computational cost.
    weight_by_magnitude
        Indicate whether to weight the histogram by 3D magnitude. If
        `False`, the histogram will be weighted by count.
    is_axial_data
        Indicate whether the data correspond to axial data. If `True`, then
        the sphere will be symmetric across a 45° plane going through all
        three axes.

    Returns
    -------
    sphere : trimesh.primitives.Sphere
        The :class:`trimesh.primitives.Sphere` object used to construct the
        histogram.
    face_weights : numpy.ndarray
        Array containing the face weights.
    """

    # Extract the components
    vectors = vector_field[:, -3:]

    # Remove the non-zero vectors
    vectors = util.remove_zero_vectors(vectors)

    # Mirror the data, if axial data
    if is_axial_data:
        axes = util.convert_vectors_to_axes(vectors)
        vectors = util.create_symmetric_vectors_from_axes(axes)

    # Construct the histogram
    sphere, face_values = construct_spherical_histogram(
        vectors=vectors,
        radius=radius,
        subdivisions=subdivisions,
        weight_by_magnitude=weight_by_magnitude,
    )

    return sphere, face_values
