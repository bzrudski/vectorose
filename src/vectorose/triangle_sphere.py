# Copyright (c) 2024-, Benjamin Rudski, Joseph Deering
#
# This code is licensed under the MIT License. See the `LICENSE` file for
# more details about copying.

"""Triangle-based Sphere Plotting.

This module provides the functions necessary to produce a triangle mesh of
a sphere, with face colours corresponding to the point count in each face.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pyvista as pv
import trimesh

from .sphere_base import SphereBase
from . import util


class TriangleSphere(SphereBase):
    """Representation of a sphere constructed using equal-area triangles.

    Compute and visualise histograms using a sphere composed of equal-area
    triangles.
    """

    # Attributes
    _sphere: trimesh.primitives.Sphere
    """Sphere mesh used to compute and visualise the histogram."""

    _faces: pd.DataFrame
    """Data frame containing information about the mesh faces."""

    @property
    def orientation_cols(self) -> List[str]:
        return ["face"]

    def _initial_vector_data_preparation(self, vectors: pd.DataFrame) -> pd.DataFrame:
        vectors_array = vectors.loc[:, ["x", "y", "z"]].to_numpy()
        unit_vectors, magnitudes = util.normalise_vectors(vectors_array)

        magnitudes = magnitudes[:, None]

        vector_data = np.concatenate([unit_vectors, magnitudes], axis=-1)

        # Create a data frame with the unit vectors and magnitudes.
        unit_vector_data_frame = pd.DataFrame(
            vector_data, columns=["ux", "uy", "uz", "magnitude"]
        )

        return unit_vector_data_frame

    def _compute_orientation_binning(self, vectors: pd.DataFrame) -> pd.Series:
        unit_vectors = vectors.loc[:, ["ux", "uy", "uz"]].to_numpy()
        proximity_query = trimesh.proximity.ProximityQuery(self._sphere)
        _, _, face_indices = proximity_query.on_surface(unit_vectors)

        face_series = pd.Series(face_indices, name="face")

        return face_series

    def __init__(
        self,
        number_of_subdivisions: int = 3,
        number_of_shells: int = 1,
        magnitude_range: Optional[Tuple[float, float]] = None,
    ):
        # Create the sphere
        sphere = trimesh.primitives.Sphere(
            radius=1, subdivisions=number_of_subdivisions, mutable=False
        )

        self._sphere = sphere

        # Get the data frame containing the faces
        face_ids = np.arange(len(sphere.faces))
        vertex_coordinates = sphere.vertices[sphere.faces].reshape(-1, 9)

        faces_dataframe = pd.DataFrame(
            vertex_coordinates,
            index=face_ids,
            columns=["x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3", "z3"],
        )

        self._faces = faces_dataframe

        super().__init__(
            number_of_shells=number_of_shells, magnitude_range=magnitude_range
        )

    def _construct_orientation_index(self) -> pd.RangeIndex:
        """Get the orientation index for the current triangulated sphere.

        Produce the orientation index for the current triangulated sphere,
        containing all face indices for a given shell.

        Returns
        -------
        pandas.RangeIndex
            Index containing all valid ``face`` indices.
        """

        # Get the number of faces
        number_of_faces = len(self._faces)

        # Get the face indices
        face_indices = pd.RangeIndex(0, number_of_faces)

        return face_indices

    def create_mesh(self) -> pv.PolyData:
        points = self._sphere.vertices
        faces = self._sphere.faces

        number_of_faces = len(faces)

        # Augment the faces by adding a column with 3s
        threes_column = np.ones(number_of_faces, dtype=int) * 3
        threes_column = np.atleast_2d(threes_column).T
        complete_faces = np.concatenate([threes_column, faces], axis=-1)

        # And now, build the mesh
        sphere_mesh = pv.PolyData(points, complete_faces)

        # And now, just to be sure, let's put in the face scalars
        sphere_mesh.cell_data["face-index"] = range(number_of_faces)

        return sphere_mesh

    def convert_vectors_to_cartesian_array(
        self, labelled_vectors: pd.DataFrame, create_unit_vectors: bool = False
    ) -> np.ndarray:
        # So, the way that the frame is structured is that we have the
        # Cartesian components of the unit vectors as `ux, uy, uz` and then
        # we have the magnitude in the `magnitude` column.

        # First, let's extract the vector components.
        unit_vectors = labelled_vectors[["ux", "uy", "uz"]].to_numpy()

        # If we only want unit vectors, great! Return these!
        if create_unit_vectors:
            return unit_vectors

        magnitudes = labelled_vectors["magnitude"].to_numpy()

        magnitudes = np.expand_dims(magnitudes, axis=-1)

        cartesian_vectors = unit_vectors * magnitudes

        return cartesian_vectors


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
