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

    def _initial_vector_component_preparation(
        self, vectors: pd.DataFrame
    ) -> pd.DataFrame:
        vectors_array = vectors.loc[:, ["vx", "vy", "vz"]].to_numpy()
        unit_vectors, magnitudes = util.normalise_vectors(vectors_array)

        magnitudes = magnitudes[:, None]

        # Prepare the data to make a new DataFrame
        vector_data = np.concatenate([unit_vectors, magnitudes], axis=-1)
        columns = ["ux", "uy", "uz", "magnitude"]

        # Create a data frame with the unit vectors and magnitudes.
        unit_vector_data_frame = pd.DataFrame(vector_data, columns=columns)

        return unit_vector_data_frame

    def _compute_orientation_binning(self, vectors: pd.DataFrame) -> pd.Series:
        unit_vectors = vectors.loc[:, ["ux", "uy", "uz"]].to_numpy()
        proximity_query = trimesh.proximity.ProximityQuery(self._sphere)
        _, _, face_indices = proximity_query.on_surface(unit_vectors)

        face_series = pd.Series(face_indices, name="face")

        return face_series

    def to_dataframe(self) -> pd.DataFrame:
        """Get the data frame representation of the sphere.

        Notes
        -----
        The produced table contains one row for each face in the
        triangulated sphere. The column headers are ``x1``, ``y1``, ``z1``,
        ``x2``, ``y2``, ``z2``, ``x3``, ``y3``, ``z3``, reflecting the
        Cartesian coordinates of each vertex forming the face.
        """

        return self._faces.copy()

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
        face_index = pd.RangeIndex(start=0, stop=len(sphere.faces), name="face")
        vertex_coordinates = sphere.vertices[sphere.faces].reshape(-1, 9)

        faces_dataframe = pd.DataFrame(
            vertex_coordinates,
            index=face_index,
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
        self,
        labelled_vectors: pd.DataFrame,
        create_unit_vectors: bool = False,
        include_spatial_coordinates: bool = False,
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

        if include_spatial_coordinates:
            spatial_locations = labelled_vectors[["x", "y", "z"]].to_numpy()
            cartesian_vectors = np.concatenate(
                [spatial_locations, cartesian_vectors], axis=-1
            )

        return cartesian_vectors
