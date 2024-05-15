# Copyright (c) 2024-, Benjamin Rudski, Joseph Deering
#
# This code is licensed under the MIT License. See the `LICENSE` file for
# more details about copying.

"""Triangle-based Sphere Plotting.

This module provides the functions necessary to produce a triangle mesh of
a sphere, with face colours corresponding to the point count in each face.
"""

from typing import Tuple

import numpy as np
import trimesh

from . import util


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
