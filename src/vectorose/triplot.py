# Copyright (c) 2024-, Benjamin Rudski, Joseph Deering
#
# This code is licensed under the MIT License. See the `LICENSE` file for
# more details about copying.

"""Triangle-based Sphere Plotting.

This module provides the functions necessary to produce and plot a triangle
mesh of a sphere, with face colours corresponding to the point count in
each face.
"""

from typing import Tuple

import numpy as np
import trimesh

from . import vectorose


def construct_spherical_histogram(
    vectors: np.ndarray,
    radius: float = 1,
    subdivisions: int = 3,
    normalise_vector_lengths: bool = False,
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
        is the number of vectors. If `normalise_vector_lengths` is set to
        `False`, then the vectors are assumed to be of unit length.
        Otherwise, the vectors are normalised.
    radius
        Radius of the sphere. The vector components are multiplied to
        ensure that they fit the sphere with this radius.
    subdivisions
        Number of subdivisions when constructing the icosphere.
    normalise_vector_lengths
        Indicate whether the vector lengths should be normalised to unit
        length.

    Returns
    -------
    sphere : trimesh.primitives.Sphere
        The sphere on which the histogram is constructed.
    face_counts : numpy.ndarray
        Array containing the counts at each face. The order corresponds to
        the order of the faces defined by the mesh.
    """

    # First, normalise the vectors if necessary
    if normalise_vector_lengths:
        normalised_vectors = vectorose.normalise_vectors(vectors)
    else:
        normalised_vectors = vectors.copy()

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
    for i in face_indices:
        counts_array[i] += 1

    # Finally, we return both the sphere and the array of counts
    return sphere, counts_array
