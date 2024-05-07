# Copyright (c) 2024-, Benjamin Rudski, Joseph Deering
#
# This code is licensed under the MIT License. See the `LICENSE` file for
# more details about copying.

"""Equal-area Rectangle Plotting.

This module provides the functions necessary to produce an approximately
equal area rectangular-based projection of a sphere, with face colours
corresponding to either the face count or a sum of the magnitudes of
vectors at each orientation. This projection is based on work by 
Beckers & Beckers. [Beckers]_

References
----------
.. [Beckers] Beckers, B., & Beckers, P. (2012). A general rule for disk and
   hemisphere partition into equal-area cells. Computational Geometry,
   45(7), 275-283. https://doi.org/10.1016/j.comgeo.2012.01.011

"""
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import mpl_toolkits.mplot3d
import mpl_toolkits.mplot3d.art3d
import matplotlib.pyplot as plt
import matplotlib.colors

import vectorose.vectorose
from vectorose.vectorose import perform_binary_search


class TregenzaSphere:
    """Representation of a Tregenza Sphere.

    Represent and interact with a Tregenza Sphere, similar to those in the
    work by Beckers & Beckers. [Beckers]_
    """

    # Attributes: phi values, theta bounds per ring, face indices
    patch_count: np.ndarray
    phi_values: np.ndarray
    theta_values: List[np.ndarray]  # TODO: Maybe change this

    @property
    def number_of_rings(self) -> int:
        """Number of rings in the sphere."""
        return len(self.patch_count)

    # Define the constructor...
    def __init__(self):
        # Define the almucantar angles (phi rings)
        almucantar_angles = [0.00, 1.50, 4.00]

        for i in range(1, 25):  # Adds almucantars for remaining upper bounds of phi
            new_almucantar_angle = 4.00 + i * 3.44
            almucantar_angles.append(new_almucantar_angle)

        self.phi_values = np.array(almucantar_angles)

        # Define the patch count
        self.patch_count = np.array(
            [
                1,
                6,
                17,
                27,
                38,
                48,
                58,
                68,
                77,
                87,
                96,
                104,
                112,
                120,
                128,
                135,
                141,
                147,
                152,
                157,
                161,
                165,
                168,
                171,
                173,
                173,
                173,
            ]
        )

        # Define the theta bins within each ring
        number_of_rings = len(self.patch_count)
        theta_bounds: List[np.ndarray] = []

        for i in range(number_of_rings):
            # Get the number of patches in the current ring
            number_of_patches = self.patch_count[i]

            # Get the angular spacing for the ring in degrees
            row_theta_bounds = np.linspace(
                start=0, stop=360, num=number_of_patches, endpoint=False
            )

            theta_bounds.append(row_theta_bounds)

        self.theta_values = theta_bounds

    def get_closest_phi_ring(self, phi: float) -> int:
        """Find the index of the closest phi ring for a specified angle.

        Parameters
        ----------
        phi
            Angle under consideration, in degrees.

        Returns
        -------
        int
            Index of the appropriate phi ring.

        """

        return perform_binary_search(seq=self.phi_values, item=phi)

    def get_closest_theta_bin_in_ring(self, phi_ring: int, theta: float) -> int:
        """Find the index of the closest theta bin in a specific phi ring.

        Parameters
        ----------
        phi_ring
            The index of the phi ring under consideration.
        theta
            The theta value under consideration.

        Returns
        -------
        int
            Index of the theta bin in the specified phi ring.

        """

        theta_bounds = self.theta_values[phi_ring]
        return perform_binary_search(seq=theta_bounds, item=theta)

    def get_closest_face(self, phi: float, theta: float) -> Tuple[int, int]:
        """Get the closest face for a specified spherical position.

        Parameters
        ----------
        phi
            The angle phi of inclination from the positive z-axis.
        theta
            The in-plane angle theta clockwise with respect to the positive
            y-axis.

        Returns
        -------
        tuple[int, int]
            Tuple containing the phi ring and theta bin containing the
            desired angular coordinates.

        """

        # Get the phi ring
        phi_ring = self.get_closest_phi_ring(phi)

        print(f"Found phi {phi} in ring {phi_ring}...")

        # Get the theta bin within this ring
        theta_bin = self.get_closest_theta_bin_in_ring(phi_ring, theta)
        print(f"Found theta {theta} is in bin {theta_bin}...")

        return phi_ring, theta_bin

    def construct_spherical_histogram(
        self, angular_coordinates: np.ndarray, use_radians: bool = False
    ) -> List[np.ndarray]:
        """Construct a histogram using the Tregenza sphere.

        Parameters
        ----------
        angular_coordinates
            Orientations from which to construct the histogram.
        use_radians
            Indicate whether the orientations are in radians.

        Returns
        -------
        list[numpy.ndarray]
            List of arrays, each of which has the number of bins
            corresponding to the respective patch number in that row. The
            bins contain the histogram counts for the given orientation
            data.

        """

        # Build the data structure
        histogram: List[np.ndarray] = []

        for count in self.patch_count:
            histogram.append(np.zeros(count))

        # Convert to degrees, if necessary
        if use_radians:
            angular_coordinates_degrees = np.degrees(angular_coordinates)
        else:
            angular_coordinates_degrees = angular_coordinates

        # Get the bin coordinates for each orientation
        orientations = [
            self.get_closest_face(phi, theta)
            for phi, theta in angular_coordinates_degrees
        ]

        # And now, to build up the histogram
        for phi_ring, theta_bin in orientations:
            histogram[phi_ring][theta_bin] += 1

        # And finally, to return the histogram
        return histogram

    def create_tregenza_plot(
        self, ax: Optional[mpl_toolkits.mplot3d.Axes3D] = None,
        face_data: Optional[List[np.ndarray]] = None,
        cmap: str = "viridis",
        norm: Optional[matplotlib.colors.Normalize] = None
    ) -> mpl_toolkits.mplot3d.Axes3D:
        """Create a plot of the current Tregenza sphere.

        Parameters
        ----------
        ax
            Axes on which to plot the sphere. If `None`, then new axes are
            generated. The axes **must** have ``projection="3d"`` set.
        face_data
            Optional data to plot on each face, in the case of a histogram.
        cmap
            Colour map to use for the face colours.
        norm
            Normaliser to use for the face colours. If `None`, then a
            linear normaliser is used.

        Returns
        -------
        mpl_toolkits.mplot3d.Axes3d
            Axes containing the Tregenza sphere plot.

        """

        # If we have a histogram, work on processing the data
        face_colours: Optional[np.ndarray]

        if face_data is not None:
            # Flatten the face data for easier processing.
            flattened_face_data = np.concatenate(face_data)

            if norm is None:
                # Find the maximum and minimum counts
                min_face_count = flattened_face_data.min()
                max_face_count = flattened_face_data.max()
                norm = matplotlib.colors.Normalize(vmin=min_face_count, vmax=max_face_count)
            else:
                norm.autoscale(flattened_face_data)

            # Compute the colours
            scalar_mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

            face_colours: np.ndarray = scalar_mapper.to_rgba(flattened_face_data)

            # for ring in face_data:
            #     face_colours.append(scalar_mapper.to_rgba(ring))
        else:
            face_colours = None

        if ax is None:
            ax: mpl_toolkits.mplot3d.Axes3D = plt.axes(projection="3d")

        # Define the patches we'll plot
        all_patch_vertices: list[np.ndarray] = []

        # So, let's start with the top row by starting with the second row.
        phi_upper = self.phi_values[1]
        thetas_second_row = self.theta_values[1]

        phi_upper_second_row = np.ones(thetas_second_row.shape) * phi_upper
        top_cap_vertices = np.stack([phi_upper_second_row, thetas_second_row], axis=-1)

        top_cap_vertices_cartesian = vectorose.vectorose.convert_spherical_to_cartesian_coordinates(
            np.radians(top_cap_vertices)
        )

        all_patch_vertices.append(top_cap_vertices_cartesian)

        # Now, let's go with the remaining rings
        number_of_rings = self.number_of_rings

        phi_rings = np.append(self.phi_values, 90)

        for i in range(1, number_of_rings):
            # Get the current phi ring and the next phi ring
            upper_phi = phi_rings[i]
            lower_phi = phi_rings[i+1]

            # Get the current theta bounds
            current_thetas = np.append(self.theta_values[i], 360)

            # Get the number of faces in the ring
            number_of_faces = len(current_thetas) - 1

            # Now, for each face, we need to construct a rectangle with
            # four vertices, which are related to the bounds.
            # current_row_faces: list[np.ndarray] = []

            for j in range(number_of_faces):
                lower_theta = current_thetas[j]
                upper_theta = current_thetas[j+1]

                # Define the vertices
                v1 = (upper_phi, lower_theta)
                v2 = (upper_phi, upper_theta)
                v3 = (lower_phi, upper_theta)
                v4 = (lower_phi, lower_theta)

                face_vertices = np.array([
                    v1, v2, v3, v4
                ])

                face_vertices_cartesian = (
                    vectorose.vectorose.convert_spherical_to_cartesian_coordinates(
                        np.radians(face_vertices)
                    )
                )

                all_patch_vertices.append(face_vertices_cartesian)

        patch_collection = mpl_toolkits.mplot3d.art3d.Poly3DCollection(all_patch_vertices)
        patch_collection.set_color(face_colours)
        ax.add_collection3d(patch_collection)

        return ax
