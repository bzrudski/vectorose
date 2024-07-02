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
import enum
from typing import List, Optional, Tuple, Type

import matplotlib.colors
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import mpl_toolkits.mplot3d.art3d
import numpy as np

import vectorose.util
import vectorose.vectorose
from vectorose.util import perform_binary_search


class TregenzaSphereBase:
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
    def __init__(
        self,
        patch_count: np.ndarray,
        phi_values: np.ndarray,
        theta_values: List[np.ndarray],
    ):
        self.patch_count = patch_count
        self.phi_values = phi_values
        self.theta_values = theta_values

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

        # print(f"Found phi {phi} in ring {phi_ring}...")

        # Get the theta bin within this ring
        theta_bin = self.get_closest_theta_bin_in_ring(phi_ring, theta)
        # print(f"Found theta {theta} is in bin {theta_bin}...")

        return phi_ring, theta_bin

    def construct_spherical_histogram(
        self,
        angular_coordinates: np.ndarray,
        use_radians: bool = False,
        magnitudes: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """Construct a histogram using the Tregenza sphere.

        Parameters
        ----------
        angular_coordinates
            Orientations from which to construct the histogram.
        use_radians
            Indicate whether the orientations are in radians.
        magnitudes
            Optional vector magnitudes. If `None`, the resulting histogram
            is weighted by counts. Otherwise, it is weighted by magnitude.

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
        for i, (phi_ring, theta_bin) in enumerate(orientations):
            if magnitudes is not None:
                histogram[phi_ring][theta_bin] += magnitudes[i]
            else:
                histogram[phi_ring][theta_bin] += 1

        # And finally, to return the histogram
        return histogram

    def correct_histogram_by_area(
        self, histogram: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Correct histogram by face area.

        Weight histogram values by face areas to compensate for slight
        deviations from equal area.

        Parameters
        ----------
        histogram
            Histogram values to correct.

        Returns
        -------
        list[numpy.ndarray]
            Corrected histogram values. Same shape as `histogram`.

        """

        # Compute the weights
        ring_weights = self.compute_weights()
        weighted_face_data = [
            histogram[i] * ring_weights[i] for i in range(self.number_of_rings)
        ]

        return weighted_face_data

    def create_tregenza_plot(
        self,
        ax: Optional[mpl_toolkits.mplot3d.Axes3D] = None,
        face_data: Optional[List[np.ndarray]] = None,
        cmap: str = "viridis",
        norm: Optional[matplotlib.colors.Normalize] = None,
        sphere_alpha: float = 1.0,
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
        sphere_alpha
            Sphere opacity.

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
                norm = matplotlib.colors.Normalize(
                    vmin=min_face_count, vmax=max_face_count
                )
            else:
                norm.autoscale_None(flattened_face_data)

            # Compute the colours
            scalar_mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

            face_colours: np.ndarray = scalar_mapper.to_rgba(flattened_face_data)

            # print("Face colours are:")
            # print(face_colours)

            # normalised_data = norm(flattened_face_data)
            # print(
            #     f"Maximum normalised data: {normalised_data.max()}\nMinimum normalised data: "
            #     f"{normalised_data.min()}"
            # )

            # for ring in face_data:
            #     face_colours.append(scalar_mapper.to_rgba(ring))
        else:
            face_colours = None

        if ax is None:
            ax: mpl_toolkits.mplot3d.Axes3D = plt.axes(projection="3d")

        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-1, 1)

        # Define the patches we'll plot
        all_patch_vertices: List[np.ndarray] = []

        # So, let's start with the top row by starting with the second row.
        phi_upper = self.phi_values[1]
        thetas_second_row = self.theta_values[1]

        phi_upper_second_row = np.ones(thetas_second_row.shape) * phi_upper
        top_cap_vertices = np.stack([phi_upper_second_row, thetas_second_row], axis=-1)

        top_cap_vertices_cartesian = (
            vectorose.util.convert_spherical_to_cartesian_coordinates(
                np.radians(top_cap_vertices)
            )
        )

        all_patch_vertices.append(top_cap_vertices_cartesian)

        # Now, let's go with the remaining rings
        number_of_rings = self.number_of_rings

        # phi_rings = np.append(self.phi_values, 90)
        phi_rings = self.phi_values

        for i in range(1, number_of_rings - 1):
            # Get the current phi ring and the next phi ring
            upper_phi = phi_rings[i]
            lower_phi = phi_rings[i + 1]

            # Get the current theta bounds
            current_thetas = np.append(self.theta_values[i], 360)

            # Get the number of faces in the ring
            number_of_faces = len(current_thetas) - 1

            # print(f"Considering ring {i} which contains {number_of_faces + 1} faces")

            # Now, for each face, we need to construct a rectangle with
            # four vertices, which are related to the bounds.
            # current_row_faces: list[np.ndarray] = []

            for j in range(number_of_faces):
                lower_theta = current_thetas[j]
                upper_theta = current_thetas[j + 1]

                # Define the vertices
                v1 = (upper_phi, lower_theta)
                v2 = (upper_phi, upper_theta)
                v3 = (lower_phi, upper_theta)
                v4 = (lower_phi, lower_theta)

                face_vertices = np.array([v1, v2, v3, v4])

                face_vertices_cartesian = (
                    vectorose.util.convert_spherical_to_cartesian_coordinates(
                        np.radians(face_vertices)
                    )
                )

                all_patch_vertices.append(face_vertices_cartesian)

        # And finally, the bottom patch
        phi_value = self.phi_values[-1]
        thetas_second_last_row = self.theta_values[-2]

        phi_value_bottom_row = np.ones(thetas_second_last_row.shape) * phi_value
        bottom_cap_vertices = np.stack(
            [phi_value_bottom_row, thetas_second_last_row], axis=-1
        )

        bottom_cap_vertices_cartesian = (
            vectorose.util.convert_spherical_to_cartesian_coordinates(
                np.radians(bottom_cap_vertices)
            )
        )

        all_patch_vertices.append(bottom_cap_vertices_cartesian)

        patch_collection = mpl_toolkits.mplot3d.art3d.Poly3DCollection(
            all_patch_vertices,
            facecolors=face_colours,
            shade=False,
            linewidths=0,
            alpha=sphere_alpha,
        )

        ax.add_collection3d(patch_collection)

        return ax

    def compute_weights(self) -> np.ndarray:
        """Compute the weights for each ring.

        Compute the weights for each ring to account for deviations from
        equal area.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(r,)`` where ``r`` is the number of rings in
            the sphere. These weights reflect the deviation from equal
            area.

        Notes
        -----
        The weights are computed in the following manner. For each ring,
        the area of the spherical cap up to the bottom of the ring is
        calculated as

        .. math::

            A_{\\textup{cap}}(r) = 2\\pi (1 - \\cos(\\phi_{r+1}))

        where :math:`\\phi_{r+1}` is the almucantar angle at the bottom of
        the ring with index ``r`` in **radians**.

        The area of the ring is obtained by subtracting the areas of the
        previous rings from the total cap area. For the first and last
        rings, which correspond to the spherical caps, the ring area is
        equal to the cap area.

        The area of each Tregenza patch is then computed by dividing the
        total ring area by the number of patches in that ring.

        Finally, the smallest patch, corresponding to the polar cap, is
        used to determine the weights. The weight for each row corresponds
        to the ratio between the areas of the smallest patch and the
        respective row patch. These values should all be between 0 and 1.
        """

        # Determine the cumulative spherical cap areas
        end_angles = np.roll(self.phi_values, -1)
        end_angles[-1] = 180

        cumulative_areas = 2 * np.pi * (1 - np.cos(np.radians(end_angles)))

        preceding_areas = np.roll(cumulative_areas, 1)
        preceding_areas[0] = 0
        ring_areas = cumulative_areas - preceding_areas

        patch_areas = ring_areas / self.patch_count

        smallest_area = patch_areas.min()

        weights = smallest_area / patch_areas

        return weights


class FineTregenzaSphere(TregenzaSphereBase):
    def __init__(self):
        # Define the almucantar angles (phi rings)
        almucantar_angles = [0.00, 1.50, 4.00]

        # Add almucantars for remaining upper bounds of phi
        for i in range(1, 25):
            new_almucantar_angle = 4.00 + i * 3.44
            almucantar_angles.append(new_almucantar_angle)

        top_phi_values = np.array(almucantar_angles)
        bottom_phi_values = 180 - np.flip(top_phi_values[1:])

        phi_values = np.concatenate([top_phi_values, [90], bottom_phi_values])

        # Define the patch count
        top_patch_count = np.array(
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

        bottom_patch_count = np.flip(top_patch_count)

        patch_count = np.concatenate([top_patch_count, bottom_patch_count])

        # Define the theta bins within each ring
        number_of_rings = len(patch_count)
        theta_bounds: List[np.ndarray] = []

        for i in range(number_of_rings):
            # Get the number of patches in the current ring
            number_of_patches = patch_count[i]

            # Get the angular spacing for the ring in degrees
            row_theta_bounds = np.linspace(
                start=0, stop=360, num=number_of_patches, endpoint=False
            )

            theta_bounds.append(row_theta_bounds)

        theta_values = theta_bounds

        super().__init__(patch_count, phi_values, theta_values)


class UltraFineTregenzaSphere(TregenzaSphereBase):
    def __init__(self):
        # Define the almucantar angles (phi rings)
        almucantar_angles = [0.00, 0.60, 1.80]

        # Add almucantars for remaining upper bounds of phi
        for i in range(1, 60):
            new_almucantar_angle = 1.80 + i * 1.47
            almucantar_angles.append(new_almucantar_angle)

        top_phi_values = np.array(almucantar_angles)
        bottom_phi_values = 180 - np.flip(top_phi_values[1:])

        phi_values = np.concatenate([top_phi_values, [90], bottom_phi_values])

        # Define the patch count
        top_patch_count = np.array(
            [
                1,
                8,
                21,
                33,
                45,
                57,
                69,
                81,
                93,
                105,
                117,
                128,
                140,
                152,
                163,
                175,
                186,
                197,
                208,
                219,
                230,
                240,
                251,
                261,
                271,
                281,
                291,
                300,
                309,
                319,
                327,
                336,
                345,
                353,
                361,
                369,
                376,
                384,
                391,
                397,
                404,
                410,
                416,
                422,
                427,
                432,
                437,
                442,
                446,
                450,
                454,
                457,
                460,
                463,
                466,
                468,
                470,
                471,
                472,
                473,
                474,
                474,
            ]
        )

        bottom_patch_count = np.flip(top_patch_count)

        patch_count = np.concatenate([top_patch_count, bottom_patch_count])

        # Define the theta bins within each ring
        number_of_rings = len(patch_count)
        theta_bounds: List[np.ndarray] = []

        for i in range(number_of_rings):
            # Get the number of patches in the current ring
            number_of_patches = patch_count[i]

            # Get the angular spacing for the ring in degrees
            row_theta_bounds = np.linspace(
                start=0, stop=360, num=number_of_patches, endpoint=False
            )

            theta_bounds.append(row_theta_bounds)

        theta_values = theta_bounds

        super().__init__(patch_count, phi_values, theta_values)


def run_tregenza_histogram_pipeline(
    vectors: np.ndarray,
    sphere: TregenzaSphereBase,
    weight_by_magnitude: bool = False,
    is_axial: bool = False,
    remove_zero_vectors: bool = True,
    correct_area_weighting: bool = True,
) -> List[np.ndarray]:
    """Run the complete histogram construction for the Tregenza sphere.

    Construct a spherical histogram based on a provided Tregenza sphere for
    the supplied vectors.

    Parameters
    ----------
    vectors
        NumPy array containing the 3D vector components in the order
        ``(x, y, z)``. This array should have shape ``(n, 3)`` where ``n``
        is the number of vectors.
    sphere
        Tregenza sphere to use for histogram construction.
    weight_by_magnitude
        Indicate whether to weight the histogram by 3D magnitude. If
        `False`, then the histogram will be weighted by count.
    is_axial
        Indicate whether the vectors are axial. If `True`, symmetric
        vectors will be created based on the dataset.
    remove_zero_vectors
        Indicate whether to remove zero-vectors. This parameter should be
        `True` unless the vector list contains no zero-vectors.
    correct_area_weighting
        Indicate whether the histogram values should be corrected using the
        area weights.

    Returns
    -------
    list[numpy.ndarray]
        List of histogram counts for each ring. The list has the same
        length as the number of rings in the provided Tregenza sphere,
        and the length of each list entry corresponds to the respective
        patch count.

    """

    # Perform vector pre-processing
    if remove_zero_vectors:
        vectors = vectorose.util.remove_zero_vectors(vectors)

    if is_axial:
        vectors = vectorose.util.convert_vectors_to_axes(vectors)
        vectors = vectorose.util.create_symmetric_vectors_from_axes(vectors)

    angular_coordinates = vectorose.util.compute_vector_orientation_angles(
        vectors, use_degrees=True
    )

    if weight_by_magnitude:
        _, magnitudes = vectorose.util.normalise_vectors(vectors)
    else:
        magnitudes = None

    histogram = sphere.construct_spherical_histogram(
        angular_coordinates=angular_coordinates, magnitudes=magnitudes
    )

    if correct_area_weighting:
        histogram = sphere.correct_histogram_by_area(histogram)

    return histogram


class TregenzaSphereDetailLevel(enum.Enum):
    """Detail level for Tregenza Sphere."""

    FINE = "Fine"
    """Fine-detail Tregenza sphere, with 27 rings per hemisphere,
    represented by :class:`FineTregenzaSphere`."""

    ULTRA_FINE = "Ultra-fine"
    """Ultra-fine-detail Tregenza sphere, with 62 rings per hemisphere,
    represented by :class:`UltraFineTregenzaSphere`."""

    def get_tregenza_class(self) -> Type[TregenzaSphereBase]:
        """Get the class for the specified level of detail.

        Returns
        -------
        Type[TregenzaSphereBase]
            Class inheriting from :class:`TregenzaSphereBase` which
            captures the correct level of detail.

        Warnings
        --------
        This method returns a **type**, not an object. The returned type
        can be used to instantiate a Tregenza sphere object. To instantiate
        an object directly, call :meth:`create_tregenza_sphere`.
        """

        if self == TregenzaSphereDetailLevel.FINE:
            return FineTregenzaSphere
        elif self == TregenzaSphereDetailLevel.ULTRA_FINE:
            return UltraFineTregenzaSphere

    def create_tregenza_sphere(self) -> TregenzaSphereBase:
        """Create a Tregenza sphere for the specified detail level.

        Returns
        -------
        TregenzaSphereBase
            Object of the correct subclass of :class:`TregenzaSphereBase`
            representing the desired level of detail.
        """

        tregenza_sphere_type = self.get_tregenza_class()

        tregenza_sphere: TregenzaSphereBase = tregenza_sphere_type()

        return tregenza_sphere
