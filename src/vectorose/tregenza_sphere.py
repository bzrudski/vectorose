# Copyright (c) 2024-, Benjamin Rudski, Joseph Deering
#
# This code is licensed under the MIT License. See the `LICENSE` file for
# more details about copying.

"""Sphere plotting with (almost) equal-area rectangles.

This module provides the functions necessary to produce an approximately
equal area rectangular-based projection of a sphere, with face colours
corresponding to either the face count or a sum of the magnitudes of
vectors at each orientation. This projection is based on work by 
Beckers & Beckers. [#Beckers]_

References
----------
.. [#Beckers] Beckers, B., & Beckers, P. (2012). A general rule for disk and
   hemisphere partition into equal-area cells. Computational Geometry,
   45(7), 275-283. https://doi.org/10.1016/j.comgeo.2012.01.011

"""
# import enum
from typing import List, Optional, Tuple, Type

import numpy as np
import pandas as pd
import pyvista as pv

from . import util
from .sphere_base import SphereBase


class TregenzaSphere(SphereBase):
    """Base class for the Tregenza sphere.

    The Tregenza sphere provides a discretisation of the sphere based on
    mostly rectangular patches of almost equal area. This class defines the
    interface for the Tregenza sphere and includes all the functions
    necessary to use it for histogram binning and 3D representation.


    Notes
    -----
    This sphere design is based on the tiling presented by Beckers &
    Beckers. [#Beckers]_

    While this class can be instantiated directly, we **strongly**
    recommend that you consider one of the derived classes presented below.
    By using these, you will avoid the tedious task of computing the
    optimal number of bins per row, as well as computing the almucantar
    angles for the rings.
    """

    binning_precision: int = 10
    """Precision for rounding when computing bin assignments."""

    _rings: pd.DataFrame
    """Tregenza sphere ring definitions."""

    _ring_increment: float
    """Angular increment for regularly-sized rings."""

    @property
    def number_of_rings(self) -> int:
        """Number of rings in the sphere."""
        return len(self._rings)

    @property
    def number_of_irregular_rings(self) -> int:
        """Number of irregular rings in the sphere."""
        return len(self._rings[~self._rings.loc[:, "regular"]])

    @property
    def ring_increment(self) -> float:
        """Angular increment for the regularly-sized rings."""
        return self._ring_increment

    @property
    def orientation_cols(self) -> List[str]:
        return ["ring", "bin"]

    # Define the constructor...
    def __init__(
        self,
        patch_count: np.ndarray,
        irregular_phi_values: np.ndarray,
        ring_increment: float,
        number_of_shells: int = 1,
        magnitude_range: Optional[Tuple[float, float]] = None,
    ):
        rings = self._construct_tregenza_sphere(
            irregular_phi_values, patch_count, ring_increment
        )

        self._rings = rings
        self._ring_increment = ring_increment

        super().__init__(
            number_of_shells=number_of_shells, magnitude_range=magnitude_range
        )

    def _construct_tregenza_sphere(
        self,
        irregular_phi_values: np.ndarray,
        patch_count: np.ndarray,
        ring_increment: float,
    ) -> pd.DataFrame:
        """Define a Tregenza sphere.

        Define a Tregenza sphere with the specified patch count and
        almucantar angles.

        Parameters
        ----------
        irregular_phi_values
            The phi values for the first rings, which are manually defined
            and of irregular increments.
        patch_count
            The number of rectangular patches in each ring.
        ring_increment
            The regular phi spacing between adjacent rings.

        Returns
        -------
        pandas.DataFrame
            The structure of the produced Tregenza sphere, containing as
            index the ring number, and as columns:

            * ``bins`` -- the number of bins in each ring;
            * ``start`` -- the starting phi value for each ring;
            * ``end`` -- the ending phi angle of each ring;
            * ``theta_inc`` -- the theta increment within each ring;
            * ``face_area`` -- the area of the faces in each ring;
            * ``weight`` -- the correction weight for the faces in each
                ring;
            * ``regular`` -- an indication of whether the ring has a
                regular almucantar spacing or not.

        Warnings
        --------
        This method assumes that the Tregenza sphere is symmetric with
        respect to the equator, i.e., that the northern and southern
        hemispheres are reflections of each other.
        """
        # Define our data frame which will hold everything
        rings = pd.DataFrame()
        rings.index.name = "ring"

        # Start by getting the number of rings
        half_number_of_rings = len(patch_count)
        number_of_rings = 2 * half_number_of_rings

        # Begin by mirroring the patch count.
        patch_count_bottom_half = np.flip(patch_count)
        all_patch_counts = np.hstack([patch_count, patch_count_bottom_half])

        # Start by defining the rings based on the patch count.
        rings["bins"] = pd.Series(all_patch_counts)

        # And now, for the ring beginning angles
        number_of_irregular_rings = len(irregular_phi_values)
        initial_angle = irregular_phi_values[-1]
        ring_initial_angles = np.arange(
            initial_angle + ring_increment, 89.999999, ring_increment
        )
        top_half_initial_angles = np.hstack([irregular_phi_values, ring_initial_angles])

        # And now, for the initial angles in the bottom half
        bottom_half_initial_angles = 180 - top_half_initial_angles[1:]
        bottom_half_initial_angles = np.flip(bottom_half_initial_angles)
        bottom_half_initial_angles = np.insert(bottom_half_initial_angles, 0, 90.0)

        # Finally, combine all the initial angles
        ring_initial_angles = np.hstack(
            [top_half_initial_angles, bottom_half_initial_angles]
        )

        # And now, put these in the data frame
        rings["start"] = ring_initial_angles

        # Compute the bin end angles
        ring_end_angles = np.roll(ring_initial_angles, -1)
        ring_end_angles[-1] = 180
        rings["end"] = ring_end_angles

        # Now, we need to compute the theta bin size for each ring
        theta_bin_size = 360 / all_patch_counts
        rings["theta_inc"] = theta_bin_size

        # And now, the face areas and weights
        face_areas = self._compute_face_areas(ring_initial_angles, all_patch_counts)
        rings["face_area"] = face_areas
        weights = face_areas.min() / face_areas
        rings["weight"] = weights

        # Indicate the regular and irregular rings
        regularity = np.array([True] * number_of_rings)
        regularity[:number_of_irregular_rings] = False
        regularity[-number_of_irregular_rings:] = False
        rings["regular"] = regularity

        return rings

    def to_dataframe(self) -> pd.DataFrame:
        """Get the data frame representation of the sphere.

        Returns
        -------
        pandas.DataFrame
            Representation of the sphere as a table. See **Notes** for
            details.

        Notes
        -----
        The :class:`pandas.DataFrame` contains as index the ring number.
        For each row, the following columns are present:

            * ``bins`` -- the number of bins in each ring;
            * ``start`` -- the starting phi value for each ring;
            * ``end`` -- the ending phi angle of each ring;
            * ``theta_inc`` -- the theta increment within each ring;
            * ``face_area`` -- the area of the faces in each ring;
            * ``weight`` -- the correction weight for the faces in each
                ring;
            * ``regular`` -- an indication of whether the ring has a
                regular almucantar spacing or not.
        """

        return self._rings.copy()

    def get_closest_faces(self, spherical_coordinates: pd.DataFrame) -> pd.DataFrame:
        """Get the closest faces for a specified spherical positions.

        Parameters
        ----------
        spherical_coordinates
            Coordinates containing ``phi`` and ``theta`` in **degrees**.

        Returns
        -------
        pandas.DataFrame
            The phi ``ring`` and theta ``bin`` for each vector.

        Notes
        --------
        To account for floating point errors, when determining the bin
        index within a ring, there is a rounding step. This step rounds the
        quotient of the angle and the bin width to a very small precision.
        This rounding is controlled by :attr:`binning_precision`.

        """

        # First, let's get the phi ring for each vector
        ring_end_angles = self._rings.loc[:, "end"]
        ring_end_angles = ring_end_angles.iloc[:-1]
        phi = spherical_coordinates.loc[:, "phi"]
        rings = ring_end_angles.searchsorted(phi, side="right")

        # Now, let's get the theta spacing for each respective ring.
        theta_increments = self._rings.loc[rings, "theta_inc"]
        theta = spherical_coordinates.loc[:, "theta"]
        theta_bin = np.floor(
            np.round(theta / theta_increments.to_numpy(), self.binning_precision)
        ).astype(int)

        closest_faces = pd.DataFrame({"ring": rings, "bin": theta_bin})

        return closest_faces

    def _initial_vector_component_preparation(
        self, vectors: pd.DataFrame
    ) -> pd.DataFrame:
        vectors_array = vectors.loc[:, ["vx", "vy", "vz"]].to_numpy()
        spherical_coordinates = util.compute_spherical_coordinates(
            vectors_array, use_degrees=True
        )
        spherical_coordinate_data_frame = pd.DataFrame(
            spherical_coordinates, columns=["phi", "theta", "magnitude"]
        )

        return spherical_coordinate_data_frame

    def _compute_orientation_binning(self, vectors: pd.DataFrame) -> pd.DataFrame:
        orientation_bins = self.get_closest_faces(vectors)
        return orientation_bins

    def _construct_orientation_index(self) -> pd.MultiIndex:
        """Get the orientation index for the current Tregenza sphere.

        Produce the histogram orientation index for the current Tregenza
        sphere.

        Returns
        -------
        pandas.MultiIndex
            Index containing all possible values of ring and bin in a given
            shell.
        """

        # Get the ring numbers and bin counts as an array
        ring_indices = self._rings.index.to_numpy()
        bin_counts = self._rings.loc[:, "bins"].to_numpy()

        # Create the indices for the ring number
        ring_indices = np.repeat(ring_indices, bin_counts)

        # Create the incremental bin indices
        bin_indices = np.concatenate([np.arange(i) for i in bin_counts])

        multi_index = pd.MultiIndex.from_arrays(
            [ring_indices, bin_indices], names=["ring", "bin"]
        )

        return multi_index

    def correct_histogram_by_area(self, histogram: pd.Series) -> pd.Series:
        """Correct histogram by face area.

        Weight histogram values by face areas to compensate for slight
        deviations from equal area.

        Parameters
        ----------
        histogram
            Histogram values to correct.

        Returns
        -------
        pandas.Series
            Corrected histogram values.

        """

        # Compute the weights
        ring_weights = self._rings.loc[:, "weight"]
        original_index = histogram.index
        weighted_face_data = (
            histogram.groupby("ring", group_keys=False).apply(
                lambda x: x, include_groups=False
            )
            * ring_weights
        )
        weighted_face_data = weighted_face_data.reindex(original_index)

        return weighted_face_data

    def create_mesh(self) -> pv.PolyData:
        """Create Tregenza sphere mesh.

        Warnings
        --------
        This mesh construction relies on an approximation that uses a
        rectangle (or polygon) for each patch. This causes some issues near
        the sphere poles. In the original work by Beckers and Beckers,
        [#Beckers]_ the cap at the pole was a circle and all other patches
        are sectors of the respective rings. In our implementation, the
        pole is a polygon with the number of vertices corresponding to the
        number of patches in the next row. The next row contains a small
        number of patches, which may appear thin or misshapen, leaving
        holes in the constructed mesh. These holes are only artifacts of
        the visualisation. In the conceptual calculation of the histogram
        binning, the holes do not exist.
        """
        # So, here's how we're going to do this...

        # Construct the inner rings.
        inner_ring_df = self._rings.iloc[1:-1]

        inner_ring_meshes = inner_ring_df.apply(
            self._construct_interior_ring, axis="columns"
        ).to_list()

        # Construct the caps
        top_cap = self._construct_cap(inner_ring_meshes[0], True, 0)
        bottom_cap = self._construct_cap(
            inner_ring_meshes[-1], False, self.number_of_rings - 1
        )

        # Combine all the meshes into a single list
        inner_ring_meshes.append(bottom_cap)

        # Merge everything
        # sphere_mesh = pv.merge([top_cap] + inner_ring_meshes + [bottom_cap])
        sphere_mesh = top_cap.append_polydata(*inner_ring_meshes)

        sphere_mesh = sphere_mesh.clean()

        return sphere_mesh

    def _construct_interior_ring(self, ring: pd.Series) -> pv.PolyData:
        """Construct a mesh for a single interior ring.

        Parameters
        ----------
        ring
            The parameters associated with the ring to construct, extracted
            from the :attr:`TregenzaSphere._rings` attribute.

        Returns
        -------
        pyvista.PolyData
            A mesh representing the constructed ring.

        Notes
        -----
        The mesh construction occurs carefully to ensure that the indexes
        of the vertices and faces are in the same order as the bins in any
        produced histogram data.
        """

        # First, get the basic information about the ring
        number_of_faces = ring["bins"]
        start_phi = ring["start"]
        end_phi = ring["end"]

        # Define the theta rings
        theta = np.linspace(0, 360, number_of_faces, endpoint=False)

        top_phi = np.tile(start_phi, number_of_faces)
        bottom_phi = np.tile(end_phi, number_of_faces)

        top_ring = np.stack([top_phi, theta], axis=-1)
        bottom_ring = np.stack([bottom_phi, theta], axis=-1)

        # Convert the rings to Cartesian coordinates
        top_ring_cartesian = util.convert_spherical_to_cartesian_coordinates(
            top_ring, use_degrees=True
        )
        bottom_ring_cartesian = util.convert_spherical_to_cartesian_coordinates(
            bottom_ring, use_degrees=True
        )

        # Combine everything
        ring_vertices = np.concatenate(
            [top_ring_cartesian, bottom_ring_cartesian], axis=0
        )

        # Now, establish the connectivity.
        vertex_count_column = 4 * np.ones(number_of_faces, dtype=int)
        top_left_corner = np.arange(number_of_faces)
        top_right_corner = np.roll(top_left_corner, -1)
        bottom_left_corner = top_left_corner + number_of_faces
        bottom_right_corner = top_right_corner + number_of_faces

        # And now to create a table
        cells = np.stack(
            [
                vertex_count_column,
                top_left_corner,
                top_right_corner,
                bottom_right_corner,
                bottom_left_corner,
            ],
            axis=-1,
        )

        # Begin building the mesh
        ring_mesh = pv.PolyData(ring_vertices, cells)

        # And now, to test, let's add the face indices as scalars
        face_indices = range(number_of_faces)

        ring_mesh.cell_data["ring-index"] = np.tile(ring.name, number_of_faces)
        ring_mesh.cell_data["face-index"] = face_indices

        return ring_mesh

    def _construct_cap(
        self, adjacent_ring: pv.PolyData, is_top: bool, index: int
    ) -> pv.PolyData:
        """Construct a Tregenza sphere cap.

        Parameters
        ----------
        adjacent_ring
            The ring immediately touching the cap.
        is_top
            Indicate whether the cap is on top of the provided ring.
        index
            The ring index.

        Returns
        -------
        pyvista.PolyData
            A single polygon representing the produced cap
        """

        number_of_vertices = adjacent_ring.n_points // 2

        # Get the vertices forming the cap.
        # If the cap is above, the first `number_of_vertices` form the cap,
        # otherwise the last `number_of_vertices` do.
        if is_top:
            vertices = adjacent_ring.points[:number_of_vertices]
        else:
            vertices = adjacent_ring.points[number_of_vertices:]

        cap_cell = np.hstack([number_of_vertices, np.arange(number_of_vertices)])

        # Now, create the mesh
        cap = pv.PolyData(vertices, cap_cell)

        cap.cell_data["ring-index"] = [index]
        cap.cell_data["face-index"] = [0]

        return cap

    @staticmethod
    def _compute_face_areas(
        start_angles: np.ndarray, patch_counts: np.ndarray
    ) -> np.ndarray:
        """Compute the face areas for each ring.

        Compute the face area for each ring the Tregenza sphere.

        Parameters
        ----------
        start_angles
            The initial angle for each ring.
        patch_counts
            The number of patches in each ring.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(r,)`` where ``r`` is the number of rings in
            the sphere. These weights reflect the deviation from equal
            area.

        Notes
        -----
        For each ring,the area of the spherical cap up to the bottom of the
        ring is calculated as

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

        To compute weights based on these values, all face areas must be
        normalised by the smallest face (typically the cap), and then the
        reciprocal of these values should be taken. These weights represent
        the ratio of the smallest patch to each patch, and can thus serve
        as a scaling factor between 0 and 1 to account for effects caused
        by the slight differences in face area.
        """

        # Determine the cumulative spherical cap areas
        end_angles = np.roll(start_angles, -1)
        end_angles[-1] = 180

        cumulative_areas = 2 * np.pi * (1 - np.cos(np.radians(end_angles)))

        preceding_areas = np.roll(cumulative_areas, 1)
        preceding_areas[0] = 0
        ring_areas = cumulative_areas - preceding_areas

        patch_areas = ring_areas / patch_counts

        return patch_areas

    def convert_vectors_to_cartesian_array(
        self,
        labelled_vectors: pd.DataFrame,
        create_unit_vectors: bool = False,
        include_spatial_coordinates: bool = False,
    ) -> np.ndarray:
        # Here's how this is going to go... We have spherical coordinates
        # with the columns `phi`, `theta` and `magnitude`. If we want unit
        # vectors, we just convert using phi and theta. If we don't, then
        # we throw in the magnitudes also.

        angular_coordinates = labelled_vectors[["phi", "theta"]].to_numpy()

        if create_unit_vectors:
            magnitudes = 1
        else:
            magnitudes = labelled_vectors["magnitude"].to_numpy()

        cartesian_coordinates = util.convert_spherical_to_cartesian_coordinates(
            angular_coordinates, radius=magnitudes, use_degrees=True
        )

        if include_spatial_coordinates:
            spatial_locations = labelled_vectors[["x", "y", "z"]].to_numpy()
            cartesian_coordinates = np.concatenate(
                [spatial_locations, cartesian_coordinates], axis=-1
            )

        return cartesian_coordinates


class CoarseTregenzaSphere(TregenzaSphere):
    """Coarse representation of the Tregenza sphere.

    This sphere is constructed with 18 rings (9 in each hemisphere). It
    contains a total of 520 patches.

    Notes
    -----
    This sphere design is based on the tiling presented by Beckers &
    Beckers. [#Beckers]_

    The almucantar spacing for the regular rings is 11.25 degrees. The
    first non-cap ring has a start almucantar angle of 5 degrees and an
    end angle of 11.25 degrees.
    """

    def __init__(
        self,
        number_of_shells: int = 1,
        magnitude_range: Optional[Tuple[float, float]] = None,
    ):
        # Define the almucantar angles (phi rings)
        irregular_phi_values = np.array([0.00, 5.0, 11.25])
        ring_increment = 11.25

        # Define the patch count
        patch_count = np.array(
            [
                1,
                4,
                15,
                24,
                32,
                39,
                45,
                49,
                51,
            ]
        )

        super().__init__(
            patch_count,
            irregular_phi_values,
            ring_increment,
            number_of_shells=number_of_shells,
            magnitude_range=magnitude_range,
        )


class FineTregenzaSphere(TregenzaSphere):
    """Fine representation of the Tregenza sphere.

    This sphere is constructed with 54 rings (27 in each hemisphere). It
    contains a total of 5806 patches.

    Notes
    -----
    This sphere design is based on the tiling presented by Beckers &
    Beckers. [#Beckers]_

    The almucantar spacing for the regular rings is 3.44 degrees. The
    first non-cap ring has a starting almucantar angle of 1.50 degrees and
    an ending angle of 4.00 degrees.
    """

    def __init__(
        self,
        number_of_shells: int = 1,
        magnitude_range: Optional[Tuple[float, float]] = None,
    ):
        # Define the almucantar angles (phi rings)
        irregular_phi_values = np.array([0.00, 1.50, 4.00])
        ring_increment = 3.44

        # Define the patch count
        patch_count = np.array(
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

        super().__init__(
            patch_count,
            irregular_phi_values,
            ring_increment,
            number_of_shells=number_of_shells,
            magnitude_range=magnitude_range,
        )


class UltraFineTregenzaSphere(TregenzaSphere):
    """Ultra-fine representation of the Tregenza sphere.

    This sphere is constructed with 124 rings (62 in each hemisphere). It
    contains a total of 36956 patches.

    Notes
    -----
    This sphere design is based on the tiling presented by Beckers &
    Beckers. [#Beckers]_

    The almucantar spacing for the regular rings is 1.47 degrees. The
    first non-cap ring has a starting almucantar angle of 0.60 degrees and
    an ending angle of 1.80 degrees.
    """

    def __init__(
        self,
        number_of_shells: int = 1,
        magnitude_range: Optional[Tuple[float, float]] = None,
    ):
        # Define the almucantar angles (phi rings)
        irregular_phi_values = np.array([0.00, 0.60, 1.80])
        ring_increment = 1.47

        # Define the patch count
        patch_count = np.array(
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

        super().__init__(
            patch_count,
            irregular_phi_values,
            ring_increment,
            number_of_shells=number_of_shells,
            magnitude_range=magnitude_range,
        )


# def run_tregenza_histogram_pipeline(
#     vectors: np.ndarray,
#     sphere: TregenzaSphere,
#     weight_by_magnitude: bool = False,
#     is_axial: bool = False,
#     remove_zero_vectors: bool = True,
#     correct_area_weighting: bool = True,
# ) -> List[np.ndarray]:
#     """Run the complete histogram construction for the Tregenza sphere.
#
#     Construct a spherical histogram based on a provided Tregenza sphere for
#     the supplied vectors.
#
#     Parameters
#     ----------
#     vectors
#         NumPy array containing the 3D vector components in the order
#         ``(x, y, z)``. This array should have shape ``(n, 3)`` where ``n``
#         is the number of vectors.
#     sphere
#         Tregenza sphere to use for histogram construction.
#     weight_by_magnitude
#         Indicate whether to weight the histogram by 3D magnitude. If
#         `False`, then the histogram will be weighted by count.
#     is_axial
#         Indicate whether the vectors are axial. If `True`, symmetric
#         vectors will be created based on the dataset.
#     remove_zero_vectors
#         Indicate whether to remove zero-vectors. This parameter should be
#         `True` unless the vector list contains no zero-vectors.
#     correct_area_weighting
#         Indicate whether the histogram values should be corrected using the
#         area weights.
#
#     Returns
#     -------
#     list[numpy.ndarray]
#         List of histogram counts for each ring. The list has the same
#         length as the number of rings in the provided Tregenza sphere,
#         and the length of each list entry corresponds to the respective
#         patch count.
#
#     """
#
#     # Perform vector pre-processing
#     if remove_zero_vectors:
#         vectors = util.remove_zero_vectors(vectors)
#
#     if is_axial:
#         vectors = util.convert_vectors_to_axes(vectors)
#         vectors = util.create_symmetric_vectors_from_axes(vectors)
#
#     angular_coordinates = util.compute_vector_orientation_angles(
#         vectors, use_degrees=True
#     )
#
#     if weight_by_magnitude:
#         _, magnitudes = util.normalise_vectors(vectors)
#     else:
#         magnitudes = None
#
#     histogram = sphere.construct_spherical_histogram(
#         spherical_coordinates=angular_coordinates, magnitudes=magnitudes
#     )
#
#     if correct_area_weighting:
#         histogram = sphere.correct_histogram_by_area(histogram)
#
#     return histogram
#
#
# class TregenzaSphereDetailLevel(enum.Enum):
#     """Detail level for Tregenza Sphere."""
#
#     FINE = "Fine"
#     """Fine-detail Tregenza sphere, with 27 rings per hemisphere,
#     represented by :class:`FineTregenzaSphere`."""
#
#     ULTRA_FINE = "Ultra-fine"
#     """Ultra-fine-detail Tregenza sphere, with 62 rings per hemisphere,
#     represented by :class:`UltraFineTregenzaSphere`."""
#
#     def get_tregenza_class(self) -> Type[TregenzaSphere]:
#         """Get the class for the specified level of detail.
#
#         Returns
#         -------
#         Type[TregenzaSphere]
#             Class inheriting from :class:`TregenzaSphere` which
#             captures the correct level of detail.
#
#         Warnings
#         --------
#         This method returns a **type**, not an object. The returned type
#         can be used to instantiate a Tregenza sphere object. To instantiate
#         an object directly, call :meth:`create_tregenza_sphere`.
#         """
#
#         if self == TregenzaSphereDetailLevel.FINE:
#             return FineTregenzaSphere
#         elif self == TregenzaSphereDetailLevel.ULTRA_FINE:
#             return UltraFineTregenzaSphere
#
#     def create_tregenza_sphere(self) -> TregenzaSphere:
#         """Create a Tregenza sphere for the specified detail level.
#
#         Returns
#         -------
#         TregenzaSphere
#             Object of the correct subclass of :class:`TregenzaSphere`
#             representing the desired level of detail.
#         """
#
#         tregenza_sphere_type = self.get_tregenza_class()
#
#         tregenza_sphere: TregenzaSphere = tregenza_sphere_type()
#
#         return tregenza_sphere
