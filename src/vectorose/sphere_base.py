"""Basics for spherical histogram construction.

This module contains basic tools for different representations of
optionally-nested spherical histograms.
"""
import abc
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pandas.core.generic
import pyvista as pv

from . import util


class SphereBase(abc.ABC):
    """Base class for a spherical histogram."""

    # Attributes
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

    @property
    def hist_group_cols(self) -> List[str]:
        """Names of the histogram columns to use for sorting."""
        return self.magnitude_shell_cols + self.orientation_cols

    @property
    def magnitude_shell_cols(self) -> List[str]:
        """Name of the histogram columns to use for magnitude."""
        return ["shell"]

    @property
    @abc.abstractmethod
    def orientation_cols(self) -> List[str]:
        """Name of the histogram columns to use for orientation."""
        raise NotImplementedError(
            "This abstract property must be implemented in subclasses."
        )

    def __init__(
        self,
        number_of_shells: int = 1,
        magnitude_range: Optional[Tuple[float, float]] = None,
    ):
        self.number_of_shells = number_of_shells
        self.magnitude_range = magnitude_range

    def assign_histogram_bins(
        self, vectors: np.ndarray
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Assign vectors to the appropriate histogram bin.

        Parameters
        ----------
        vectors
            Array of shape ``(n, 3)`` containing the Cartesian components
            of the vectors from which to construct the histogram.

        Returns
        -------
        pandas.DataFrame
            All the vectors, including additional columns for the shell and
            the implementation-specific orientation bin.
        numpy.ndarray
            Histogram bin edges for the magnitude shells.

        Warnings
        --------
        All zero-vectors must be removed from the dataset before
        processing. These vectors have no orientation and thus cannot be
        properly assigned to an orientation bin.
        """

        # Create the vector data frame
        histogram = util.convert_vectors_to_data_frame(vectors)

        # Perform any additional histogram preparation
        histogram = self._initial_vector_data_preparation(histogram)

        # Perform the magnitude computations
        magnitude_bins, magnitude_bin_edges = self._compute_magnitude_bins(histogram)
        histogram = pd.concat([histogram, magnitude_bins], axis=1)

        # Perform the orientation binning
        orientation_bins = self._compute_orientation_binning(histogram)
        histogram = pd.concat([histogram, orientation_bins], axis=1)

        # Return the complete histogram
        return histogram, magnitude_bin_edges

    def _initial_vector_data_preparation(self, vectors: pd.DataFrame) -> pd.DataFrame:
        """Prepare the vectors for histogram construction.

        Convert the vectors into a representation specific for the
        histogram spherical implementation. If spatial coordinates are
        provided for the vectors, these are preserved.

        Parameters
        ----------
        vectors
            DataFrame with ``n`` rows and either 3 or 6 columns. The
            required vector component columns are ``vx, vy, vz``. Optional
            spatial coordinate columns are ``x, y, z``. It is preferrable
            (but not required) for the spatial columns to be the first 3
            columns.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing ``n`` rows and a subclass-specific
            number of columns. The columns represent an alternative
            representation of the vectors to assist in orientation binning.
            If spatial coordinate columns were present in the original data
            they will be preserved in the output.

        Warnings
        --------
        This method should typically **not** be overridden. The
        implementation-specific functionality should be written in the
        method :meth:`_initial_vector_component_preparation`, which is
        called by this function.
        """

        processed_vector_data = self._initial_vector_component_preparation(vectors)

        # Add back the locations, if necessary
        number_of_columns = len(vectors.columns)

        if number_of_columns > 3:
            location_data = vectors.loc[:, ["x", "y", "z"]]
            processed_vector_data = location_data.join(processed_vector_data)

        return processed_vector_data

    def _initial_vector_component_preparation(
        self, vectors: pd.DataFrame
    ) -> pd.DataFrame:
        """Prepare the vector components for histogram construction.

        Override this method to include specific operations that should be
        performed on the vectors in order to construct the histogram in the
        specific implementation.

        Warnings
        --------
        This function should **not** perform any tasks related to the
        vector spatial locations, if those are included in the data. Those
        are handled separately by :meth:`._initial_vector_data_preparation`
        which calls this function.
        """

        vector_components = vectors.loc[:, ["vx", "vy", "vz"]]

        return vector_components

    def _compute_magnitude_bins(
        self, vectors: pd.DataFrame
    ) -> Tuple[pd.Series, np.ndarray]:
        """Perform binning based on magnitude.

        Construct the magnitude histogram for the provided vectors.

        Parameters
        ----------
        vectors
            The vectors from which the magnitude histogram is to be
            constructed.

        Returns
        -------
        pandas.Series
            The magnitude shell number for each vector, in a
            :class:`pandas.Series` called ``shell``.
        numpy.ndarray
            Array containing the histogram bin boundaries used to construct
            the histogram. The length of this array corresponds is one more
            than :attr:`SphereBase.number_of_shells`.
        """
        magnitudes = vectors.loc[:, "magnitude"]

        # Define the magnitude bin edges
        if self.number_of_shells > 1:
            if self.magnitude_range is None:
                offset = 10 ** -(self.magnitude_precision or 8)
                max_magnitude = magnitudes.max() + offset
                min_magnitude = magnitudes.min() - offset
                magnitude_range = (min_magnitude, max_magnitude)
            else:
                magnitude_range = self.magnitude_range
            magnitude_bin_edges = np.histogram_bin_edges(
                magnitudes, bins=self.number_of_shells, range=magnitude_range
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
            # Set the minimum bin to be the smallest dataset value
            lower_bin = magnitudes.min()

            # Set the upper bin to be just above the maximum value
            upper_bin = magnitudes.max() + (10 ** -(self.magnitude_precision - 1))
            magnitude_bin_edges = np.array([lower_bin, upper_bin])

        magnitude_histogram_bins = pd.Series(magnitude_histogram_bins, name="shell")

        return magnitude_histogram_bins, magnitude_bin_edges

    @abc.abstractmethod
    def _compute_orientation_binning(
        self, vectors: pd.DataFrame
    ) -> pd.core.generic.NDFrame:
        """Bin the provided vectors based on orientation.

        Parameters
        ----------
        vectors
            The vectors to place in orientation bins.

        Returns
        -------
        pandas.Series or pandas.DataFrame
            The orientation bin(s) corresponding to each vector. The number
            of columns will depend on the specific sphere representation
            used.
        """

        raise NotImplementedError("Subclasses must implement this abstract method!")

    def _construct_histogram_index(self) -> pd.MultiIndex:
        """Construct the index for the histogram."""

        magnitude_index = self._construct_magnitude_index()
        orientation_index = self._construct_orientation_index()

        magnitude_index_arr = magnitude_index.to_frame(index=False).to_numpy()
        orientation_index_arr = orientation_index.to_frame(index=False).to_numpy()

        number_of_shells = len(magnitude_index_arr)
        number_of_orientations = len(orientation_index_arr)

        # Repeat each index
        magnitude_index_complete = np.repeat(
            magnitude_index_arr, number_of_orientations, axis=0
        )
        orientation_index_complete = np.tile(
            orientation_index_arr, (number_of_shells, 1)
        )

        raw_index_arrays = [magnitude_index_complete, orientation_index_complete]
        headers_arrays = [self.magnitude_shell_cols, self.orientation_cols]

        # And now combine everything!
        index_array = np.concatenate(raw_index_arrays, axis=-1)
        headers = np.concatenate(headers_arrays)

        # And build a data frame from this
        index_data_frame = pd.DataFrame(index_array, columns=headers)

        multi_index = pd.MultiIndex.from_frame(index_data_frame)

        return multi_index

    def _construct_magnitude_index(self) -> pd.Index:
        """Construct the index for the magnitude bins."""

        index = pd.RangeIndex(
            start=0, stop=self.number_of_shells, name=self.magnitude_shell_cols[0]
        )

        return index

    @abc.abstractmethod
    def _construct_orientation_index(self) -> pd.Index:
        """Construct the index for the orientation bins."""

        raise NotImplementedError("Subclasses must implement this abstract method!")

    def construct_histogram(
        self,
        binned_data: pd.DataFrame,
        return_fraction: bool = True,
    ) -> pd.Series:
        """Construct a histogram based on the labelled data.

        Using the binned data, construct a histogram with either the counts
        or the proportion of points in each face.

        Parameters
        ----------
        binned_data
            All vectors, with their respective bins, depending on the
            current sphere design.
        return_fraction
            Indicate whether the values returned should be the raw counts
            or the proportions.

        Returns
        -------
        pandas.Series
            The counts or proportions of vectors in each case, ordered by
            the columns specified in
            :attr:`SphereBase.hist_group_cols`.
        """

        grouping_columns = self.hist_group_cols

        # Use groupby to perform the grouping
        original_histogram = binned_data.groupby(grouping_columns).apply(
            len, include_groups=False
        )

        # Modify the index to account for any missing bins.
        multi_index = self._construct_histogram_index()

        filled_histogram = original_histogram.reindex(index=multi_index, fill_value=0)
        filled_histogram.name = "frequency"

        if return_fraction:
            number_of_vectors = len(binned_data)

            filled_histogram /= number_of_vectors

        return filled_histogram

    def construct_marginal_magnitude_histogram(
        self, binned_data: pd.DataFrame, return_fraction: bool = True
    ) -> pd.Series:
        """Construct the marginal magnitude histogram.

        Compute the marginal histogram of the magnitude data, disregarding
        the orientation differences. The resulting histogram has the same
        number of bins as the number of shells.

        Parameters
        ----------
        binned_data
            Data frame containing the labelled vectors.
        return_fraction
            Indicate whether the values returned should be the raw counts
            or the proportions.

        Returns
        -------
        pandas.Series
            The counts or proportions of vectors in each magnitude shell.

        See Also
        --------
        SphereBase.assign_histogram_bins:
            Label a set of vectors into different bins.
        SphereBase.construct_histogram:
            Construct a bivariate magnitude and orientation histogram.
        SphereBase.construct_marginal_orientation_histogram:
            Construct a marginal orientation histogram.
        """

        # Group based only on the magnitude bin
        counts_by_shell = binned_data.groupby(self.magnitude_shell_cols).apply(
            len, include_groups=False
        )

        # Construct the index (in case some bins are zero).
        magnitude_index = self._construct_magnitude_index()

        magnitude_histogram = counts_by_shell.reindex(
            index=magnitude_index, fill_value=0
        )

        if return_fraction:
            number_of_vectors = len(binned_data)

            magnitude_histogram /= number_of_vectors

        return magnitude_histogram

    def construct_marginal_orientation_histogram(
        self, binned_data: pd.DataFrame, return_fraction: bool = True
    ) -> pd.Series:
        """Construct the marginal orientation histogram.

        Compute the marginal histogram of the orientation data,
        disregarding the magnitude differences. The resulting histogram has
        the same configuration of bins as a single shell.

        Parameters
        ----------
        binned_data
            Data frame containing the labelled vectors.
        return_fraction
            Indicate whether the values returned should be the raw counts
            or the proportions.

        Returns
        -------
        pandas.Series
            The counts or proportions of vectors in each orientation bin.

        See Also
        --------
        SphereBase.assign_histogram_bins:
            Label a set of vectors into different bins.
        SphereBase.construct_histogram:
            Construct a bivariate magnitude and orientation histogram.
        SphereBase.construct_marginal_magnitude_histogram:
            Construct a marginal magnitude histogram.
        """

        # Group based on only the orientation data
        counts_by_orientation = binned_data.groupby(self.orientation_cols).apply(
            len, include_groups=False
        )

        # Construct the index (in case some orientations are zero).
        orientation_index = self._construct_orientation_index()
        orientation_index.names = self.orientation_cols

        orientation_histogram = counts_by_orientation.reindex(
            orientation_index, fill_value=0
        )

        if return_fraction:
            number_of_vectors = len(binned_data)

            orientation_histogram /= number_of_vectors

        return orientation_histogram

    def construct_conditional_orientation_histogram(
        self, binned_data: pd.DataFrame
    ) -> pd.Series:
        """Construct the conditional orientation histogram.

        Construct the histogram of orientations conditioned on the
        magnitude. Within each shell, the returned fractions sum to 1.

        Parameters
        ----------
        binned_data
            Data frame containing the labelled vectors.

        Returns
        -------
        pandas.Series
            The proportion of vectors in each orientation relative to all
            vectors within that shell. The index used is the same as that
            obtained in the bivariate case.

        Warnings
        --------
        Unlike the bivariate and marginal histograms, this method does not
        allow returning raw counts. The returned values are proportions
        relative to each shell.
        """

        # Get the bivariate histogram with the counts
        bivariate_histogram = self.construct_histogram(
            binned_data, return_fraction=False
        )

        # And now, get the marginal magnitude histogram
        marginal_magnitude_histogram = self.construct_marginal_magnitude_histogram(
            binned_data, return_fraction=False
        )

        # Divide the bivariate distribution by the marginal to get the
        # conditional distribution.
        orientation_given_magnitude = bivariate_histogram / marginal_magnitude_histogram

        return orientation_given_magnitude

    def construct_conditional_magnitude_histogram(
        self, binned_data: pd.DataFrame
    ) -> pd.Series:
        """Construct the conditional magnitude histogram.

        Construct the histogram of magnitudes conditioned on the
        orientation. Within each orientation bin, the returned fractions
        sum to 1.

        Parameters
        ----------
        binned_data
            Data frame containing the labelled vectors.

        Returns
        -------
        pandas.Series
            The proportion of vectors in each magnitude shell relative to
            all vectors having that orientation. The index used is the
            same as that obtained in the bivariate case, having the
            magnitude first, followed by the orientation parameters.

        Warnings
        --------
        Unlike the bivariate and marginal histograms, this method does not
        allow returning raw counts. The returned values are proportions
        relative to each shell.
        """

        # Get the bivariate histogram with the counts
        bivariate_histogram = self.construct_histogram(
            binned_data,
            return_fraction=False,
        )

        # And now, get the marginal magnitude histogram
        marginal_orientation_histogram = self.construct_marginal_orientation_histogram(
            binned_data, return_fraction=False
        )

        # Divide the bivariate distribution by the marginal to get the
        # conditional distribution.
        magnitude_given_orientation = (
            bivariate_histogram / marginal_orientation_histogram
        )

        return magnitude_given_orientation

    @abc.abstractmethod
    def create_mesh(self) -> pv.PolyData:
        """Return the mesh representation of the current sphere."""

        raise NotImplementedError("Subclasses must implement this abstract method!")

    def create_shell_mesh(
        self,
        histogram: pd.Series,
        radius: float = 1.0,
        series_name: Optional[str] = "frequency",
    ) -> pv.PolyData:
        """Create the mesh for a given shell.

        Using the provided histogram data for a specific shell, produce a
        sphere with the desired radius, storing the frequencies as face
        values.

        Parameters
        ----------
        histogram
            The counts or frequencies of orientations in each sphere face
            of the specific shell.
        radius
            Desired shell radius. This typically corresponds to a magnitude
            bin upper bound.
        series_name
            The name to associate with the provided scalar data. If `None`,
            then the value of :attr:`pandas.Series.name` is used.

        Returns
        -------
        pyvista.PolyData
            The constructed shell containing the desired scalars in the
            specified slot.
        """

        # First, construct the mesh that will underlie the shell
        shell_mesh = self.create_mesh()

        # Now, adjust the radius
        shell_mesh = shell_mesh.scale(radius)

        # Get the name
        series_name = series_name or histogram.name

        # Set the scalar values
        shell_mesh.cell_data[series_name] = histogram.astype(float)

        return shell_mesh

    def create_histogram_meshes(
        self,
        histogram_data: pd.Series,
        magnitude_bins: Optional[np.ndarray],
        normalise_by_shell: bool = False,
    ) -> List[pv.PolyData]:
        """Create mesh shells for the supplied histogram.

        Parameters
        ----------
        histogram_data
            The binned histogram data, ordered by shell and then other
            implementation-specific parameters.
        magnitude_bins
            The upper bounds for the magnitude bins. These are used to
            determine the radius of each shell. If None, then all shells
            will have a radius of 1.
        normalise_by_shell
            Indicate whether each shell should be normalised with respect
            to its maximum value.


        Returns
        -------
        list of pyvista.PolyData
            List containing one mesh for each shell, with the appropriate
            scalar values assigned to the ``frequency`` array.

        Warnings
        --------
        The provided histogram must have been constructed with the current
        sphere, or an equivalent sphere.

        Notes
        -----
        The option `normalise_by_shell` produces meshes where the faces
        values are divided by the maximum value in their corresponding
        shell. The values can therefore be thought of as representing
        fractions of the respective maxima.
        """

        number_of_shells = self.number_of_shells

        if normalise_by_shell:
            shell_maxima = histogram_data.groupby("shell").max()

            # Warning! Assignment operator /= changes the original!
            normalised_data = histogram_data / shell_maxima
            histogram_data = normalised_data

        shell_list = []

        for i in range(number_of_shells):
            shell_histogram = histogram_data.loc[i]
            shell_radius = 1 if magnitude_bins is None else magnitude_bins[i + 1]

            shell = self.create_shell_mesh(shell_histogram, shell_radius)

            shell_list.append(shell)

        return shell_list

    @abc.abstractmethod
    def convert_vectors_to_cartesian_array(
        self,
        labelled_vectors: pd.DataFrame,
        create_unit_vectors: bool = False,
        include_spatial_locations: bool = False,
    ) -> np.ndarray:
        """Convert a set of labelled vectors into Cartesian coordinates.

        Each concrete implementation of a sphere may internally represent
        the vectors differently. This abstract method converts from that
        implementation-specific formatting to Cartesian coordinates.

        Parameters
        ----------
        labelled_vectors
            The set of labelled ``n`` labelled vectors in ``d`` dimensions,
            in the same format as produced by
            :meth:`SphereBase.assign_histogram_bins`.
        create_unit_vectors
            Indicate where the returned vectors should be unit vectors.
            Depending on the implementation, this may either remove an
            extraneous normalisation step later, or add an extra
            normalisation step now.
        include_spatial_locations
            Indicate whether to include spatial coordinates in the new
            array. This option may only be called if the vectors have
            spatial coordinates.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n, d)`` containing the vector components in
            Cartesian coordinates.

        Warnings
        --------
        The option `include_spatial_coordinates` is only valid if the
        `labelled_vectors` include spatial coordinates.
        """

        raise NotImplementedError(
            "This abstract method must be implemented in subclasses."
        )
