"""Basics for spherical histogram construction.

This module contains basic tools for different representations of
optionally-nested spherical histograms.
"""
import abc
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pandas.core.generic


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
    @abc.abstractmethod
    def hist_group_cols(self) -> List[str]:
        """Names of the histogram columns to use for sorting."""
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
        """

        # Create the histogram
        histogram = pd.DataFrame(vectors, columns=["x", "y", "z"])

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

    def _initial_vector_data_preparation(
        self, vectors: pd.DataFrame
    ) -> pd.DataFrame:
        """Prepare the vectors for histogram construction.

        Override this method to include specific operations that should be
        performed on the vectors in order to construct the histogram in the
        specific implementation.
        """
        return vectors

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
            the histogram. The length of this array corresponds to
            :attr:`SphereBase.number_of_shells`.
        """
        magnitudes = vectors.loc[:, "magnitude"]

        # Define the magnitude bin edges
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

        raise NotImplementedError(
            "Subclasses must implement this abstract method!"
        )

    @abc.abstractmethod
    def _construct_histogram_index(self) -> pd.MultiIndex:
        """Construct the index for the histogram."""

        raise NotImplementedError(
            "Subclass must implement this abstract method!"
        )

    def construct_histogram(
        self,
        binned_data: pd.DataFrame,
        return_fraction: bool = True
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
            the columns specified in :prop:`SphereBase.hist_group_cols`.
        """

        # Get the total number of vectors
        number_of_vectors = len(binned_data)

        # Use groupby to perform the grouping
        original_histogram = binned_data.groupby(self.hist_group_cols).apply(len)

        # Modify the index to account for any missing bins.
        multi_index = self._construct_histogram_index()

        filled_histogram = original_histogram.reindex(index=multi_index, fill_value=0)

        if return_fraction:
            filled_histogram /= number_of_vectors

        return filled_histogram