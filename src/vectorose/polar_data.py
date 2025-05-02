# Copyright (c) 2023-, Benjamin Rudski, Joseph Deering
#
# This code is licensed under the MIT License. See the `LICENSE` file for
# more details about copying.

"""Polar data histogram calculations.

Compute 1D histograms for phi and theta angles separately.

Warnings
--------
Currently, this polar analysis is purely based on orientation and cannot
be done in conjunction with studies of magnitude.
"""

import numpy as np
import pandas as pd

from . import util


class PolarDiscretiser:
    """Construct polar histograms based on vectorial and axial data."""

    _is_axial: bool
    """Indicate whether the data are axial.
    
    This attribute is used to determine the range of the phi values.
    """

    _phi_bins: pd.DataFrame
    """Definition of the phi bins."""

    _theta_bins: pd.DataFrame
    """Definition of the theta bins."""

    _phi_increment: float
    """Angular width of the phi bins."""

    _theta_increment: float
    """Angular width of the theta bins."""

    binning_precision: int = 10
    """Rounding precision when assigning bin values."""

    @property
    def number_of_phi_bins(self) -> int:
        """Number of bins to capture the phi angles."""
        return len(self._phi_bins)

    @property
    def number_of_theta_bins(self) -> int:
        """Number of bins to capture the theta angles."""
        return len(self._theta_bins)

    @property
    def phi_increment(self) -> float:
        """Angular width of the phi bins."""
        return self._phi_increment

    @property
    def theta_increment(self) -> float:
        """Angular width of the theta bins."""
        return self._theta_increment

    @property
    def is_axial(self) -> bool:
        """Indicate whether the discretiser is considering axial data."""
        return self._is_axial

    def __init__(
        self, number_of_phi_bins: int, number_of_theta_bins: int, is_axial: bool
    ):
        # Define the angular bins for phi
        phi_bins, phi_increment = self._generate_phi_bins(number_of_phi_bins, is_axial)
        self._phi_bins = phi_bins
        self._phi_increment = phi_increment

        # Define the angular bins for theta
        theta_bins, theta_increment = self._generate_theta_bins(number_of_theta_bins)
        self._theta_bins = theta_bins
        self._theta_increment = theta_increment

        self._is_axial = is_axial

    @staticmethod
    def _generate_theta_bins(
        number_of_theta_bins: int,
    ) -> tuple[pd.DataFrame, np.floating]:
        """Generate the theta bins.

        The theta bins fill the interval from 0 degrees (included) to 360
        degrees (excluded). The last bin wraps around to zero degrees.

        Parameters
        ----------
        number_of_theta_bins
            Number of bins to generate in the theta direction.

        Returns
        -------
        theta_bins : pandas.DataFrame
            Bin start and end angles, containing `number_of_theta_bins`
            rows.
        theta_step : np.floating
            Angular width of each bin.
        """

        start_theta = 0
        end_theta = 360
        theta_bin_edges, theta_step = np.linspace(
            start_theta, end_theta, number_of_theta_bins, endpoint=False, retstep=True
        )
        theta_bin_starts = theta_bin_edges
        theta_bin_ends = np.roll(theta_bin_starts, -1)
        theta_bin_array = np.vstack([theta_bin_starts, theta_bin_ends]).T
        theta_bins = pd.DataFrame(theta_bin_array, columns=["start", "end"])
        return theta_bins, theta_step

    @staticmethod
    def _generate_phi_bins(
        number_of_phi_bins: int, is_axial: bool
    ) -> tuple[pd.DataFrame, np.floating]:
        """Generate the phi bins.

        The phi bins fill the interval from 0 degrees (included) to either
        90 degrees in the case of axial data, or 180 degrees in the case of
        vectorial data. This upper bound is included, the last bin starts
        at this final value to account for vectors at this value.

        Parameters
        ----------
        number_of_phi_bins
            Number of bins to generate in the phi direction.
        is_axial
            Indicate whether to consider data as axial. In this case, only
            the phi values corresponding to the upper hemisphere are
            considered.

        Returns
        -------
        theta_bins : pandas.DataFrame
            Bin start and end angles, containing `number_of_theta_bins`
            rows.
        theta_step : np.floating
            Angular width of each bin.
        """

        start_phi = 0
        end_phi = 90 if is_axial else 180
        phi_bin_edges, phi_step = np.linspace(
            start_phi, end_phi, number_of_phi_bins, endpoint=True, retstep=True
        )

        phi_bin_starts = phi_bin_edges
        phi_bin_ends = phi_bin_starts + phi_step
        phi_bin_array = np.vstack([phi_bin_starts, phi_bin_ends]).T
        phi_bins = pd.DataFrame(phi_bin_array, columns=["start", "end"])
        return phi_bins, phi_step

    def assign_histogram_bins(self, vectors: np.ndarray) -> pd.DataFrame:
        """Assign histogram orientation bins.

        Label each provided vector with a bin index in phi and theta.

        Parameters
        ----------
        vectors
            Array of shape ``(n, 3)`` or ``(n, 6)`` containing the
            Cartesian components of the vectors from which to construct the
            histogram. If 6 columns are present, the first 3 are assumed
            to be the spatial locations.

        Returns
        -------
        pandas.DataFrame
            All the vectors, including additional columns for the phi and
            theta bins.

        Warnings
        --------
        All zero-vectors must be removed from the dataset before
        processing. These vectors have no orientation and thus cannot be
        properly assigned to an orientation bin.
        """

        ncols = vectors.shape[-1]

        columns = ["vx", "vy", "vz"]

        components = vectors

        if ncols == 6:
            columns = ["x", "y", "z"] + columns
            components = vectors[:, 3:]

        # Build up the data frame
        vector_data_frame = pd.DataFrame(vectors, columns=columns)

        # Convert the spherical coordinates
        spherical_coordinates = util.compute_spherical_coordinates(components, True)

        spherical_coordinates_data_frame = pd.DataFrame(
            spherical_coordinates, columns=["phi", "theta", "magnitude"]
        )

        vector_data_frame = pd.concat(
            [vector_data_frame, spherical_coordinates_data_frame], axis=1
        )

        # Compute the phi bins
        phi_spacing = self._phi_increment
        phi = spherical_coordinates[:, util.AngularIndex.PHI]
        phi_indices = np.floor(
            np.round(phi / phi_spacing, self.binning_precision)
        ).astype(int)

        vector_data_frame["phi_bin"] = phi_indices

        # Compute the theta bins
        theta_spacing = self._theta_increment
        theta = spherical_coordinates[:, util.AngularIndex.THETA]
        theta_indices = np.floor(
            np.round(theta / theta_spacing, self.binning_precision)
        ).astype(int)

        vector_data_frame["theta_bin"] = theta_indices

        return vector_data_frame

    def _construct_histogram(
        self,
        labelled_vectors: pd.DataFrame,
        angle_name: util.AngleName,
    ) -> pd.DataFrame:
        """Construct polar histogram for one of the two angles.

        Parameters
        ----------
        labelled_vectors
            Vectors with phi and theta bins assigned.
        angle_name
            Indicate which angle should be considered

        Returns
        -------
        pandas.DataFrame
            Histogram containing each bin start, end, count and frequency
            values.
        """
        bin_assignment_column = f"{angle_name.value}_bin"

        if angle_name == util.AngleName.PHI:
            bins = self._phi_bins
        else:
            bins = self._theta_bins

        # Count the vectors in each bin
        counts = labelled_vectors.groupby(bin_assignment_column).apply(
            len, include_groups=False
        )
        counts.name = "count"

        # Normalise to get the frequencies
        number_of_vectors = len(labelled_vectors)
        frequencies = counts / number_of_vectors
        frequencies.name = "frequency"

        # Combine everything to get a nice histogram
        counts = counts.reindex(index=bins.index, fill_value=0)
        frequencies = frequencies.reindex(index=bins.index, fill_value=0)
        histogram = pd.concat([bins, counts, frequencies], axis=1)

        return histogram

    def construct_phi_histogram(
        self,
        labelled_vectors: pd.DataFrame,
    ) -> pd.DataFrame:
        """Construct the phi polar histogram.

        Parameters
        ----------
        labelled_vectors
            Vectors with phi and theta bins assigned.

        Returns
        -------
        pandas.DataFrame
            Histogram containing each bin start, end, count and frequency
            values for each phi bin.
        """

        angle_name = util.AngleName.PHI

        return self._construct_histogram(labelled_vectors, angle_name)

    def construct_theta_histogram(
        self,
        labelled_vectors: pd.DataFrame,
    ) -> pd.DataFrame:
        """Construct the theta polar histogram.

        Parameters
        ----------
        labelled_vectors
            Vectors with theta and theta bins assigned.

        Returns
        -------
        pandas.DataFrame
            Histogram containing each bin start, end, count and frequency
            values for each theta bin.
        """

        angle_name = util.AngleName.THETA

        return self._construct_histogram(labelled_vectors, angle_name)
