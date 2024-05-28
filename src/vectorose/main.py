# Copyright (c) 2023-, Benjamin Rudski, Joseph Deering
#
# This code is licensed under the MIT License. See the `LICENSE` file for
# more details about copying.

"""
Main runner for vector rose construction

This module provides a command line tool that allows constructing
the 2D and 3D vector rose for a specified a vector field.

Todo
----
* Add command line parser.
* Add ability to save vector rose figures.

"""
from typing import Dict, Optional, Any

import numpy as np

from .vectorose import create_angular_binning_from_vectors
from .plotting import CardinalDirection, RotationDirection, produce_histogram_plots


def perform_anisotropy_rose_pipeline(
    vectors: np.ndarray,
    half_number_of_bins: int = 18,
    use_degrees: bool = True,
    sphere_radius: float = 2.0,
    weight_by_magnitude: bool = True,
    zero_position_2d: CardinalDirection = CardinalDirection.NORTH,
    rotation_direction: RotationDirection = RotationDirection.CLOCKWISE,
    colour_map: str = "gray",
    plot_title: Optional[str] = None,
    **kwargs: Dict[str, Any],
):
    """
    Run the entire anisotropy rose pipeline.

    Construct anisotropy roses from a set of vectors. For more details
    about each step, please consult the relevant functions.

    Parameters
    ----------

    vectors
        ``n`` by 6 or ``n`` by 3 array of vectors whose orientations will
        be analysed. If the vector array contains 6 columns, **the last
        three are assumed to be the components**.

    half_number_of_bins
        number of bins in :math:`\\phi,\\theta` in half the angular range.

    use_degrees
        Indicates whether the angles should be computed in degrees. If
        ``True``, all angles will be stored in degrees (default).
        Otherwise, all angles will be stored in radians.

    sphere_radius
        Radius of the sphere for plotting.

    weight_by_magnitude
        Indicate whether the histograms should be weighted by magnitude.
        If ``True``, the :math:`\\phi` histogram is weighted by the 3D
        magnitude and the :math:`\\theta` histogram is weighted by the
        magnitude in the :math:`(x,y)`-plane.

    zero_position_2d
        The cardinal orientation of zero in the 2D polar histograms.
        Default: :attr:`CardinalDirection.North`.

    rotation_direction
        The direction of increasing angles in the 2D polar histograms.
        Default: :attr:`RotationDirection.Clockwise`.

    colour_map
        Name of the matplotlib colour map to be used in the 3D hemisphere
        plot. If an invalid name is specified, the default greyscale map
        ("gray") is used.

    plot_title
        Title of the overall plot.

    **kwargs
        Additional keyword arguments for plotting.

    Todo
    ----
    * Add the ability to save plots from here.
    """

    binned_data, bins = create_angular_binning_from_vectors(
        vectors=vectors,
        half_number_of_bins=half_number_of_bins,
        use_degrees=use_degrees,
    )

    produce_histogram_plots(
        binned_data=binned_data,
        bins=bins,
        sphere_radius=sphere_radius,
        weight_by_magnitude=weight_by_magnitude,
        zero_position_2d=zero_position_2d,
        rotation_direction=rotation_direction,
        use_degrees=use_degrees,
        colour_map=colour_map,
        plot_title=plot_title,
        **kwargs,
    )
