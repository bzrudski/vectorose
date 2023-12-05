"""
Anisotropy Rose - main runner

Joseph Deering, Benjamin Rudski
2023

This module provides a command line tool that allows constructing
the 2D and 3D rose diagrams of anisotropy vector fields given a
vector field.

"""
from typing import Optional

import numpy as np

from .core import create_angular_binning_from_vectors
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
):
    """
    Run the entire anisotropy rose pipeline.

    Construct anisotropy roses from a set of vectors. For more details about each step, please consult the relevant
    functions.

    :param vectors: ``n`` by 6 or ``n`` by 3 array of vectors whose orientations will be analysed. If the vector
                    array contains 6 columns, **the last three are assumed to be the components**.
    :param half_number_of_bins: number of bins in :math:`\phi,\theta` in half the angular range.
    :param use_degrees: Indicates whether the angles should be computed in degrees. If ``True``, all angles will be
                        stored in degrees (default). Otherwise, all angles will be stored in radians.
    :param sphere_radius: Radius of the sphere for plotting.
    :param weight_by_magnitude: Indicate whether the histograms should be weighted by magnitude. If ``True``,
                                the :math:`\phi` histogram is weighted by the 3D magnitude and the :math:`\theta`
                                histogram is weighted by the magnitude in the :math:`(x,y)`-plane.
    :param zero_position_2d: The cardinal orientation of zero in the 2D polar histograms. Default: North.
    :param rotation_direction: The direction of increasing angles in the 2D polar histograms. Default: Clockwise.
    :param colour_map: Name of the matplotlib colour map to be used in the 3D hemisphere plot. If an invalid name is
                       specified, the default greyscale map ("gray") is used.
    :param plot_title: Title of the overall plot.
    :return: ``None``, but produces a figure on screen.

    **TODO: Add the ability to easily save from here.**
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
        plot_title=plot_title
    )
