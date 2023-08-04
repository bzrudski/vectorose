"""
Anisotropy Rose

Joseph Deering, Benjamin Rudski
2023

This package provides the ability to construct 2D and 3D rose diagrams of anisotropy vector fields.

"""

import enum
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


class MagnitudeType(enum.IntEnum):
    """
    Type of magnitude.

    Type of magnitude considered when constructing the histograms.
    """
    THREE_DIMENSIONAL = 0
    IN_PLANE = 1


class CardinalDirection(enum.Enum):
    """
    Cardinal Directions

    This string-based enumerated type is useful when preparing 2D polar figures.

    See: https://matplotlib.org/stable/api/projections/polar.html#matplotlib.projections.polar.PolarAxes.set_theta_zero_location
    """
    NORTH = "N"
    NORTH_WEST = "NW"
    WEST = "W"
    SOUTH_WEST = "SW"
    SOUTH = "S"
    SOUTH_EAST = "SE"
    EAST = "E"
    NORTH_EAST = "NE"


class RotationDirection(enum.Enum):
    """
    Rotation Direction

    This int-based enumerated type defines two members:

    * Clockwise: -1
    * Counter-clockwise / anti-clockwise: 1

    See: https://matplotlib.org/stable/api/projections/polar.html#matplotlib.projections.polar.PolarAxes.set_theta_direction
    """
    CLOCKWISE = -1
    COUNTER_CLOCKWISE = 1


def remove_zero_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Prune zero-vectors.

    Remove vectors of zero magnitude from the list of vectors.

    :param vectors: ``n`` by 6 or ``n`` by 3 array of vectors. If the array has 6 columns, **the last 3 are assumed
                    to be the vector components**.
    :return: list of vectors with the same number of columns as the original, without any vectors of zero magnitude.
    """


def convert_spherical_to_cartesian_coordinates(angular_coordinates: np.ndarray, radius: float = 1) -> np.ndarray:
    """
    Convert spherical coordinates to cartesian coordinates.

    Convert spherical coordinates provided in terms of phi and theta into cartesian coordinates. For the conversion
    to be possible, a sphere radius must also be specified. If none is provided, the sphere is assumed to be the
    unit sphere.

    The input is provided as a 2D array with 2 columns representing the angles phi and theta, and ``n`` rows,
    representing the datapoints. The returned array is also a 2D array, with three columns (X, Y, Z) and ``n`` rows.

    :param angular_coordinates: Array with 2 columns representing phi and theta, respectively, and ``n`` rows
                                representing the datapoints.
    :param radius: A single float representing the radius of the sphere (default: unit radius).
    :return: Array with 3 columns, corresponding to the cartesian coordinates in X, Y, Z, and ``n`` rows,
             one for each data point.
    """


def compute_vector_orientation_angles(vectors: np.ndarray, use_degrees: bool = False) -> np.ndarray:
    """
    Compute the vector orientation angles phi and theta.

    For all vectors passed in ``vectors``, compute the :math:`\\phi` and :math:`\\theta` orientation angles. We
    define the angles to be:

    * :math:`\\phi` - The angle of tilt with respect to the positive :math:`z`-axis. A vector with :math:`\\phi=0`
      will be oriented parallel to the :math:`z`-axis, while a vector with :math:`\\phi=\\pi/2` will be oriented
      parallel to the :math:`(x,y)`-plane. A vector with :math:`\\phi=\\pi` will be oriented parallel to the
      negative :math:`z`-axis.

    * :math:`\\theta` - The orientation in the :math:`(x,y)`-plane with respect to the *positive* :math:`y`-axis. A
      vector with :math:`\\theta=0` will be parallel to the *positive* :math:`y`-axis, while a vector with
      :math:`\\theta=\\pi/2` will be oriented parallel to the *positive* :math:`x`-axis.

    These angles are computed in the following manner:

    .. math::

        \\phi_i = \\textup{arctan} ( \\frac{\\sqrt{{x_i} ^ 2 + {y_i} ^ 2}}{z_i} )

        \\theta_i = \\textup{arctan} ( \\frac{x_i}{y_i} )

    The unit for the angles is **radians** unless ``use_degrees`` is set to ``True``. The angles are modified to be
    in the range of :math:`0` to :math:`\\pi` for :math:`\\phi` and 0 to :math:`\\pi` for
    :math:`\\theta`. The first column in the returned array corresponds to :math:`\\phi` and the second to
    :math:`\\theta`.

    :param vectors: 2D NumPy array containing 3 columns, corresponding to the x, y and z **components** of the
                    vectors, and ``n`` rows, one for each vector. **Note:** We only require the vector *components*,
                    not the *coordinates* in space.
    :param use_degrees: indicate whether the returned angles should be in degrees. If ``False`` (default),
                        the angles will be returned in **radians**.
    :return: 2D NumPy array containing 2 columns, corresponding to :math:`\phi,\theta` for ``n`` rows.
    """


def compute_vector_magnitudes(vectors: np.ndarray) -> np.ndarray:
    """
    Compute vector magnitudes.

    Compute vector magnitudes in 3D, as well as the component of the magnitude in the :math:`(x,y)`-plane. These
    magnitudes are calculated as:

    .. math::

        \| v \| = \\sqrt{{v_x}^2 + {v_y}^2 + {v_z} ^ 2}

        \| v \|_{xy} = \\sqrt{{v_x}^2 + {v_y}^2}

    :param vectors: 2D NumPy array containing 3 columns, corresponding to the x, y and z **components** of the
                    vectors, and ``n`` rows, one for each vector. **Note:** We only require the vector *components*,
                    not the *coordinates* in space.
    :return: 2D NumPy array containing two columns and ``n`` rows. The first column corresponds to the 3D vector
             magnitude while the second column corresponds to the :math:`(x,y)` in-plane magnitude.
    """


def create_binned_orientation(vector_orientations: np.ndarray, vector_magnitudes: Optional[np.ndarray],
                              half_number_of_bins: int = 16, use_degrees: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bin the vector orientation data.

    Construct an array containing the histogram data obtained by binning the orientation data. This function will
    return a >=2-dimensional array. If ``vector_magnitudes`` is ``None``, the result is a simple count-based
    2D histogram of :math:`\\phi` vs. :math:`\\theta`. If ``vector_magnitudes`` is provided and contains **both**
    the in-plane and 3D magnitudes, the output is two 2D histograms (stacked one on top of the other), which contain
    magnitude-weighted frequencies.

    The input angles must be in the range :math:`0 \\leq \\phi < \\pi` and :math:`0 \\leq \\theta < \\pi`
    in radians, or :math:`0 \\leq \\phi < 180` and :math:`0 \\leq \\theta < 180` in degrees. There **cannot** be
    any other orientations included, as these will be overwritten.

    Once the bins are assigned, a mirroring step is performed to fill in the missing angles. The mirroring is as
    follows for the case where the magnitudes are provided:

    * In the 3D magnitude-weighted histogram, no mirroring occurs, as :math:`\\phi` only extends from 0 to 180
      degrees. The values are copied to the second half of the bins to allow simpler visualisation. However,
      these values must **not** be used in a 3D visualisation.
    * In the in-plane magnitude-weighted histogram, the values for the :math:`\\theta` bins going from 180 to 360
      degrees are assigned the same values as the bins from 0 to 180 degrees.

    In the case of count-weighted histograms, a simple mirroring approach is used that copies the counts for the
    first half of the theta bins to the second half. No :math:`\\phi`-mirroring is necessary.

    :param vector_orientations: 2D NumPy array containing ``n`` rows, one for each vector, and 2 columns,
                                corresponding to the angles :math:`\phi,\theta`.
    :param vector_magnitudes: Optional 2D NumPy array containing ``n`` rows, one for each vector, and 2 columns,
                              corresponding to the 3D and in-plane magnitudes, respectively. If ``None``,
                              then the histogram is based simply on counts, not on magnitudes.
    :param half_number_of_bins: The half-number of bins. This represents the number of bins that should be produced
                                in the 180\u00b0 (:math:`\\pi` rad) range for each set of angles. This number
                                **must** be even.
    :param use_degrees: Indicate whether the angles are provided in degrees. If ``True``, angles are interpreted as
                        degrees, otherwise the angles are interpreted as radians.
    :return: Tuple containing 2D histogram of :math:`\phi,\theta` and an array providing bounds of the histogram bins.
             If the magnitudes are provided, this will be a two-sheet histogram, with dimensions
             ``(half_number_of_bins * 2, half_number_of_bins * 2, 2)``. If the histogram is count-based,
             the array is 2D (omitting the final index).
    """


def produce_phi_theta_1d_histogram_data(binned_data: np.ndarray) -> np.ndarray:
    """
    Return the marginal 1D :math:`\\phi,\\theta` histogram arrays.

    This function computes the marginal histogram frequencies for :math:`\\phi, \\theta`. The :math:`\\phi`
    histogram relies on the 3D magnitude while the :math:`\\theta` histogram relies on the in-plane magnitude. If the
    binned data is a 2D array, containing a count-based histogram, both marginals are computed using the same array,
    summing along their respective axes.

    :param binned_data: NumPy array containing the 2D :math:`\phi,\theta` histogram. This may be either
                        magnitude-weighted or count-weighted.
    :return: NumPy array containing the marginal :math:`\phi,\theta` histograms. The zero-axis will have size 2,
             with the first element containing the :math:`\phi` histogram and the second element containing the
             :math:`theta` histogram.
    """


def produce_histogram_plots(binned_data: np.ndarray, bins: np.ndarray, sphere_radius: float = 2.0,
                            zero_position_2d: CardinalDirection = CardinalDirection.NORTH,
                            rotation_direction: RotationDirection = RotationDirection.CLOCKWISE,
                            use_degrees: bool = True, colour_map: str = "gray"):
    """
    Produce a show the anisotropy rose histograms.

    This function produces and shows a 3-panel figure containing (from left to right):

    * The 3D hemisphere plot of :math:`\\phi,\\theta`.
    * The 2D polar histogram of :math:`\\theta`.
    * The 2D polar histogram of :math:`\\phi`.

    A number of plotting parameters may be modified here. See the parameter descriptions for more details.

    :param binned_data: The binned histogram data for the :math:`\phi,\theta` plane.
    :param bins: The boundaries of the bins.
    :param sphere_radius: The radius of the sphere used for 3D plotting.
    :param zero_position_2d: The cardinal direction where zero should be placed in the 2D polar histograms
                             (default: North).
    :param rotation_direction: The direction of increasing angles in the 2D polar histograms (default: clockwise).
    :param use_degrees: Indicate whether the values are in degrees. If ``True``, values are assumed to be in
                        degrees. Otherwise, radians are assumed.
    :param colour_map: Name of the matplotlib colourmap to use to colour the hemisphere. If an invalid name is
                       provided, a default greyscale colourmap ("gray") will be used.
    :return: ``None``, but produces a figure on the screen.
    """


def perform_anisotropy_rose_pipeline(vectors: np.ndarray, half_number_of_bins: int = 18, use_degrees: bool = True,
                                     sphere_radius: float = 2.0, weight_by_magnitude: bool = True,
                                     zero_position_2d: CardinalDirection = CardinalDirection.NORTH,
                                     rotation_direction: RotationDirection = RotationDirection.CLOCKWISE,
                                     colour_map: str = "gray"):
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
    :return: ``None``, but produces a figure on screen.

    **TODO: Add the ability to easily save from here.**
    """
