"""VectoRose: A new tool for visualising and analysing directional data."""

from vectorose import polar_data
from vectorose import plotting
from vectorose import io
from vectorose import triangle_sphere
from vectorose import tregenza_sphere
from vectorose import util
from vectorose import stats

from vectorose.sphere_base import SphereBase

from importlib_metadata import version

__version__ = version("vectorose")
