"""Sample datasets for running VectoRose.

This module grants access to the sample data used in the documentation.
These datasets may be loaded and accessed as NumPy arrays without requiring
any external download.

Warnings
--------
This module does not exist as an automatic top-level import in vectorose,
and must therefore be explicitly imported.
"""

import enum
import os

from .. import io
import numpy as np

class SampleData(enum.Enum):
    """Interface for accessing sample datasets.

    This enumeration provides a list of sample datasets provided with
    VectoRose, as well as a simple interface to load them. The string
    values associated with each enumerated instance correspond to the base
    filename for the respective dataset.

    Warnings
    --------
    All datasets are assumed to be stored as a NumPy array (``*.npy``).

    Notes
    -----
    This system is used to ensure the flexibility to easily add new sample
    data in the future.
    """

    CLUSTER_GIRDLE = "cluster_girdle"
    """Overlapping cluster and girdle with different magnitudes."""

    TWO_CLUSTERS = "two_clusters"
    """Two clusters with different magnitudes and orientations."""

    TWISTED_BLOCKS = "twisted_blocks"
    """Anisotropy of offset rotated layers of cylinders.
    
    Notes
    -----
    Computed using Dragonfly 3D World.
    """

    def load(self) -> np.ndarray:
        """Load the current dataset to use with VectoRose.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n, d)`` containing the loaded vectors. If the
            specific dataset has location coordinates, then ``d == 6``.
            Otherwise, ``d == 3``. ``n`` represents the number of vectors
            in the dataset.
        """

        parent_dir = os.path.dirname(__file__)

        filename = os.path.join(parent_dir, f"{self.value}.npy")

        vectors = io.import_vector_field(filename)

        return vectors
