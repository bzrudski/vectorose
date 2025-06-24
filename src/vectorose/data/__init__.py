"""Sample data

This module grants access to the sample data used in the documentation.
These datasets may be loaded and accessed as NumPy arrays without requiring
any external download.
"""

import enum
import os

from .. import io
import numpy as np

class SampleData(enum.Enum):
    """Sample dataset.

    This enumeration provides a list of sample datasets provided with
    VectoRose, as well as a simple interface to load them.

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
    """Anisotropy of offset rotated layers of cylinders."""

    def load(self) -> np.ndarray:
        """Load the current dataset to use with VectoRose."""

        parent_dir = os.path.dirname(__file__)

        filename = os.path.join(parent_dir, f"{self.value}.npy")

        vectors = io.import_vector_field(filename)

        return vectors
