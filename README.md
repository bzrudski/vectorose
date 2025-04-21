# VectoRose

![VectoRose: visualise and analyse 3D directed data](https://github.com/bzrudski/vectorose/blob/main/resources/splash/splash.png?raw=true)

Spherical and polar histogram plotting for non-unit vectorial and axial
data.

[![PyPI - Version](https://img.shields.io/pypi/v/vectorose)](https://pypi.org/project/vectorose/)
[![codecov](https://codecov.io/github/bzrudski/vectorose/graph/badge.svg)](https://codecov.io/github/bzrudski/vectorose)
[![docs](https://app.readthedocs.org/projects/vectorose/badge/)](https://vectorose.readthedocs.io/en/latest/)
[![CI/CD](https://github.com/bzrudski/vectorose/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/bzrudski/vectorose/actions/workflows/ci-cd.yml)
[![CI - Windows](https://github.com/bzrudski/vectorose/actions/workflows/ci-win.yml/badge.svg)](https://github.com/bzrudski/vectorose/actions/workflows/ci-win.yml)

## Overview

Many fields of science rely on oriented data. In these contexts, scalar
values alone can't describe the quantities under consideration. The values
of interest are **vectors**, consisting of a *direction* or *orientation*,
in addition to an optional magnitude (length). Examples include wind
velocities, trabecular bone co-alignment (anisotropy) and cardiac fibre
orientations.

Traditional histograms and statistical tools can't be directly applied to
analyse these data. To be able to visualise and quantitatively describe and
analyse oriented datasets in 3D, we present **VectoRose**.

### Features

**VectoRose** provides tools for *visualising* and quantitatively
*analysing* data sets consisting of vectors and orientations of unit and
non-unit length.

Using VectoRose, it is possible to:

* Construct spherical histograms of directions and orientations in 3D.
* Construct 1D scalar histograms of vector magnitudes.
* Construct nested spherical histograms to understand collections of 
  non-unit vectors and axes.
* Construct 1D polar histograms of vector orientation spherical coordinate
  angles.
* Compute directional statistics to understand the distributions of
  orientations and directions, as described by Fisher, Lewis and
  Embleton.[^fle]

![Nested shells representing vectors of non-unit length.](https://github.com/bzrudski/vectorose/blob/main/resources/sample_plots/nested_shells.gif?raw=true)

![Rotation animation showing various orientations.](https://github.com/bzrudski/vectorose/blob/main/resources/sample_plots/rotating_orientation.gif?raw=true)

## Installation

VectoRose can be installed from PyPI using `pip`.

```bash
$ pip install vectorose
```

Alternatively, you can install it from source by cloning this repository.

## Usage

To use VectoRose, you must have a collection of **3D vectors** stored in a
NumPy array. These may be read from a NumPy file (`*.npy`) or a
comma-separated values (`*.csv`) file using the functions provided in
VectoRose.

VectoRose must be imported in order to be used. We recommend using the
alias `vr` when importing VectoRose:

```python
import vectorose as vr
```

### Histogram Construction

Histogram construction requires two steps:

1. Assigning all vectors to magnitude and orientation bins.
2. Computing histograms and generating the histogram plots.

The first step requires a discrete representation of a sphere, such as a
fine Tregenza sphere, which divides the surface of the sphere into 5806
faces, most of which are rectangular, of approximately equal surface area.
Two keyword arguments can be used to set the number of magnitude bins
(`number_of_shells`) and to fix the histogram domain (`magnitude_range`).

In the second step, a variety of histograms can be constructed. These
histograms may consider the counts (or frequencies) of vectors at each
combination of magnitude and direction (*bivariate histogram*), or within
the bins of each variable separately (*marginal histograms*). Histograms
can also be constructed that consider relative frequencies of one variable
within a specific range of the other (*conditional histograms*).

In this brief code snippet, we will generate some random vectors from a
von Mises-Fisher unimodal directional distribution, with some noise in the
magnitude. We'll then construct the bivariate histogram and visualise it in
3D using PyVista.

```python
import vectorose as vr
import vectorose.mock_data

# Create random vectors for demonstration
my_vectors = vr.mock_data.create_vonmises_fisher_vectors_single_direction(
    phi=45,
    theta=70,
    kappa=20,
    number_of_points=10000,
    magnitude=1.0,
    magnitude_std=0.25,
    use_degrees=True,
    seed=20250317,
)

# Construct the discrete sphere representation
my_sphere = vr.tregenza_sphere.FineTregenzaSphere(number_of_shells=10)
my_binned_vectors, magnitude_bin_edges = my_sphere.assign_histogram_bins(my_vectors)

# Compute the bivariate histogram
my_histogram = my_sphere.construct_histogram(my_binned_vectors, return_fraction=False)

# Generate the histogram meshes
my_histogram_meshes = my_sphere.create_histogram_meshes(my_histogram, magnitude_bin_edges)

# Create a 3D SpherePlotter to view the histogram in 3D and show it
my_sphere_plotter = vr.plotting.SpherePlotter(my_histogram_meshes)
my_sphere_plotter.produce_plot()
my_sphere_plotter.show()
```

![Shell animation for VectoRose example](https://github.com/bzrudski/vectorose/blob/main/resources/readme_examples/histogram_construction_shells.gif?raw=true)

When this code is run in a Jupyter notebook, an interactive plotting output
will appear beneath the code cell. When this code is run in a Python
console, a new interactive window will appear that blocks the main thread.

In addition to showing the plot in 3D, VectoRose includes various functions
to produce animations and screenshots of spherical histograms.

### Directional Statistics

The functions in the `vectorose.stats` module enable directional statistics
to be computed. These functions have been adapted from the work by Fisher,
Lewis and Embleton.[^fle]

VectoRose implements a variety of descriptive statistics and hypothesis
tests. Most of these consider pure directions or orientations, which are
represented as unit vectors. These statistics include:

* Correlation between magnitude and orientation
* Hypothesis testing of uniform vs. unimodal distribution
* Woodcock's shape and strength parameters
* Mean resultant vector
* Spherical median vector
* Von Mises-Fisher parameter estimation
  * Mean direction, including confidence cone
  * Concentration parameter

In this code snippet, we generate two sets of mock vectors: a cluster,
following a von Mises-Fisher distribution, and a girdle, following a Watson
distribution with a negative parameter value. We then compute Woodcock's
shape and strength parameters, as described by Woodcock[^woodcock] and as
explained by Fisher, Lewis and Embleton.[^fle]

```python
import vectorose as vr
import vectorose.mock_data
import numpy as np

# Create random vectors for demonstration
my_cluster_vectors = vr.mock_data.create_vonmises_fisher_vectors_single_direction(
    phi=45,
    theta=70,
    kappa=20,
    number_of_points=10000,
    magnitude=1.0,
    magnitude_std=0,
    use_degrees=True,
    seed=20250318,
)

direction = np.array([1, 0, 0])
my_girdle_vectors = vr.mock_data.generate_watson_distribution(
    direction, -20, n=10000, seed=20250318
)

# Compute Woodcock's parameters for both sets of vectors
cluster_orientation_matrix_eigs, _ = vr.stats.compute_orientation_matrix_eigs(
    my_cluster_vectors
)
girdle_orientation_matrix_eigs, _ = vr.stats.compute_orientation_matrix_eigs(
    my_girdle_vectors
)

cluster_woodcock_parameters = vr.stats.compute_orientation_matrix_parameters(
    cluster_orientation_matrix_eigs
)
girdle_woodcock_parameters = vr.stats.compute_orientation_matrix_parameters(
    girdle_orientation_matrix_eigs
)

print(f"The VMF distribution has shape parameter {cluster_woodcock_parameters.shape_parameter:.3f}"
      f" and strength parameter {cluster_woodcock_parameters.strength_parameter:.3f}.")

print(f"The Watson distribution has shape parameter {girdle_woodcock_parameters.shape_parameter:.3f}"
      f" and strength parameter {girdle_woodcock_parameters.strength_parameter:.3f}.")
```

Running this code produces the following output:
```
The VMF distribution has shape parameter 48.085 and strength parameter 2.987.
The Watson distribution has shape parameter 0.005 and strength parameter 2.955.
```

Additional statistical operations are provided in the VectoRose API and are
described in the **User's Guide**.

## Citation

If you've found VectoRose helpful for your research, please cite our
publication:

```
TBA
```

If you've modelled your analysis based on our sample case studies, please
also cite the following:

```
TBA
```

## Contributing

Interested in contributing? Check out the contributing guidelines. Please
note that this project is released with a Code of Conduct. By contributing
to this project, you agree to abide by its terms.

VectoRose is built on a number of existing, well-supported open-source
packages, including: [NumPy](https://numpy.org/),
[PyVista](https://docs.pyvista.org/),
[Matplotlib](https://matplotlib.org/),
[pandas](https://pandas.pydata.org/),
[SciPy](https://scipy.org/) and [trimesh](https://trimesh.org/).


## License

VectoRose was created by Benjamin Z. Rudski and Joseph Deering. It is
licensed under the terms of the MIT license. See the `LICENSE` file for
more details.

## Acknowledgements

The VectoRose project is developed by Benjamin Z. Rudski and Joseph Deering
under the supervision of Dr. Natalie Reznikov at McGill University, in
Montreal, Quebec, Canada &#x1f1e8;&#x1f1e6;.

Works consult in this project are available in our [online documentation](
https://vectorose.readthedocs.io/en/latest/users_guide/references.html), as
well as in [`docs/refs.bib`](https://github.com/bzrudski/vectorose/blob/main/docs/refs.bib).
For the directional statistics approaches, we made extensive use of
*Statistic analysis of spherical data* by Fisher, Lewis and Embleton.[^fle]

We also made extensive use of the book [*Python Packages*](https://py-pkgs.org/)
by Tomas Beuzen and Tiffany Timbers to inform the structure and development
of this package.

## Credits

`vectorose` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).

[^fle]: Fisher, N. I., Lewis, T., & Embleton, B. J. J. (1993).
  *Statistical analysis of spherical data* ([New ed.], 1. paperback ed).
  Cambridge Univ. Press. <https://www.cambridge.org/ca/universitypress/subjects/physics/astronomy-general/statistical-analysis-spherical-data?format=PB>

[^woodcock]: Woodcock, N. H. (1977). Specification of fabric shapes using
  an eigenvalue method. *Geological Society of America Bulletin, 88*(9), 1231.
  [https://doi.org/10.1130/0016-7606(1977)88<1231:SOFSUA>2.0.CO;2](https://doi.org/10.1130/0016-7606(1977)88<1231:SOFSUA>2.0.CO;2)
