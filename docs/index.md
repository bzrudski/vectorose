---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# VectoRose

Spherical and polar histogram plotting for non-unit vectorial and axial
data.

```{code-cell} ipython3
:tags: [remove-cell]

# Configure PyVista behind the scenes before starting
import pandas as pd
import pyvista as pv

pv.set_jupyter_backend("html")
pv.global_theme.font.fmt = "%.6g"

pd.options.display.max_rows = 20
```

![VectoRose: visualise and analyse 3D directed data](../resources/splash/splash.png)

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
  orientations and directions, as described by 
  {cite:t}`fisherStatisticalAnalysisSpherical1993`.

```{figure} ./assets/sample_plots/nested_shells.gif
:alt: Nested shells representing vectors of non-unit length.
:scale: 50%
:align: center

Bivariate histogram showing the frequency of vectors at each combination of
orientation and magnitude. Smaller shells represent lower magnitude vectors
within the dataset.
```

```{figure} ./assets/sample_plots/rotating_orientation.gif
:alt: Rotation animation showing various orientations.
:scale: 50%
:align: center

Marginal spherical histogram showing the direction distribution of vectors.
All magnitudes are considered equally, and as such magnitude differences
are ignored.
```

```{hint}
Curious about how these figures were generated? Check out the
{doc}`Producing Animations <users_guide/animations>` page in the **Users'
Guide**.
```

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

The first step requires a discrete representation of a sphere inspired by a
Tregenza sphere {cite:p}`beckersGeneralRuleDisk2012,
tregenzaSubdivisionSkyHemisphere1987`, which divides the surface of the
sphere into 5806 faces, most of which are rectangular, of approximately
equal surface area. Two keyword arguments can be used to set the number of
magnitude bins (`number_of_shells`) and to fix the histogram domain
(`magnitude_range`).

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

```{code-cell} ipython3
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

When this code is run in a Jupyter notebook, an interactive plotting output
will appear beneath the code cell. When this code is run in a Python
console, a new interactive window will appear that blocks the main thread.
In both cases, a set of sliders appear that allow the opacity of the nested
spheres to be configured. You can change which sphere is visible and
control the opacity of all spheres.

```{hint}
In the static HTML version of this page, the visualisation sliders don't
appear. Try copying the above code and running it in a Python console or a
Jupyter Notebook to see the interactive viewer and try the sliders.
```

In addition to showing the plot in 3D, VectoRose includes various functions
to produce animations and screenshots of spherical histograms.

```{code-cell} ipython3
import os

# Export a gif of the histogram
output_dir = "assets/readme_examples/"

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

output_file = os.path.join(output_dir, "histogram_construction_shells.gif")

my_sphere_plotter.produce_shells_video(
    output_file,
    fps=3,
    inward_direction=True,
    boomerang=True,
    add_shell_text=True
)
```

Now, let's take a look at the produced animation.

```{image} ./assets/readme_examples/histogram_construction_shells.gif
:alt: Shell animation for VectoRose example.
:scale: 50%
:align: center
```

### Directional Statistics

The functions in the {mod}`.vectorose.stats` module enable directional
statistics to be computed. These functions have been adapted from the work
by {cite:t}`fisherStatisticalAnalysisSpherical1993`.

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
shape and strength parameters, as described by
{cite:t}`woodcockSpecificationFabricShapes1977` and as explained by
{cite:t}`fisherStatisticalAnalysisSpherical1993`.

```{code-cell} ipython3
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


Additional statistical operations are provided in the VectoRose API and are
described in the **{doc}`User's Guide <users_guide/users_guide>`**.

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

Works consult in this project are available on the
{doc}`References <users_guide/references>` page. For the directional
statistics approaches, we made extensive use of *Statistic analysis of
spherical data* by {cite:t}`fisherStatisticalAnalysisSpherical1993`.

We also made extensive use of the book [*Python Packages*](https://py-pkgs.org/)
by Tomas Beuzen and Tiffany Timbers to inform the structure and development
of this package.

## Credits

`vectorose` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).

```{toctree}
:maxdepth: 1
:hidden:

users_guide/users_guide.md
autoapi/index
changelog.md
contributing.md
conduct.md
```
