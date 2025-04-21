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

# Quick Start

Welcome to VectoRose! This document is intended to help you get up and
running quickly with our package. If you want a more thorough introduction,
make sure to keep reading the next pages of this manual. This page is meant
to provide an overview of the features present in VectoRose, as well as a
quick example pipeline, without diving so deeply into the theory behind
everything.

In this page, we'll be using a sample dataset of simulated vectors. These
vectors, stored in {download}`quickstart_vectors.npy
<quickstart_vectors.npy>`, are from a simulated dataset. We'll analyse
these vectors quite closely during this example.

## What is VectoRose?

VectoRose is a Python package for analysing vectorial and axial data.
Unlike existing tools, these vectors are not required to have unit length.
VectoRose contains tools for two main purposes: **plotting** and
**statistical analysis**. We'll scratch the surface of both of these in
this guide.

Three different types of histograms can be constructed using VectoRose:
* 1D scalar histograms to visualise vector magnitudes
* Spherical histograms to visualise vector directions
* Nested spherical histograms to visualise all combinations of magnitude
  and orientation.

In this example, we'll see how to generate and plot each of these.

## Can I use VectoRose in my research?

Yes! VectoRose is open-source and licensed under the **MIT License**.
Anyone can use VectoRose for any purpose and even modify and further
develop it. The only requirement is that you attribute us as the original
authors. We also request that you cite us if you publish any articles that
use our tool.

```{note}
Interested in modifying and extending VectoRose? Check out our
{doc}`Contributing <../contributing>` guide to get started.
```

## Installing VectoRose

Installing VectoRose is very straightforward. The only requirement is that
you have Python installed. At the command line, type the following to
install VectoRose:

```shell
pip install vectorose
```

There are more advanced installation options available, of course. Make
sure to check out our {doc}`installation page <installation>` for more
details.

## Importing VectoRose

Now, let's get to the actual code! Make sure you have Python open. To
start, let's import the `vectorose` package. To make things easier, we use
the alias `vr` when we import.

```{code-cell} ipython3
import vectorose as vr
```

We can now use all the VectoRose functionality.

### Extra Configuration

To improve how this example appears in our rendered HTML web page, we can
configure a couple of extra options.

First, to ensure that our interactive 3D plots appear, we need to call
{func}`pyvista.set_jupyter_backend` to change the PyVista rendering backend
to `"html"`.

```{code-cell} ipython3
import pyvista as pv
import platform

if platform.uname() != "Windows":
  pv.start_xvfb()

pv.set_jupyter_backend("html")
```

```{attention}
The line `pv.start_xvfb()`{l=python} is only required if running on a
Unix-like or Unix-based operating system (Linux or macOS). For techincal
reasons, it is not required when running on Windows (and will, in fact,
produce an error).
```

We can also change how our data tables produced using Pandas appear using
[`pandas.options`](https://pandas.pydata.org/docs/user_guide/options.html).
We'll just control the number of rows that appear, but there are many other
options that can be changed.

```{code-cell} ipython3
import pandas as pd

pd.options.display.max_rows = 20
```

We're now ready to dive into our example.

## Loading and Preprocessing the Vectors

Let's begin by loading our vectors into Python. To do this, we can use the
function {func}`vectorose.io.import_vector_field`.

```{code-cell} ipython3
vectors = vr.io.import_vector_field("quickstart_vectors.npy", location_columns=None)
```

Before doing any of our analysis, we must remove the zero-magnitude vectors
using {func}`.util.remove_zero_vectors`.

```{code-cell} ipython3
vectors = vr.util.remove_zero_vectors(vectors)

# Now, let's look at the vectors
vectors
```

```{tip}
There are additional pre-processing steps available. Make sure to check out
the functions in {mod}`vectorose.util` to see other options. The loading
process is covered in more detail in {doc}`Loading Vectors into VectoRose
<data_format>`.
```

## Constructing and Visualising Histograms

In this section, we'll give an overview on constructing histograms using
VectoRose. For more in-depth discussion, see the pages on {doc}`Histogram
Plotting <plot_types>` and {doc}`Advanced Histogram Plotting
<advanced_histograms>`. There are three major types of histograms that can
be constructed using VectoRose:

* **Nested spherical histograms** - provide insight into magnitude and
  direction together.
* **Magnitude histograms** - provide insight into magnitude alone.
* **Direction histograms** - provide insight into direction alone.

We'll see how to construct each of these histograms for our sample data.

### Data Binning

The first step to construct a histogram is to bin the data. To do this, we
need a representation of a sphere. In VectoRose, this can be obtained using
a {class}`.TriangleSphere` or one of our {class}`.TregenzaSphere`
subclasses. In our example, we'll use a {class}`.FineTregenzaSphere`. This
discretised sphere contains 5806 patches in 54 rings. To study the
magnitude, we'll construct 32 bins, which will be represented as the nested
spherical shells.

We'll then assign the bins using {meth}`.SphereBase.assign_histogram_bins`.

```{code-cell} ipython3
sphere = vr.tregenza_sphere.FineTregenzaSphere(number_of_shells=32)

labelled_vectors, magnitude_bins = sphere.assign_histogram_bins(
    vectors
)

# Let's look at the labelled vectors
labelled_vectors
```

### Nested Spherical Histograms

To construct the nested spherical histogram, we can use the method
{meth}`.SphereBase.construct_histogram` for our sphere representation.

```{code-cell} ipython3
histogram = sphere.construct_histogram(labelled_vectors)

histogram.to_frame()
```

We can then visualise the histogram by constructing the shell meshes using
{meth}`.SphereBase.create_histogram_meshes` and the show them in 3D using
{class}`.SpherePlotter` from the {mod}`.plotting` module.

```{code-cell} ipython3
sphere_meshes = sphere.create_histogram_meshes(histogram, magnitude_bins)

sphere_plotter = vr.plotting.SpherePlotter(sphere_meshes)
sphere_plotter.produce_plot()
sphere_plotter.show()
```

```{danger}
Remember to call the {meth}`.SpherePlotter.produce_plot` method to add the
spheres to the plot before showing it.
```

Now we can see the constructed meshes, and we also have some sliders that
let us change which shell is visible. This may not appear in the static
HTML version of the documentation, so we're exporting and embedding a video
and a screenshot here showing the different shells.

```{code-cell} ipython3
# Make sure our output path exists
import os
output_path = "./assets/quickstart/"

if not os.path.exists(output_path):
  os.mkdir(output_path)

sphere_plotter.export_screenshot(
    os.path.join(output_path, "nested_spheres.png"),
    transparent_background=False,
 )

sphere_plotter.produce_shells_video(
    os.path.join(output_path, "nested_spheres.mp4"),
    quality=5,
    fps=4,
    boomerang=True,
    add_shell_text=True
)
```

```{video} ./assets/quickstart/nested_spheres.mp4
:width: 100%
:autoplay:
:loop:
:poster: ./assets/quickstart/nested_spheres.png
:alt: Example video of the nested histogram shells.
```


```{seealso}
For more information on exporting images and videos from plots, check out
the following {class}`.SpherePlotter` methods.

{meth}`.SpherePlotter.export_screenshot`
: Export raster screenshots of the 3D plot.

{meth}`.SpherePlotter.export_graphic`
: Export vector screenshots of the 3D plot (raster image + text).

{meth}`.SpherePlotter.produce_shells_video`
: Export video going through the nested shells of a histogram plot.

{meth}`.SpherePlotter.produce_rotating_video`
: Export video of the spherical histogram rotating.
```

Looking at our series of nested spheres, we see that there are two major
patterns in the data. At the higher magnitude levels, represented by the
outer shells, we have a ring of vectors, known as a **girdle**. At the
lower magnitude levels, we have a single bright spot of vectors, known as a
**cluster**. These patterns are special, and quite common in **directional
statistics** {cite:p}`{see}fisherStatisticalAnalysisSpherical1993,
mardiaDirectionalStatistics2000{for more information}`.

### Magnitude Histograms

To construct the magnitude histogram with 32 bins, we can use the same
sphere representation. We first get the histogram bin counts using the
{meth}`.SphereBase.construct_marginal_magnitude_histogram` method.

```{code-cell} ipython3
magnitude_histogram = sphere.construct_marginal_magnitude_histogram(
  labelled_vectors
)

magnitude_histogram.to_frame()
```

To visualise the histogram, we can use the function
{func}`.produce_1d_scalar_histogram` in the {mod}`.plotting` module. To
customise the plot, we can use all the usual tricks from {mod}`matplotlib`.

```{code-cell} ipython3
---
mystnb:
    figure:
        align: center
---

ax = vr.plotting.produce_1d_scalar_histogram(magnitude_histogram, magnitude_bins)
ax.set_title("Magnitude Histogram")
ax.set_xlabel("Magnitude")
ax.set_ylabel("Count")
```

```{attention}
Don't forget to pass the magnitude bin edges to the function
{func}`.produce_1d_scalar_histogram`!
```

Based on this plot, we have a bimodal distribution of magnitudes. Each mode
corresponds to one of the two features (girdle and cluster) we described
earlier.

### Direction Histograms

To construct the direction histogram, we can call the method
{meth}`.SphereBase.construct_marginal_orientation_histogram` on our sphere
representation.

```{code-cell} ipython3
orientation_histogram = sphere.construct_marginal_orientation_histogram(
  labelled_vectors
)

orientation_histogram.to_frame()
```

To visualise the histogram, we can use an approach similar to the nested
spheres approach. The key difference is that we construct a **single** mesh
using the {meth}`SphereBase.create_shell_mesh` method. We can then
visualise this mesh using a {class}`.SpherePlotter` like above. We've also
added spherical axes using {meth}`.SpherePlotter.add_spherical_axes` to
show the $\phi$ and $\theta$ axes in 3D.

```{code-cell} ipython3
orientation_mesh = sphere.create_shell_mesh(orientation_histogram)
sphere_plotter = vr.plotting.SpherePlotter(orientation_mesh)
sphere_plotter.produce_plot()
sphere_plotter.add_spherical_axes()
sphere_plotter.show()
```

To share these results, we can export a video of the spherical histogram
rotating about its axis using
{meth}`.SpherePlotter.produce_rotating_video`. We can also export a single
still frame using {meth}`.SpherePlotter.export_screenshot`, as before.

```{code-cell} ipython3
sphere_plotter.export_screenshot(
    "./assets/quickstart/orientation_histogram.png",
    transparent_background=False,
 )

sphere_plotter.produce_rotating_video(
    "./assets/quickstart/orientation_histogram.mp4",
    quality=5,
    fps=12,
    number_of_frames=36,
    hide_sliders=True
)
```

```{video} ./assets/quickstart/orientation_histogram.mp4
:width: 100%
:autoplay:
:loop:
:poster: ./assets/quickstart/orientation_histogram.png
:alt: Example video of the rotating orientation histogram.
```

Using this spherical histogram, we can see that the two patterns, the
girdle and the cluster, overlap. However, the nested spherical histograms
above illustrate that these are indeed distinct patterns, separated by
differences in magnitude.

```{tip}
Directions and orientations can also be visualised using polar histograms,
as described in the page on {doc}`Histogram Plotting <plot_types>`.
```

## Statistics

In addition to visualising vectorial data, VectoRose also allows computing
*directional statistics* on the loaded vectors. These statistics are
explained in detail in our {doc}`Statistics Overview <statistics>` page.
The functions necessary for performing directional statistics are defined
in the module {mod}`.stats`.

Directional statistics can be computed on any collection of **unit
vectors**, that is, vectors with a magnitude equal to 1. This set of
vectors may be obtained from all vectors, representing the *marginal
orientation distribution*, or from a subset of vectors depending on their
magnitude, representing a *conditional orientation distribution*. We will
do both in this example.

We've mentioned that our data show two patterns: a **girdle** in the higher
magnitude shells and a **cluster** in the lower magnitude shells. These two
patterns overlap in the marginal orientation histogram.

To quantitatively describe these patterns, we can use Woodcock's shape and
strength parameters {cite:p}`woodcockSpecificationFabricShapes1977`. These
parameters are described in more detail in our
{doc}`Statistics Overview <statistics>` page. The main idea is that we
compute **two** parameters:

* **Shape parameter:** a value between 0 and 1 reflects a *girdle*, a value
  greater than 1 represents a *cluster* and a value equal to 1 represents
  something in between.
* **Strength parameter:** a large value represents a compact distribution,
  while a small value represents a diffuse distribution.

These parameters are computed using the eigenvalues of the orientation
matrix defined by these vectors (see {doc}`Statistics Overview
<statistics>`).

Since we suspect we have a girdle and a cluster here, let's compute these
parameters for our distribution. We first need to compute the eigenvalues
of the orientation matrix for our set of vectors using the function
{func}`~.stats.compute_orientation_matrix_eigs` and then we can compute the
shape and strength parameters using the function
{func}`~.stats.compute_orientation_matrix_parameters`.

Let's start with the marginal orientation distribution. This considers all
the vectors. We **must** convert them first to unit vectors. We can do this
either using {func}`.util.normalise_vectors` or
{meth}`.SphereBase.convert_vectors_to_cartesian_array`. In this example,
we'll use the latter.

```{warning}
If using {meth}`.SphereBase.convert_vectors_to_cartesian_array`, remember
to set `create_unit_vectors=True` when calling the function.
```

```{code-cell} ipython3
unit_vectors_all = sphere.convert_vectors_to_cartesian_array(
  labelled_vectors, create_unit_vectors=True
)

orientation_matrix_eig_result = vr.stats.compute_orientation_matrix_eigs(
  unit_vectors_all
)

eigenvalues = orientation_matrix_eig_result.eigenvalues

woodcock_parameters = vr.stats.compute_orientation_matrix_parameters(
  eigenvalues
)

print(f"The marginal distribution has a shape parameter of "
      f"{woodcock_parameters.shape_parameter} and a strength parameter of "
      f"{woodcock_parameters.strength_parameter}.")
```

We can now do the same for our conditional shells. We can isolate
individual shells, or iterate over every shell. Let's do the same operation
for each shell and print out the values on the screen.

We can extract the vectors using our `labelled_vectors` from before with a
bit of data indexing thanks to Pandas.

```{code-cell} ipython3
number_of_shells = sphere.number_of_shells

for i in range(number_of_shells):
  shell_vectors = labelled_vectors[labelled_vectors["shell"] == i]
  shell_unit_vectors = sphere.convert_vectors_to_cartesian_array(
    shell_vectors, create_unit_vectors=True
  )
  orientation_matrix_eig_result = vr.stats.compute_orientation_matrix_eigs(
    shell_unit_vectors
  )
  
  eigenvalues = orientation_matrix_eig_result.eigenvalues
  
  woodcock_parameters = vr.stats.compute_orientation_matrix_parameters(
    eigenvalues
  )
  
  print(f"Shell {i}: shape parameter {woodcock_parameters.shape_parameter};"
        f" strength parameter {woodcock_parameters.strength_parameter}.")
```

We can see that as we change which window of magnitude is being considered,
the shape and strength parameters change. In the lower magnitude shells, we
compute very high values for the shape parameter, reflecting the *cluster*.
In the higher magnitude shells, we have very low shape parameter values,
reflecting the *girdle* distribution. In all cases, the strength parameter
is quite high, representing a compact distribution.

```{tip}
Shape and strength parameter are by far not the only values that we can
compute to understand the shape and distribution of orientations. Make sure
to check out our {doc}`Statistics Overview <statistics>` for a more
thorough description of the analyses that can be performed and the insights
that can be obtained from collections of vectors.
```

## Next Steps

This is the end of our **Quick Start** guide. This guide is not meant to be
a comprehensive overview; it's just a first look to help you get up and
running. If you want to explore VectoRose in more depth, we recommend
continuing to read this **Users' Guide**.

In addition to our resources, it can also help to familiarise yourself with
[**NumPy**](https://numpy.org/doc/stable/) and
[**pandas**](https://pandas.pydata.org/docs/). VectoRose uses these
packages *extensively* and knowing more about them will enable you to do
more advanced operations, including:
* Exporting and importing histograms from files.
* Filtering vectors based on magnitude, orientation and angular distance.
* Developing other statistical functionality.

We hope that this first guide has been helpful. If you encounter any
difficulties or bugs using VectoRose, make sure to open a GitHub issue at
[https://github.com/bzrudski/vectorose/issues](
https://github.com/bzrudski/vectorose/issues). Welcome to VectoRose!
