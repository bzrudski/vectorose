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
vectors, stored in [`quickstart_vectors.npy`](quickstart_vectors.npy), are
from a simulated dataset. We'll analyse these vectors quite closely during
this example.

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

Yes! VectoRose is open-source and licensed under the MIT License. Anyone
can use VectoRose for any purpose and even modify and further develop it.
The only requirement is that you attribute us as the original authors. We
also request that you cite us if you publish any articles that use our
tool.

## Installing VectoRose

Installing VectoRose is very straightforward. The only requirement is that
you have Python installed. At the command line, type the following to
install VectoRose:

```bash
pip install vectorose
```

There are more advanced installation options available, of course. Make
sure to check out our `installation page <installation>` for more details.

## Importing VectoRose

Now, let's get to the actual code! Make sure you have Python open. To
start, let's import the `vectorose` package. To make things easier, we use
the alias `vr` when we import.

```python
import vectorose as vr
```

We're now ready to dive into our example.

## Loading and Preprocessing the Vectors

Let's begin by loading our vectors into Python. To do this, we can use the
function {func}`vectorose.io.import_vector_field`.

```python
vectors = vr.io.import_vector_field("quickstart_vectors.npy", location_columns=None)
```

[//]: # (Note to self: include some zero-vectors in the file!)

Before doing any of our analysis, we must remove the zero-magnitude vectors
using {func}`.util.remove_zero_vectors`.

```python
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
<advanced_histograms>`.

### Data Binning

The first step to construct a histogram is to bin the data. To do this, we
need a representation of a sphere. In VectoRose, this can be obtained using
a {class}`.TriangleSphere` or one of our {class}`.TregenzaSphere`
subclasses. In our example, we'll use a {class}`.FineTregenzaSphere`. This
discretised sphere contains 5806 patches in 54 rings. To study the
magnitude, we'll construct 32 bins, which will be represented as the nested
spherical shells.

We'll then assign the bins using {meth}`.SphereBase.assign_histogram_bins`.

```python
sphere = vr.tregenza_sphere.FineTregenzaSphere(number_of_shells=32)

labelled_vectors, magnitude_bins = my_sphere.assign_histogram_bins(
    vectors
)

# Let's look at the labelled vectors
labelled_vectors
```


### Nested Spherical Histograms

To construct the nested spherical histogram, we can use the method
{meth}`.SphereBase.construct_histogram` for our sphere representation.

```python
histogram = sphere.construct_histogram(labelled_vectors)

histogram.to_frame()
```

We can then visualise the histogram by constructing the shell meshes using
{meth}`.SphereBase.create_histogram_meshes` and the show them in 3D using
{class}`.SpherePlotter` from the {mod}`.plotting` module.

```python
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

```python
sphere_plotter.export_screenshot(
    "./assets/quickstart/nested_spheres.png",
    transparent_background=False,
 )

sphere_plotter.produce_shells_video(
    "./assets/quickstart/nested_spheres.mp4",
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

### Magnitude Histograms

To construct the magnitude histogram with 32 bins, we can use the same
sphere representation. We first get the histogram bin counts using the
{meth}`.SphereBase.construct_marginal_magnitude_histogram` method.

```python
magnitude_histogram = sphere.construct_marginal_magnitude_histogram(
  labelled_vectors
)

magnitude_histogram.to_frame()
```

To visualise the histogram, we can use the function
{func}`.produce_1d_scalar_histogram` in the {mod}`.plotting` module. To
customise the plot, we can use all the usual tricks from {mod}`matplotlib`.

```python
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

### Orientation Histograms

To construct the orientation histogram, we can call the method
{meth}`.SphereBase.construct_marginal_orientation_histogram` on our sphere
representation.

```python
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

```python
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

```python
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

## Statistics

In addition to visualising vectorial data, VectoRose also allows computing
*directional statistics* on the loaded vectors. These statistics are
explained in more detail in our {doc}`Statistics Overview <statistics>`
page.
