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

# Advanced Histogram Plotting

```{code-cell} ipython3
:tags: [remove-cell]

# Configure PyVista behind the scenes before starting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv

pv.start_xvfb()
pv.set_jupyter_backend("html")
pv.global_theme.font.fmt = "%.6g"

pd.options.display.max_rows = 20
```

In the {doc}`previous section <plot_types>`, we saw how to contruct
histogram plots of **magnitude**, **direction/orientation** and **both**.
In this page, we'll see an easier way to produce these plots, as well as
other types of advanced plots that we can construct to further analyse our
data.

We'll use the same dataset as in the previous example, which we'll load
from the NumPy file [random_vectors.npy](./random_vectors.npy) right now.

```{attention}
You probably know this by now, but don't forget to import VectoRose into
your Python shell by writing `import vectorose as vr`{l=python}.
```

```{code-cell} ipython3
import vectorose as vr

my_vectors = vr.io.import_vector_field("random_vectors.npy", location_columns=None)
my_vectors = vr.util.remove_zero_vectors(my_vectors)

my_vectors
```

We now have our vectors loaded and we can begin constructing our
histograms.

## Some Statistics Terminology

VectoRose is designed to analyse collections of vectors having a magnitude
that is not necessarily 1. We analyse collections of *non-unit* vectors or
axes, as we explained in the {doc}`Introduction to Vectors <vectors_intro>`
section. When converting to spherical coordinates, we break these vectors
down into two different sets of numbers: the **magnitudes** and the
**directions or orientations**. These two quantities are separate but
linked through some external process. By combining these two variables, our
non-unit vectors are **bivariate**. When we consider the **nested spheres**
plot, we are looking at a **bivariate histogram**, showing both the
magnitude of the vectors (which shell the vector falls into) and the
direction of the vectors (which patch the shall the vector falls into).
This histogram may also be referred to as the **joint histogram**.

As we saw before, we can construct histograms of the magnitude and the
orientation *separately*. These are known as the **marginal histograms**.
In these cases, we consider **only** the magnitudes *or* the directions of
the vectors, and we completely ignore the other variable.

But what if we don't want to completely ignore the other variable? What if
we only want to know the directions in which the largest-magnitude vectors
are pointing? Or what if we want to know what magnitudes are present in a
given direction? The answer is the **conditional histogram**. In a
conditional histogram, we fix the value of one of the variables and study
how the other one changes *within the selected data*. For example, if we
want to study the directions of the largest-magnitude vectors, we can
simply select all the vectors with a magnitude above certain threshold and
then study their directions, ignoring the low-magnitude vectors. Similarly,
to study the magnitudes of the vectors in a certain direction, we may
select only the vectors in that direction, and then study their magnitudes.

VectoRose allows easy construction of **bivariate/joint**, **marginal** and
**conditional** histograms based on vectorial and axial data. In this
section, we'll see how to build each of these histograms, while the
{doc}`next section <statistics>` will introduce quantitative *statistics*
that can be computed for direction and orientation on these distributions.

## Some Setup

For all our histograms, we need to have an instance of a subtype of
{class}`.SphereBase`. The exact same steps work regardless of which sphere
discretisation we use. We can use either {class}`.TriangleSphere` or one of
our {class}`.TregenzaSphere` types. In this case, we'll use a
{class}`.FineTregenzaSphere`. To appreciate the magnitude distribution,
we'll consider 32 shells.

```{code-cell} ipython3
my_sphere = vr.tregenza_sphere.FineTregenzaSphere(number_of_shells=32)
```

This sphere object will be important to all our downstream tasks. Let's
also assign all our vectors to the correct histogram bins using
{meth}`.SphereBase.assign_histogram_bins`.

```{code-cell} ipython3
labelled_vectors, magnitude_bins = my_sphere.assign_histogram_bins(my_vectors)

labelled_vectors
```

With our vectors assigned to their proper magnitude and orientation bins,
we can start constructing histograms.

## Joint Histograms

Let's begin with the joint histogram. We've actually already seen this
example in the previous section. To show the frequency or count of vectors
at *all possible combinations* of magnitude and direction, we construct a
series of nested spheres, each representing a range of magnitude values.
These magnitude bin values are captured in the `magnitude_bins` variable we
defined above.

We can directly construct the bivariate histogram using the method
{meth}`.SphereBase.construct_histogram`.

```{code-cell} ipython3
my_bivariate_histogram = my_sphere.construct_histogram(labelled_vectors)

my_bivariate_histogram.to_frame()
```

We can now plot this histogram by constructing the shell meshes using
{meth}`.SphereBase.create_histogram_meshes` and then visualise them using
{class}`.SpherePlotter` in {mod}`.plotting`.

```{code-cell} ipython3
my_bivariate_meshes = my_sphere.create_histogram_meshes(
    my_bivariate_histogram, magnitude_bins=magnitude_bins
)

my_bivariate_sphere_plotter = vr.plotting.SpherePlotter(my_bivariate_meshes)
my_bivariate_sphere_plotter.produce_plot()
my_bivariate_sphere_plotter.show()
```

```{attention}
Again, this may not work so well if you're looking at the static HTML
rendered page in a web browser. Try running this page as a Jupyter notebook
to see the results.
```

To visualise this bivariate histogram more clearly (for the web version),
we can also export a video going through each shell using
{meth}`.SpherePlotter.produce_shells_video`.

```{code-cell} ipython3
:tags: [remove-cell]
my_bivariate_sphere_plotter.export_screenshot(
  "./assets/advanced_shells/advanced_shells.png",
  False
)
```

```{code-cell} ipython3
my_bivariate_sphere_plotter.produce_shells_video(
  "./assets/advanced_shells/advanced_shells.mp4",
  quality=5,
  fps=4,
  boomerang=True,
  add_shell_text=True,
  hide_sliders=True
)
```

```{video} ./assets/advanced_shells/advanced_shells.mp4
:width: 100%
:autoplay:
:loop:
:poster: ./assets/advanced_shells/advanced_shells.png
:alt: Example video of the histogram shells.
```

From this plot, we can see which orientations are present at specific
magnitude levels. But, it's a bit difficult to get a full appreciation of
the data just from this one plot. For starters, we can only really see one
shell at a time. This limitation **can** be resolved using some of the
methods of the {class}`.SpherePlotter` class. A single shell of interest
can be activated using {meth}`.SpherePlotter.activate_shell` and the
opacity of that shell can be modified using
{meth}`.SpherePlotter.set_active_shell_opacity` while the opacity of all
other shells can be modified using
{meth}`.SpherePlotter.set_inactive_shell_opacity`.

Still, it may be helpful to study the **marginal** histograms to get a
big-picture overview of the data.

One the more local side of things, we also run into some issues.
Given that there are so many vectors in the dataset, shells where few
vectors land may be hard to analyse. **Conditional** histograms will help
overcome this issue by ignoring all other vectors an only focussing on
those within the shell of interest.

Now, let's see how to easily construct each of these types of histograms
using VectoRose.

## Marginal Histograms

As our vectors have both **magnitude** and **direction**, we can construct
marginal histograms for both. Let's start first with the magnitude.

### Marginal Magnitude Histograms

Recall from the {doc}`previous section <plot_types>` that the vector
magnitudes can be plotted easily on a 1D histogram. In the previous
example, we had to manually calculate the vector magnitudes and build the
histogram. Now, we'll get VectoRose to take care of everything. All we need
is a sphere representation (some instance of {class}`.SphereBase`) and our
labelled vectors. The key method that we'll use is
{meth}`.SphereBase.construct_marginal_magnitude_histogram`. We simply need
to provide our labelled vectors to this method. We may also indicate if we
would like count or frequency data, using the `return_fraction` keyword
argument.

Let's generate the magnitude histogram using count data.

```{code-cell} ipython3
my_marginal_magnitude_histogram = my_sphere.construct_marginal_magnitude_histogram(
    labelled_vectors, return_fraction=False
)

my_marginal_magnitude_histogram.to_frame()
```

To plot these data as a histogram, we can call the function
{func}`.produce_1d_scalar_histogram` in the {mod}`.plotting` module.

```{code-cell} ipython3
---
mystnb:
    figure:
        align: center
---
ax = vr.plotting.produce_1d_scalar_histogram(
    my_marginal_magnitude_histogram, magnitude_bins
)
ax.set_title("Magnitude Histogram")
ax.set_xlabel("Magnitude")
ax.set_ylabel("Count")
```

```{note}
As shown in the example above, the plot does *not* include axis labels or a
title. Behind the scenes, we use [Matplotlib](https://matplotlib.org/) to
produce this plot. Make sure to check our their documentation to
customise your plots.
```

This histogram shows the distribution of magnitude values in the data
without considering the orientation.

### Marginal Direction and Orientation Histograms

In our example in the
{doc}`previous section <plot_types>`, we generated direction histograms
using a {class}`.SphereBase` with only a single shell. But what if we don't
want to perform the bin assignment again? We can construct the marginal
direction or orientation plot using the method
{meth}`.SphereBase.construct_marginal_orientation_histogram`.

```{code-cell} ipython3
my_marginal_direction_histogram = my_sphere.construct_marginal_orientation_histogram(
    labelled_vectors, return_fraction=False
)

my_marginal_direction_histogram.to_frame()
```

Similar to the bivariate histogram, we need to get a sphere mesh. To get a
single sphere with the faces coloured according to our new histogram, we
can use the {meth}`.SphereBase.create_shell_mesh` method.

```{code-cell} ipython3
my_marginal_direction_mesh = my_sphere.create_shell_mesh(
    my_marginal_direction_histogram
)
```

We can now create a {class}`.SpherePlotter` to plot the direction
histogram.

```{code-cell} ipython3
my_marginal_direction_sphere_plotter = vr.plotting.SpherePlotter(
    my_marginal_direction_mesh
)
my_marginal_direction_sphere_plotter.produce_plot()
my_marginal_direction_sphere_plotter.show()
```

Now we can see the directions of all the vectors, regardless of magnitude.

## Conditional Histograms

But now, what if we don't want to disregard the magnitude completely? Well,
VectoRose includes functions for constructing **conditional histograms**.
As with the marginal histograms, we can compute conditional histograms of
either variable, either studying the orientation for specific magnitude
values or studying the magnitudes for specific orientations.

```{important}
VectoRose 
  ```{todo}
  Talk about how it does it for all orientations and shells, then have to
  choose which ones to look at.
  ```
```

### Conditional Magnitude Histograms

To study the magnitudes in a specific orientation, we can construct
**conditional magnitude histograms** using the
{meth}`.SphereBase.construct_conditional_magnitude_histogram` method.