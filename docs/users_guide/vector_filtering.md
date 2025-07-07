---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Vector Filtering

```{code-cell} ipython3
:tags: [remove-cell]

# Configure PyVista behind the scenes before starting
import matplotlib.pyplot as plt
import pandas as pd
import pyvista as pv

try:
    pv.start_xvfb()
except OSError:
    pass

pv.set_jupyter_backend("html")
pv.global_theme.font.fmt = "%.6g"

pd.options.display.max_rows = 20
```

At this point, we've now seen how to generate rich histogram visualisations
that allow us to observe patterns in a collection of vectors. But, what if
we want to take things a step further and only study vectors with a
magnitude in a specific range, or with a specific orientation?

In this section, we'll see how to *filter* vector datasets based on
magnitude and orientation.

```{important}
When we use the term *filter*, we are referring to selecting specific
vectors on the basis of magnitude and/or orientation.
```

We'll continue with our {download}`two_clusters.npy <./two_clusters.npy>`
dataset, which can also be loaded directly in VectoRose without any
separate download using {attr}`.SampleData.TWO_CLUSTERS` in the
{mod}`.data` module.

As usual, let's start by loading the data.

```{code-cell} ipython3
import numpy as np # We'll use NumPy a bit later

import vectorose as vr
import vectorose.data

my_vectors = vr.data.SampleData.TWO_CLUSTERS.load()
my_vectors = vr.util.remove_zero_vectors(my_vectors)

my_vectors
```

As usual, let's also create a {class}`.FineTregenzaSphere` with 10 nested
shells to use for bin assignment in terms of magnitude and orientation.

```{code-cell} ipython3
my_sphere = vr.tregenza_sphere.FineTregenzaSphere(number_of_shells=10)
labelled_vectors, magnitude_bins = my_sphere.assign_histogram_bins(my_vectors)

labelled_vectors
```

Our vectors have now been assigned to bins based on magnitude and
orientation. With this in mind, we can already see a couple of possible
ways that we may want to filter:

* based on **magnitude** using the shell index, or
* based on **orientation** using the angular bin index.

With these ideas in mind, let's get started on filtering!

```{tip}
This page is not a comprehensive guide on filtering. Since vector datasets
are represented as NumPy arrays and pandas DataFrames, the possibilities
are quite endless. This page seeks to show some simple workflows for
filtering using the representations and functions included in VectoRose.
```

Now that we have the vectors assigned to histogram bins, we can use the
built-in tools provided by [**pandas**](https://pandas.pydata.org/) to
perform filtering based on both magnitude and orientation. We'll also see
some novel tools provided by VectoRose to simplify this filtering process.

Before we get started, here's what the nested histogram shells for this
dataset look like:

```{video} ./assets/shells_video/shells_video.mp4
:width: 100%
:autoplay:
:loop:
:poster: ./assets/shells_video/shells_video.png
:alt: Example video of the histogram shells.
```

We'll keep this distribution in mind as we perform filtering.

## Magnitude Filtering

The vector magnitude is represented by a scalar value. Filtering based on
the scalar magnitude is quite trivial. Let's immediately dive into an
example to see this.

Looking at our histogram shells, it's quite clear that fifth shell contains
an interesting pattern. It contains a cluster with a high frequency. If we
want to study only the vectors in this magnitude level, we can simply look
for all vectors with a shell index of `4`.

```{warning}
Recall that indexing in Python starts at zero, so the fifth shell has index
**4**, not 5.
```

Let's see how we can do that in code:

```{code-cell} ipython
shell_4_vectors = labelled_vectors[labelled_vectors["shell"] == 4]

shell_4_vectors
```

Now, notice that all of our vectors are in shell 4. We can then convert
these vectors to a NumPy array using the method
:meth:`.SphereBase.convert_vectors_to_cartesian_array`:

```{code-cell} ipython
shell_4_vectors_array = my_sphere.convert_vectors_to_cartesian_array(
    shell_4_vectors,
)

shell_4_vectors_array
```

We can, of course, perform more complicated filtering tasks to extract the
vectors from multiple shells. For example, to select all vectors in the
fifth shell or higher, we can simply run:

```{code-cell} ipython3
labelled_vectors[labelled_vectors["shell"] >= 4]
```

We can get the other vectors by running:

```{code-cell} ipython3
labelled_vectors[labelled_vectors["shell"] < 4]
```

We can also combine both of these to select from a range of shells:

```{code-cell} ipython3
labelled_vectors[
    (4 <= labelled_vectors["shell"]) &  (labelled_vectors["shell"] <= 6)
]
```

We can also work directly based on the magnitude values, ignoring the shell
indices altogether:

```{code-cell} ipython3
labelled_vectors[labelled_vectors["magnitude"] <= 0.25]
```

For more details on basic indexing using pandas, make sure to check out
[this page](https://pandas.pydata.org/docs/user_guide/indexing.html) in the
pandas documentation.


## Orientation Filtering - Single Bin

Orientation presents a bit more of a challenge. On a basic level, we can do
something similar for filtering by orientation. Let's say we want all
vectors containing in the ring with index 15 and the bin with index 10 in
that ring. We can once again use pandas directly:

```{code-cell} ipython3
filtered_vectors = labelled_vectors[
    (labelled_vectors["ring"] == 15) & (labelled_vectors["bin"] == 10)
]

filtered_vectors
```

In practice, this isn't very helpful. It's hard to intuitively know what
bin we want for a specific orientation.

The good news is that we can use other tools in VectoRose to convert
between angles and face index information.

Let's plot the orientation histogram for our dataset to find the
orientations of interesting features. To help, we'll add angular $\phi$ and
$\theta$ axes using the method {meth}`.SpherePlotter.add_spherical_axes`.

```{code-cell} ipython3
orientation_histogram = my_sphere.construct_marginal_orientation_histogram(
    labelled_vectors
)

orientation_histogram_mesh = my_sphere.create_shell_mesh(orientation_histogram)

sphere_plotter = vr.plotting.SpherePlotter(orientation_histogram_mesh)
sphere_plotter.produce_plot()
sphere_plotter.add_spherical_axes()
sphere_plotter.show()
```

Using these axes, we can see the orientations of our two clusters in the
dataset. The upper cluster seems centred around $\phi=55$ and $\theta=0$
degrees.

Let's extract all the vectors that fall into the bin containing this
orientation. To do this, we just need to create a unit vector pointed in
that direction in Cartesian coordinates and pass it to
{meth}`.SphereBase.assign_histogram_bins` to get the closest bin.

```{code-cell} ipython3
my_spherical_coordinates = np.array([55, 0])
my_cartesian_coordinates = vr.util.convert_spherical_to_cartesian_coordinates(
    my_spherical_coordinates, radius=1, use_degrees=True
)

my_bin, _ = my_sphere.assign_histogram_bins(my_cartesian_coordinates)

my_bin
```

Now we see that the vectors we want to filter are found in ring 16, bin 0.
To extract these vectors, we can once again use the indexing features from
pandas:

```{code-cell} ipython3
vectors_in_cell = labelled_vectors.loc[
    (labelled_vectors["ring"] == my_bin.loc[0, "ring"])
    & (labelled_vectors["bin"] == my_bin.loc[0, "bin"])
]

print(f"We have extracted {len(vectors_in_cell)} vectors!")

vectors_in_cell
```

```{caution}
Don't forget to put the initial `0` index in
`my_bin.loc[0, "ring"]`{l=python}. Otherwise, Python will get unhappy. We
need to extract the ring and bin for the specific face of interest.
```

Using this approach, we can extract vectors from within single cells. But,
the syntax seems a bit long due to the explicit indexing for the ring and
the bin indices. This syntax also only works for Tregenza spheres. We
would need to figure something else out for triangulated spheres.

Thankfully, we don't have to go to this bother! VectoRose includes the
helpful method {meth}`.SphereBase.get_vectors_from_single_cell`. This
method takes in the bin information for a single cell, regardless of the
specific sphere implementation, and extracts all the vectors located in
that one cell.

```{attention}
The method {meth}`.SphereBase.get_vectors_from_single_cell` takes in two
arguments:

1. The {class}`~pandas.DataFrame` containing the labelled vectors as
   returned by {meth}`.SphereBase.assign_histogram_bins`.
2. A {class}`~pandas.Series` containing the information for the single bin
   examined. If **magnitude information is present**, then filtering will
   automatically happen by orientation *and* magnitude. If you do **not**
   want to filter by magnitude, **only pass in the orientation bin
   information**.

```

Here's the thing... We got a {class}`~pandas.DataFrame` from our call to
{meth}`.SphereBase.assign_histogram_bins`. We need to now just extract our
lone row. We can do that easily using the {attr}`~pandas.DataFrame.iloc`
attribute.

```{code-cell} ipython3
my_bin_series = my_bin.iloc[0]

my_bin_series
```

And now we're ready to perform the extraction using the call to
{meth}`.SphereBase.get_vectors_from_single_cell`.

```{code-cell} ipython3
vectors_in_cell = my_sphere.get_vectors_from_single_cell(
    labelled_vectors, my_bin_series
)

print(f"We have extracted {len(vectors_in_cell)} vectors!")

vectors_in_cell
```

But, wait! We have fewer rows here! What's going on?

The answer is that our `my_bin_series` contains the magnitude `shell`, so
filtering is done automatically by both magnitude and orientation. If we
want to just use the orientation, we can extract the `bin` and `ring` data:

```{code-cell} ipython3
vectors_in_cell = my_sphere.get_vectors_from_single_cell(
    labelled_vectors, my_bin_series[["ring", "bin"]]
)

print(f"We have extracted {len(vectors_in_cell)} vectors!")

vectors_in_cell
```

We now have the exact same result. And, we didn't need to figure out any
quirks of pandas indexing!

This method offers a flexible approach that will work regardless of whether
we are working with a {class}`.TregenzaSphere` or a
{class}`.TriangleSphere`.

But, what if we want to extract vectors from multiple cells? I'm glad you
asked...

## Orientation Filtering - Multiple Bins

Let's say we don't want to confine our filtering to a single bin. Well, the
solution is actually quite easy! Just like we have the method
{meth}`.SphereBase.get_vectors_from_single_cell` for getting vectors in a
single cell, we have the similar
{meth}`.SphereBase.get_vectors_from_selected_cells` to get the vectors
contained in multiple cells. It's actually even easier this time, because
we don't need to convert our cells into a {class}`~pandas.Series`. We can
**directly** pass in the {class}`~pandas.DataFrame` containing the cell
information.

Let's say we want to get the vectors from a couple of other cells. We want
to extract those centred around
$(\phi, \theta) = \{ (55, 0), (60, 5), (70, 10) \}$

Well, we can easily get the corresponding cell information, and then
extract the vectors using a similar pipeline.

```{code-cell} ipython3
spherical_coordinates = np.array(
    [
        [55, 0],
        [60, 5],
        [70, 10],
    ]
)

cartesian_coordinates = vr.util.convert_spherical_to_cartesian_coordinates(
    spherical_coordinates, radius=1, use_degrees=True
)

bin_indices, _ = my_sphere.assign_histogram_bins(cartesian_coordinates)

# Let's not filter by magnitude
orientation_bins = bin_indices[["ring", "bin"]]

orientation_bins
```

Now that we have our bins, let's extract our vectors!

```{code-cell} ipython3
vectors_in_cells = my_sphere.get_vectors_from_selected_cells(
    labelled_vectors, orientation_bins
)

print(
    f"We have extracted {len(vectors_in_cells)} vectors from the "
    f"{len(orientation_bins)} cells."
)

vectors_in_cells
```

Seems quite straight-forward, eh?

But... what if you don't know the angles?

### Interactively Selecting Histogram Cells

If you don't know the exact angles of your faces of interest, no problem!
VectoRose contains a way to interactively select the histogram cells to use
for this process. This interactivity is controlled by the
{class}`.SpherePlotter` class.

```{warning}
The interactive face selector only works when running VectoRose in a local
Python shell or using the `trame` renderer when using a Jupyter notebook.
It **will not appear** in the rendered HTML documentation. We have embedded
some videos to illustrate the process, but to get the full experience of
this example, please make sure to run the code locally.
```

To be able to interactively select histogram cells, you must set the
property {attr}`.SpherePlotter.cell_picking_active` to be `True`. Then, in
the interactive plotter, you can select cells by **right-clicking** them.
To deselect a cell, you must **right-click** again.

```{tip}
Cell selection is done by **right-clicking**.
```

When a cell is selected, it will appear to have a thick magenta border.

```{video} ./assets/cell_picking/cell_picking.mp4
:width: 100%
:autoplay:
:loop:
:poster: ./assets/cell_picking/cell_picking.png
:alt: Demonstration of cell picking.

```

To clear selected cells, simply call
{meth}`.SpherePlotter.clear_picked_cells`. The picked cells are cleared
automatically if cell picking is deactivated.

```{warning}
For reasons potentially beyond our control, it may be difficult to select
certain cells (for example, the poles of a Tregenza sphere). We have
provided programmatic ways to select cells, shown below. See
{meth}`.SpherePlotter.pick_cells` and {meth}`.SphereBase.get_cell_indices`
for the two key methods.
```

Once the cells are picked, you can access the bin information for the
selected cells through the property {attr}`.SpherePlotter.picked_cells`.
The {class}`~pandas.DataFrame` provided by this property can then be passed
**directly** to {meth}`.SphereBase.get_vectors_from_selected_cells` to
extract the vectors.

Let's do a demonstration with the three cells we considered earlier, which
are still stored in the variable `orientation_bins`. Since this example is
rendered automatically in HTML, we'll programmatically select the cells.

```{code-cell}
orientation_bin_cell_indices = my_sphere.get_cell_indices(orientation_bins)

orientation_plotter = vr.plotting.SpherePlotter(
    orientation_histogram_mesh
)
orientation_plotter.produce_plot()
orientation_plotter.cell_picking_active = True
orientation_plotter.pick_cells(orientation_bin_cell_indices)
orientation_plotter.show()
```

```{code-cell} ipython3
:tags: [remove-cell]

orientation_plotter.rotate_to_view(phi=55, theta=0, zoom=1.5)
orientation_plotter.export_screenshot("./assets/cell_picking/picked_cells.png")

orientation_plotter.open_movie_file(
    "./assets/cell_picking/picked_cells.mp4",
    fps = 5
)

phis = np.linspace(40, 70, 20)
thetas = np.linspace(0, 15, 20)
zoom_factors = np.linspace(1, 1.1, 20)

for phi, theta, zoom in zip(phis, thetas, zoom_factors):
    orientation_plotter.rotate_to_view(phi=phi, theta=theta, zoom=zoom)
    orientation_plotter.write_frame()
    
for phi, theta, zoom in zip(reversed(phis), reversed(thetas), reversed(1/zoom_factors)):
    orientation_plotter.rotate_to_view(phi=phi, theta=theta, zoom=zoom)
    orientation_plotter.write_frame()
    
orientation_plotter.close_movie()
```

```{video} ./assets/cell_picking/picked_cells.mp4
:width: 100%
:autoplay:
:loop:
:poster: ./assets/cell_picking/picked_cells.png
:alt: Demonstration of picked cells.

```

Now that we have our cells picked, we can get the cell information and
extract the vectors.

```{code-cell} ipython3
picked_cells = orientation_plotter.picked_cells
vectors_in_cells = my_sphere.get_vectors_from_selected_cells(
    labelled_vectors, picked_cells
)

print(f"Extracted {len(vectors_in_cells)} vectors from the {len(picked_cells)} picked cells")

vectors_in_cells
```

Using this approach, we can easily filter vectors based on user-defined
cells of interest.

## Other Filtering Approaches

As we mentioned above, this tutorial is not a comprehensive guide on vector
filtering. There are many approaches that we haven't gone into here. For
example, by computing arc lengths using the function
{func}`.util.compute_arc_lengths`, it is possible to filter vectors based
on angular distance from a reference orientation. This could be useful for
separating the two clusters in our dataset. Many of these approaches rely
more heavily on the capabilities of [pandas](https://pandas.pydata.org/),
rather than new features introduced by VectoRose.

VectoRose also provides the ability to filter based on both magnitude and
orientation. Combining these two variables provides the user with much
greater control over which vectors are kept for analysis.

## Conclusion

In this guide, we have seen the basics of performing vector filtering using
VectoRose and pandas. You can now easily extract vectors from individual
histogram cells, or from collections of cells. These operations enable rich
analyses of directed data.

Before leaving, let's close our {class}`.SpherePlotter` objects to release
the resources back to the operating system.

```{code-cell} ipython3
sphere_plotter.close()
orientation_plotter.close()
```

Now, you are equipped to not only construct histograms using VectoRose, but
also to select specific data points to analyse further.
