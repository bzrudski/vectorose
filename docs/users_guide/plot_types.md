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
# pv.global_theme.window_size = [1024, 1024]

# Set the random seed for reproducibility
RANDOM_SEED = 20241212

# Control the table display
pd.options.display.max_rows = 20

```

# Histogram Plotting

One of the main features of VectoRose is the ability to construct
**histograms**. In this page, we'll discuss the different types of
histograms that can be constructed using VectoRose.

## Much Ado About Histograms

Before we get too in-depth about VectoRose, let's talk about histograms.
**Histograms** are a data visualisation tool that present the frequency of
all possible data values.

Let's look at a simple 1D case. Let's say we have measurements of
individual heights. We can easily build a 1D histogram, consisting of
equal-width bars. Each bar has a height proportional to the number of
individuals with a height in the range covered by the respective bin.

```{code-cell} ipython3
---
tags: [remove-input, remove-stdout, remove-stderr]

mystnb:
    figure:
        align: center
        caption: |
            Heights in a very Gaussian population.
---
n = 15_000

# Create the average for our distribution
average_height = 150
std_dev = 20

random_heights = np.random.default_rng(RANDOM_SEED).normal(
    average_height, std_dev, size=n
)

plt.hist(random_heights)
plt.title("Heights in Gaussian Population")
plt.xlabel("Height (cm)")
plt.ylabel("Count")
plt.show()
```

This hopefully should not come as a surprise, as these types of histograms
are quite common. One thing that is important to note is that this
histogram of 1D data is actually a *2D plot* (heights and counts).

```{tip}
This type of histogram can be used to visualise 1D data, such as heights,
weights, ... or **vector magnitudes**.
```

Well, what about slightly more complicated data? Let's say we meet a group
of people and measure their heights and weights. We now have a collection
of data pairs, or two measured **variables**. While we *can* plot the
heights and widths separately, looking at these two variables separately
doesn't give the whole picture. The measurements may be correlated, and
certain values of height may be more common for certain values of weight.
As a solution, we can produce a **2D histogram**, like so:

```{code-cell} ipython3
---
tags: [remove-input, remove-stdout, remove-stderr]

mystnb:
    figure:
        align: center
        caption: |
            Heights and weights in a very Gaussian population.
---

# Create the weight distribution
average_weight = 65
weight_std_dev = 10

random_weights = np.random.default_rng(RANDOM_SEED).normal(
    average_weight, weight_std_dev, size=n
)

plt.hist2d(random_heights, random_weights)
plt.title("Heights and Weights in Gaussian Population")
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.colorbar(label="Count")
plt.show()
```

This 2D histogram is essentially an image, where each pixel represents a
range of height and weight values. The colour intensity reflects the number
of measurements falling into that bin for *both* variables. This plot can
be thought of as three-dimensional (height, weight, count). Alternatively,
this histogram can be plotted using a surface, with the heights
corresponding to the bin counts.

```{tip}
The important idea here is that a histogram plot is always one dimension
**higher** than the data it represents. The plot must be able to capture
all possible data values **and** represent the *counts* or *frequencies* of
the observed data.
```

## Magnitude Histograms

As we saw in our {doc}`Introduction to Vectors <vectors_intro>`, vectors
have a scalar magnitude. We can easily construct a histogram to show the
frequencies of different vector magnitudes. We can use functions in NumPy
to bin the data. Here, we'll use {func}`numpy.histogram`. VectoRose
includes the function {func}`.produce_1d_scalar_histogram` in the
{mod}`.plotting` module that can be used to plot the histogram.

Throughout this section of the **Users' Guide**, we'll do a running example
using some random vectors stored in the file
{download}`random_vectors.npy <./random_vectors.npy>`.

Before getting into the histogram plots, let's load these vectors from the
file. We'll assume that the represent vectorial data, but we'll still make
sure to remove any zero-vectors.

```{attention}
As always, remember to start your code with `import vectorose as vr` to be
able to access everything included in VectoRose.
```

```{code-cell} ipython3
import vectorose as vr

# Load the vectors
my_vectors = vr.io.import_vector_field("random_vectors.npy", location_columns=None)
my_vectors = vr.util.remove_zero_vectors(my_vectors)

my_vectors
```

Before we can construct the histogram, we must compute the vector
magnitudes. We can do this using {func}`numpy.linalg.norm`:

```{code-cell} ipython3
import numpy as np
magnitudes = np.linalg.norm(my_vectors, axis=-1)

magnitudes
```

```{note}
We must set `axis=-1` to compute the magnitude of each vector individually.
```

We can now compute the histogram. Let's consider 10 bins.

```{code-cell} ipython3
magnitude_counts, magnitude_bins = np.histogram(magnitudes, bins=10)

magnitude_counts
```

Using VectoRose, we can generate the 1D plot using the {mod}`.plotting`.

```{code-cell} ipython3
---
mystnb:
    figure:
        align: center
---
ax = vr.plotting.produce_1d_scalar_histogram(
    magnitude_counts, magnitude_bins
)
ax.set_title("Magnitude Histogram")
ax.set_xlabel("Magnitude")
ax.set_ylabel("Count")
plt.show()
```

These scalar histograms give us some very basic insight into our collection
of vectors. We can tell how many vectors have high magnitudes and low
magnitudes. But, this is just the beginning of what we can learn from
vectors.

## Direction and Orientation Histograms

So, we've now seen how to generate histograms of scalar data. While these
can provide insight into vector magnitudes, they aren't effective for
studying orientations and directions. It is important to recall that
directions and orientations are **not** scalar values.

As we explained in our {doc}`Introduction to Vectors <vectors_intro>`, we
can represent orientations and directions in spherical coordinates using
two angles:

* $\phi$ - the angle of inclination from the positive *z*-axis, known as
  the *colatitude*.
* $\theta$ - the clockwise angle in the *xy*-plane, measured from the
  positive *y*-axis, known as the *azimuthal angle*.

```{note}
Recall the valid angular ranges for each angle:
* $0^\circ \leq \theta < 360^\circ$ for both vectorial and axial data.
* $0^\circ \leq \phi \leq 180^\circ$ for vectorial data;
  $0^\circ \leq \phi \leq 90^\circ$ for axial data.
```

We can analyse these angles *separately* using
[polar histograms](#polar-histograms), or we can analyse the true
directions and orientations using
[spherical histograms](#spherical-histograms).

### Polar Histograms

We can start by studying the two angles $\phi$ and $\theta$ separately.
While these values *can* be plotted on a conventional linear histogram,
this doesn't fully represent the data we are studying. On a linear
histogram, the angles 2&#x00b0; and 358&#x00b0; are very far from each
other, but in reality, they are only 4&#x00b0; apart!

To take advantage of the circular nature of angles we can use **polar
histograms**. These histograms are *circular* and show bars radiating from
the centre of the circle. The bar heights still reflect the proportion of
vectors having an angle in each bin. These plots allow simpler
interpretation and better representation of the data.

In VectoRose, we can construct polar histograms using the
{mod}`~vectorose.polar_data` module, and we can visualise the results using
functions from the {mod}`~vectorose.plotting` module.

The process begins with the {class}`~vectorose.polar_data.PolarDiscretiser`
class. When constructing this class, you must specify the number of angular
bins to consider for both $\phi$ and $\theta$, and indicate whether the
data under consideration are axial.

Returning to our example, we can discretise our loaded vectors based
on orientation using {class}`~vectorose.polar_data.PolarDiscretiser`. The
first step is to create an object from this class, and then we must pass
our vectors to the method {meth}`.PolarDiscretiser.assign_histogram_bins`.
This method produces a new table of vectors with some extra columns,
providing the spherical coordinates of each vector, as well as the angular
bin for both $\phi$ and $\theta$.

```{code-cell} ipython3
# Begin the process of constructing the angular histograms
my_polar_discretiser = vr.polar_data.PolarDiscretiser(
    number_of_phi_bins=18,
    number_of_theta_bins=36,
    is_axial=False
)

my_labelled_vectors = my_polar_discretiser.assign_histogram_bins(my_vectors)

my_labelled_vectors
```

Now we see that each vector has been assigned an angular bin. Remember, in
Python indexing starts at **zero**, so the first bin has index `0`.

This process has produced a labelling for each vector, but it hasn't yet
given us a histogram. We can compute the $\phi$ and $\theta$ histograms,
respectively, using the methods
{meth}`.PolarDiscretiser.construct_phi_histogram` and
{meth}`.PolarDiscretiser.construct_theta_histogram`. First, let's look at
the $\phi$ histogram:

```{code-cell} ipython3
phi_histogram = my_polar_discretiser.construct_phi_histogram(my_labelled_vectors)

phi_histogram
```

And now, for the theta histogram:

```{code-cell} ipython3
theta_histogram = my_polar_discretiser.construct_theta_histogram(my_labelled_vectors)

theta_histogram
```

Notice that the bins reflect the different angular ranges of $\phi$ and
$\theta$. For each histogram, we have the start and end angles of each bin,
as well as the count and frequency associated with each.

At this point, you may be thinking, "this is great, but I signed up for
a histogram **plot**, not just a table of numbers!" Well, now we switch
over to the functions in {mod}`vectorose.plotting` to visualise our polar
histograms. The individual histograms can be constructed *separately* using
the function {func}`.produce_polar_histogram_plot`, or *together* using the
function {func}`.produce_phi_theta_polar_histogram_plots`. We'll
demonstrate the latter.

```{code-cell} ipython3
---
mystnb:
    figure:
        align: center
---
phi_theta_figure = vr.plotting.produce_phi_theta_polar_histogram_plots(
  phi_histogram, theta_histogram
)
```

We now have two polar histogram plots showing the distribution of our data.
For the ability to customise these plots, check out all the parameters for
{func}`.produce_phi_theta_polar_histogram_plots`, and for more flexibility
consult {func}`.produce_polar_histogram_plot`.

```{tip}
For more information about plotting and analysing circular data, check out
{cite:t}`fisherStatisticalAnalysisCircular1995`.
```

### Spherical Histograms

These polar histograms provide some insight into the directions present,
but like in the discussion about height and weight above, looking at these
angles separately doesn't give us a perfect picture of how the data are
arranged in space. We need to visualise both angles together... on a
**sphere**.

As we mentioned in the {doc}`Introduction to Vectors <vectors_intro>`,
the two angles describing direction and orientation can also describe
positions on a sphere (or hemisphere). So, to visualise the freqencies
associated with each orientation, we need to take a sphere and colour its
surface in different patches to reflect the number of vectors present
within each orientation bin.

And so, this brings up an important question: how can we tile a sphere?

#### Tiling the Sphere

The answer is not so trivial. There are many different ways to divide the
surface of a sphere.

##### UV Spheres

The simplest way is to wrap a flat 2D histogram onto a
sphere. In this case, we would define a constant angular bin width for
$\phi$ and another constant angular bin width in $\theta$, and overlay a
grid on the sphere. This is similar to overlaying the latitude and
longitude lines onto the surface of a globe. In computer graphics, this
type of sphere is known as a **UV sphere**.

```{code-cell} ipython3
:tags: [remove-input,remove-stdout,remove-stderr]

my_uv_sphere = vr.plotting.construct_uv_sphere_mesh(16, 36)

plotter = pv.Plotter()
plotter.add_mesh(my_uv_sphere, show_edges=True)
plotter.show()
```

This type of sphere is trivial to construct and the histogram simply
involves considering the pairs of $\phi$ and $\theta$ bins computed in the
polar case. However, the faces in the sphere have very different surface
areas. The bins at the equator are much larger than those at the poles.
This leads to difficulties in interpretation: does a larger patch have a
higher count due to properties of the data, or simply by virtue of the fact
that it is larger?

##### Triangulated Spheres

So, the UV sphere is very problematic. An alternative solution involves
tiling the sphere using triangles, as is done in a geodesic dome (Montreal,
where we have developed this package, is quite famous for
[one](https://en.wikipedia.org/wiki/Montreal_Biosphere)). In this tiling,
all faces are triangular and are similar (but not identical) in area.

```{code-cell} ipython3
:tags: [remove-input,remove-stdout,remove-stderr]

triangle_sphere = vr.triangle_sphere.TriangleSphere(number_of_subdivisions=4)
triangle_sphere_mesh = triangle_sphere.create_mesh()

plotter = pv.Plotter()
plotter.add_mesh(triangle_sphere_mesh, scalars=None, color="white", show_edges=True)
plotter.show()
```

Unfortunately, the triangles are not defined as a simple function of the
spherical coordinates, which makes the binning process more complicated.

##### Tregenza Sphere

To resolve the issues present with the UV, continuing work undertaken by
{cite:t}`tregenzaSubdivisionSkyHemisphere1987`,
{cite:t}`beckersGeneralRuleDisk2012` developed a new incongruent method to
approximate a sphere using rectangular patches of **near-equal area**.
{cite:t}`beckersGeneralRuleDisk2012` first
divided the sphere into a series of rings using *almucantars* based on a
constant angle of inclination. Each ring is then subdivided into
rectangular patches based on a consistent azimuthal angle specific to that
ring. This pattern achieves near-equal area patch sizing across the entire
sphere. Instead of a triangular fan, the sphere pole is filled with a
single polygonal cap, approximating a small circle.

Although this technique reduces most discrepancies in face area, near the
pole, face areas may still deviate by up to 21%. We have modified the
approach presented by {cite:t}`beckersGeneralRuleDisk2012` to
ensure that the top rings of the sphere better approximate equal-area
patching. While a consistent $\phi$-spacing is maintained for bins close to
the equator, we manually set a smaller inclination angle for the first two
rings to ensure that the face areas are closer to being equal. We have
implemented three levels of granularity for these spheres:

* Coarse - contains 18 rings and 520 patches.
* Fine - contains 54 rings and 5806 patches.
* Ultra fine - contains 124 rings and 36956 patches.

```{code-cell} ipython3
:tags: [remove-input,remove-stdout,remove-stderr]

my_tregenza_sphere = vr.tregenza_sphere.FineTregenzaSphere()
my_sphere_mesh = my_tregenza_sphere.create_mesh()

plotter = pv.Plotter()
plotter.add_mesh(my_sphere_mesh, scalars=None, color="white", show_edges=True)
plotter.show()
```

Using our modified technique, patch areas are more similar, but minor area
deviations can persist in each sphere.

Although this representation of the sphere may appear more complicated,
since each ring is defined in terms of a start and end $\phi$ angle and
each bin within a ring has the same $\theta$ width, assigning histogram
bins remains very straightforward:

* First the $\phi$ angle is used to determine the appropriate ring.
* Then the $\theta$ angle is used to determine the closest bin within the
  ring almucantar.

Similar to the UV sphere, but unlike the triangulated sphere, this
representation has a visible sphere pole above a series of rings of
rectangular patches.

##### Comparison

As we discussed, an important criterion for a spherical histogram is that
the sphere faces have approximately equal area. We can construct each and
look at the deviations from the average area in each.

```{code-cell}
:tags: [remove-input,remove-stdout,remove-stderr]

# Step 1 - Construct a simple UV sphere
uv_sphere = vr.plotting.construct_uv_sphere_mesh(phi_steps=45, theta_steps=90)
uv_sphere = uv_sphere.compute_cell_sizes(length=False, area=True, volume=False)
uv_face_areas = uv_sphere.cell_data["Area"]
uv_mean_face_area = uv_face_areas.mean()
uv_area_deviation = (uv_face_areas - uv_mean_face_area) / uv_mean_face_area * 100
uv_sphere.cell_data["Deviation from Mean Area"] = uv_area_deviation

# Step 2 - Construct the triangulated icosphere
triangle_sphere = vr.triangle_sphere.TriangleSphere(number_of_subdivisions=4)
triangle_sphere_mesh = triangle_sphere.create_mesh()
triangle_sphere_mesh = triangle_sphere_mesh.compute_cell_sizes(
    length=False, area=True, volume=False
)
triangle_face_areas = triangle_sphere_mesh.cell_data["Area"]
triangle_mean_face_area = triangle_face_areas.mean()
triangle_area_deviation = (
    (triangle_face_areas - triangle_mean_face_area) / triangle_mean_face_area * 100
)
triangle_sphere_mesh.cell_data["Deviation from Mean Area"] = triangle_area_deviation

# Step 3 - Construct the Tregenza sphere
tregenza_sphere = vr.tregenza_sphere.FineTregenzaSphere()
tregenza_sphere_mesh = tregenza_sphere.create_mesh()
tregenza_sphere_mesh = tregenza_sphere_mesh.compute_cell_sizes(
    length=False, area=True, volume=False
)
tregenza_face_areas = tregenza_sphere_mesh.cell_data["Area"]
tregenza_mean_face_area = tregenza_face_areas.mean()
tregenza_area_deviation = (
    (tregenza_face_areas - tregenza_mean_face_area) / tregenza_mean_face_area * 100
)
tregenza_sphere_mesh.cell_data["Deviation from Mean Area"] = tregenza_area_deviation

# And now, to plot everything.
all_deviations = np.concatenate(
    [uv_area_deviation, tregenza_area_deviation, triangle_area_deviation]
)
all_deviations = np.abs(all_deviations)
max_deviation = all_deviations.max()
min_deviation = -max_deviation

clim = [min_deviation, max_deviation]
cmap = "RdBu_r"

# Create the plotter
plotter = pv.Plotter(shape=(1, 3), border=False)
plotter.enable_parallel_projection()

plotter.subplot(0, 0)
plotter.add_mesh(
    uv_sphere,
    scalars="Deviation from Mean Area",
    clim=clim,
    cmap=cmap,
    show_edges=True,
    show_scalar_bar=False,
)
plotter.add_text("UV sphere", font_size=20, position="lower_edge")

plotter.subplot(0, 1)
plotter.add_mesh(
    triangle_sphere_mesh,
    scalars="Deviation from Mean Area",
    clim=clim,
    cmap=cmap,
    show_edges=True,
    scalar_bar_args={
        "title": "Deviation from Mean Area (%)",
        "position_y": 0.8,
        "position_x": 0.1,
        "width": 0.8,
        "title_font_size": 40,
        "label_font_size": 40,
        "unconstrained_font_size": True,
    },
)
plotter.add_text("Triangulated sphere", font_size=20, position="lower_edge")

plotter.subplot(0, 2)
plotter.add_mesh(
    tregenza_sphere_mesh,
    scalars="Deviation from Mean Area",
    clim=clim,
    cmap=cmap,
    show_edges=True,
    show_scalar_bar=False,
)
plotter.add_text("Tregenza sphere", font_size=20, position="lower_edge")

plotter.link_views()

plotter.show()
```

#### Constructing Spherical Histograms

Now that we've discussed how to tile the sphere, we can actually construct
histograms on these spheres. In VectoRose, we provide tools to produce
histograms on both the **triangulated** sphere and the **Tregenza** sphere.
The triangulated sphere is represented using the class
{class}`.triangle_sphere.TriangleSphere` while the Tregenza sphere is
defined using {class}`.tregenza_sphere.TregenzaSphere`. For simplicity, we
have provided three levels of discretisation for the Tregenza sphere,
discussed above. These are implemented as {class}`.CoarseTregenzaSphere`,
{class}`.FineTregenzaSphere` and {class}`.UltraFineTregenzaSphere`.

Both {class}`.TregenzaSphere` and {class}`.TriangleSphere` inherit from the
abstract class {class}`.sphere_base.SphereBase`, which defines all the
functions necessary for computing orientation histograms. As a result, the
workflow is very similar for both; we will demonstrate using the fine
Tregenza sphere.

First, to be able to construct the spherical histogram, we must create a
sphere object. Since we are using the Tregenza sphere, we run the following
code:

```{code-cell} ipython3
my_sphere = vr.tregenza_sphere.FineTregenzaSphere()
```

We can view the structure of the sphere by converting it to a pandas
{class}`~pandas.DataFrame` object using the
{meth}`.TregenzaSphere.toDataFrame` method:

```{code-cell}
my_sphere.to_dataframe()
```

Now we can see exactly how our sphere is made. This sphere gives us a tool
that we can use to assign histogram bins, similar to what we did earlier
in the [Polar Histograms](#polar-histograms) section. Let's assign our
vectors, which are still stored in the variable `my_vectors`, to histogram
bins using the method {meth}`.SphereBase.assign_histogram_bins`. This
function produces two outputs: labelled vectors and bins for a magnitude
histogram. We'll worry about the second one
[a bit later](#vector-histograms).

```{code-cell} ipython3
labelled_vectors, _ = my_sphere.assign_histogram_bins(my_vectors)

labelled_vectors
```

Now we can see that each vector has some extra data: we have the spherical
coordinates, as well as columns for the **ring** and **bin** that each
vector falls in. We'll discuss later what the **shell** means.

We can now gather these labelled vectors into a histogram using
{meth}`.SphereBase.construct_histogram`. We can choose to either get the
actual number of vectors in each face, or the fraction of vectors in each
face.

```{code-cell} ipython3
my_histogram = my_sphere.construct_histogram(labelled_vectors)

my_histogram.to_frame()
```

Looking at this table isn't terribly informative. To visualise the
orientation histogram in 3D, we need to construct a sphere mesh with the
corresponding face values. We can easily do this using the method
{meth}`.SphereBase.create_histogram_meshes`.

```{code-cell} ipython3
my_histogram_meshes = my_sphere.create_histogram_meshes(
    my_histogram, magnitude_bins=None
)
```

We can now visualise the histogram using the {class}`.SpherePlotter` class
in the {mod}`.plotting` module.

```{tip}
The histogram is a {class}`pandas.Series` object, so you can leverage all
the functions defined by pandas to export these data.
```

Let's produce the histogram plot. First, we need to create a
{class}`.SpherePlotter` object with the histogram meshes. Then, we must
create the plot using the {meth}`.SpherePlotter.produce_plot` method, and
then we can show the plot using {meth}`.SpherePlotter.show`.

```{code-cell} ipython3
my_sphere_plotter = vr.plotting.SpherePlotter(my_histogram_meshes)
my_sphere_plotter.produce_plot()
my_sphere_plotter.show()
```

```{danger}
In order to add the spherical histogram to the plot, you **must** call the
{meth}`.SpherePlotter.produce_plot` method. Otherwise, no spheres will
appear!
```

There are different parameters for each method. Please consult the
documentation for each method to learn about all the parameters.

```{tip}
Confused about the angles? You can add spherical axes showing the $\phi$
and $\theta$ labels and ticks using the method 
{meth}`.SpherePlotter.add_spherical_axes`.
```

So, we now have 3D sphere plots! You are probably wondering how you can
take these beautiful plots and share them with the world. Good news!
VectoRose allows you to easily export images and videos of your spherical
histograms.

To export your plot as a raster image (PNG, TIFF, JPEG, BMP) you may call
the method {meth}`.SpherePlotter.export_screenshot`. To preserve any
text annotations, you can also export the image as a vector graphic (PDF,
SVG and more) using {meth}`.SpherePlotter.export_graphic`. Finally, you can
export a video of your sphere spinning about its vertical axis using
{meth}`.SpherePlotter.produce_rotating_video`.

```{code-cell} ipython3
:tags: [remove-cell]
import os
export_dir = "./assets/rotating_video/"
if not os.path.isdir(export_dir):
  os.mkdir(export_dir)

my_sphere_plotter.export_screenshot(
  "./assets/rotating_video/rotating_video.png",
  False
)
```

```{code-cell} ipython3
my_sphere_plotter.produce_rotating_video(
  "./assets/rotating_video/rotating_video.mp4",
  quality=5,
  fps=12,
  number_of_frames=36,
  hide_sliders=True
)
```


```{video} ./assets/rotating_video/rotating_video.mp4
:width: 100%
:autoplay:
:loop:
:poster: ./assets/rotating_video/rotating_video.png
:alt: Example video of the histogram rotating.
```

Each function has a number of possible parameters. Please consult the
documentation for each.

```{seealso}
Orientation histograms can also be plotted in 3D using Matplotlib. Check
out the functions {func}`.plotting.produce_3d_triangle_sphere_plot` and
{func}`.plotting.produce_3d_tregenza_sphere_plot`.
```

The workflow for using a triangulated sphere is almost identical. Simply
replace the {class}`.FineTregenzaSphere` with {class}`.TriangleSphere` in
the code above.

## Vector Histograms

We've now seen how to construct 1D histograms of vector magnitude and
spherical histograms of vector orientation. Each of these gives us
important information, but studying how they relate to each other may
provide us with additional insight.

So, how can we study the two together?

Like we said before, a histogram has to show the frequency at all possible
data values. In the case of non-unit vectors, then we need to find a way of
showing the frequency at all possible combinations of orientation and
magnitude.

In VectoRose, we do this by creating **nested spherical histograms**. Each
spherical shell represents a certain magnitude level, with the smallest,
innermost sphere corresponding to the lowest-magnitude vectors and the
largest, outermost sphere corresponding to the highest-magnitude vectors.
By default, the colour map is **universal**, colouring each sphere patch
based on the frequency across all shells and all faces.

To generate spheres with multiple shells, we can use the same sphere
objects as before, {class}`.TriangleSphere` and {class}`.TregenzaSphere`.
The important thing is to now pass the `number_of_shells` parameter.

Let's now take our same vectors from before, and consider ten histogram
shells.

```{code-cell} ipython3
my_sphere = vr.tregenza_sphere.FineTregenzaSphere(number_of_shells=10)
labelled_vectors, magnitude_bin_edges = my_sphere.assign_histogram_bins(my_vectors)

labelled_vectors
```

Here there are a couple of small differences from our previous
demonstration:

* The keyword argument `number_of_shells=10` specifies that we want 10
  magnitude shells.
* We store the magnitude bin edges from the bin assignment in a variable
  `magnitude_bin_edges`.
* Our labelled vectors now have non-zero values in the **shell** column.

```{attention}
When assigning the magnitude bins, we **exclude** the lower bin limit and
**include** the upper bin limit. This is done to avoid counting zero-length
vectors.
```

We now once again need to create the histogram using
{meth}`.SphereBase.construct_histogram`.

```{code-cell} ipython3
my_histogram = my_sphere.construct_histogram(labelled_vectors)
```

To plot the spherical histogram, we once again have to generate the
histogram meshes using {meth}`.SphereBase.create_histogram_meshes`.

```{code-cell} ipython3
my_histogram_meshes = my_sphere.create_histogram_meshes(
    my_histogram, magnitude_bins=magnitude_bin_edges
)
```

We pass in the `magnitude_bin_edges` to be able to set the radius for each
sphere.

As before, we can use the {class}`.SpherePlotter` to visualise the plots.

```{code-cell} ipython3
my_sphere_plotter = vr.plotting.SpherePlotter(my_histogram_meshes)
my_sphere_plotter.produce_plot()
my_sphere_plotter.show()
```

Our plot is similar, but we now have sliders that can help us activate
individual shells and adjust the opacity of the active shell and the
inactive shell.

```{attention}
If you're reading this page on our website in a web browser, you probably
can't see the sliders. This is normal due to how the static documentation
is rendered. To be able to take full advantage of the plotting features,
open this file using Jupyter Lab or Jupyter Notebooks, and make sure to
set the backend to `"trame"` instead of `"html"` at the top of the file.
```

In addition to everything that we can do with a spherical histogram, we can
also export a video that iterates through the different shells using the
method {meth}`.SpherePlotter.produce_shells_video`.

```{code-cell} ipython3
:tags: [remove-cell]

export_dir = "./assets/shells_video/"
if not os.path.isdir(export_dir):
  os.mkdir(export_dir)

my_sphere_plotter.export_screenshot(
  "./assets/shells_video/shells_video.png",
  False
)
```

```{code-cell} ipython3
my_sphere_plotter.produce_shells_video(
  "./assets/shells_video/shells_video.mp4",
  quality=5,
  fps=4,
  boomerang=True,
  add_shell_text=True,
  hide_sliders=True
)
```

```{video} ./assets/shells_video/shells_video.mp4
:width: 100%
:autoplay:
:loop:
:poster: ./assets/shells_video/shells_video.png
:alt: Example video of the histogram shells.
```

In this plot, we see not only where the vectors are pointing and what
their magnitudes are, but the combination of the two. We can see which
orientations are associated with higher magnitudes. This information could
be useful in downstream analyses.

In some cases, the signal may be quite low on some of the shells. In order
to make the patterns more visible, the frequency values may be normalised
within each shell. This option is set when generating the histogram meshes
using the keyword argument `normalise_by_shell=True` in
{meth}`.SphereBase.create_histogram_meshes`. In this case, all faces have a
value between 0 and 1, representing the fraction of the shell maximum value
stored in the face.

```{tip}
We'll see more about what these normalised face values mean in the next
section.
```

## Clean-Up and Summary

One last thing: when you're done plotting, make sure to close the sphere
plotter using the {meth}`.SpherePlotter.close` method.

```{code-cell} ipython3
my_sphere_plotter.close()
```

By doing this, you can make sure that the resources used to generate the
plot are freed up. This will make it easier to produce additional plots.

Now we've seen how to generate 1D, spherical and nested spherical
histograms. In the next section, we'll see a bit more about how to generate
these histograms from a single {class}`.SphereBase` object and how to gain
additional insights.
