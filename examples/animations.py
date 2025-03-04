"""
Animations
==========

Export animations based on 3D spherical plots.

Data Overview
-------------

In this example, we will use the :mod:`.mock_data` module to generate
random vectors following different cluster distributions.

A **cluster** appears on a spherical histogram when many vectors fall in a
round pattern near a specific point.

Let's start by creating our dataset, and then we'll visualise it using
VectoRose.

.. warning::
    The :mod:`.mock_data` module is not pre-imported into ``vectorose``, so
    it **must** be explicitly imported.

.. note::
    Due to some quirks with using Sphinx-Gallery with PyVista, the
    underlying :class:`pyvista.Plotter` objects need to be recreated in
    each code cell. The code required to perform this task is hidden in the
    rendered HTML version of the documentation and in the exported Jupyter
    notebook, but not in the exported Python source code.
"""

# sphinx_gallery_thumbnail_number = 2

import vectorose as vr
import vectorose.mock_data

# Set the random seed for reproducibility
RANDOM_SEED = 20250120

my_vectors = vr.mock_data.create_von_mises_fisher_vectors_multiple_directions(
    phis=[30, 78, 127, 160],
    thetas=[70, 50, 130, 300],
    kappas=[20, 8, 10, 12],
    numbers_of_vectors=50000,
    magnitudes=[0.4, 0.7, 0.2, 0.8],
    magnitude_stds=[0.1, 0.1, 0.05, 0.05],
    use_degrees=True,
    seeds=[RANDOM_SEED, RANDOM_SEED + 1, RANDOM_SEED + 2, RANDOM_SEED + 3],
)

# %%
# Histogram Construction
# ----------------------
# We'll use the **fine Tregenza sphere** to construct our histograms.
# To allow fine-grained analysis of the magnitudes, we'll use 32 shells.
# As usual, we will assign the vectors to bins, create the histograms and
# construct the meshes that we will use to visualise the histograms.

my_sphere = vr.tregenza_sphere.FineTregenzaSphere(number_of_shells=32)
labelled_vectors, magnitude_bins = my_sphere.assign_histogram_bins(my_vectors)

my_bivariate_histogram = my_sphere.construct_histogram(labelled_vectors)
my_orientation_histogram = my_sphere.construct_marginal_orientation_histogram(
    labelled_vectors
)

my_bivariate_meshes = my_sphere.create_histogram_meshes(my_bivariate_histogram, magnitude_bins)
my_orientation_mesh = my_sphere.create_shell_mesh(my_orientation_histogram)

# %%
# And, as usual, we will construct a :class:`.SpherePlotter` for each
# histogram we want to visualise.

my_bivariate_plotter = vr.plotting.SpherePlotter(my_bivariate_meshes)
my_bivariate_plotter.produce_plot()
my_bivariate_plotter.show()

my_orientation_plotter = vr.plotting.SpherePlotter(my_orientation_mesh)
my_orientation_plotter.produce_plot()
my_orientation_plotter.show()

# sphinx_gallery_start_ignore
my_bivariate_plotter.close()
my_orientation_plotter.close()
# sphinx_gallery_end_ignore

# %%
# Animations
# ----------
# And now the part that you came here for! Let's see how to create
# animations. We'll see three different types of animations:
#
# 1. Nested sphere animations.
# 2. Rotation animations.
# 3. Custom animations.
#
# The first type of animation only works for nested spherical histograms.
# The other two work for any :class:`.SpherePlotter`.
#
# Before going into the specifics for each, there are some parameters
# common to all types of animations:
#
# ``filename`` :
#       Defines the path to the saved movie file. Must end with ``*.mp4``
#       or with ``.gif``.
# ``quality`` :
#       The desired image quality, between 0 and 10. Ignored if saving as a
#       GIF image.
# ``fps`` :
#       The frame rate, given as the number of frames per second. A higher
#       value results in a faster animation.
#
# Now, let's look at each type of animation in a bit more detail.
#
# Nested Sphere Animations
# ^^^^^^^^^^^^^^^^^^^^^^^^
# These animations should only be generated for nested sphere plots. There
# are a number of different parameters available that control the animation
# itself, as well as the scene. Let's produce some animations to show the
# possibilities.
#
# Nested sphere animations are produced using the
# :meth:`.SpherePlotter.produce_shells_video` method.
#
# Let's first set ``inward_direction=True``.

# sphinx_gallery_start_ignore
# Due to a quirk of sphinx-gallery, we must re-generate the plot in each
# cell.
my_bivariate_plotter = vr.plotting.SpherePlotter(my_bivariate_meshes, off_screen=True)
my_bivariate_plotter.produce_plot()
# sphinx_gallery_end_ignore

my_bivariate_plotter.produce_shells_video(
    "nested_inward.gif",
    fps=8,
    inward_direction=True
)

# sphinx_gallery_start_ignore
my_bivariate_plotter.close()
# sphinx_gallery_end_ignore

# %%
# The shells pass from largest to smallest. Now let's set
# ``inward_direction=False`` to see the difference.

# sphinx_gallery_start_ignore
# Due to a quirk of sphinx-gallery, we must re-generate the plot in each
# cell.
my_bivariate_plotter = vr.plotting.SpherePlotter(my_bivariate_meshes, off_screen=True)
my_bivariate_plotter.produce_plot()
# sphinx_gallery_end_ignore

my_bivariate_plotter.produce_shells_video(
    "nested_outward.gif",
    fps=8,
    inward_direction=False
)

# sphinx_gallery_start_ignore
my_bivariate_plotter.close()
# sphinx_gallery_end_ignore

# %%
# The shells now pass from smallest to largest.
#
# We can also use the ``boomerang`` parameter to have the shells go through
# the reverse order after they have each appeared.

# sphinx_gallery_start_ignore
# Due to a quirk of sphinx-gallery, we must re-generate the plot in each
# cell.
my_bivariate_plotter = vr.plotting.SpherePlotter(my_bivariate_meshes, off_screen=True)
my_bivariate_plotter.produce_plot()
# sphinx_gallery_end_ignore

my_bivariate_plotter.produce_shells_video(
    "nested_boomerang.gif",
    fps=8,
    boomerang=True,
)

# sphinx_gallery_start_ignore
my_bivariate_plotter.close()
# sphinx_gallery_end_ignore

# %%
# Finally, we can also add the shell number to the bottom of the video
# during the animation using the ``add_shell_text`` parameter.

# sphinx_gallery_start_ignore
# Due to a quirk of sphinx-gallery, we must re-generate the plot in each
# cell.
my_bivariate_plotter = vr.plotting.SpherePlotter(my_bivariate_meshes, off_screen=True)
my_bivariate_plotter.produce_plot()
# sphinx_gallery_end_ignore

my_bivariate_plotter.produce_shells_video(
    "nested_shell_text.gif",
    fps=8,
    boomerang=True,
    add_shell_text=True
)

# sphinx_gallery_start_ignore
my_bivariate_plotter.close()
# sphinx_gallery_end_ignore

# %%
# Now we can clearly identify the shells that are being viewed in the
# animation.

# %%
# There's a problem here, though! We only see one side of the sphere. We
# can view other individual positions by using the
# :meth:`.SpherePlotter.rotate_to_view` method and then save videos from
# new perspectives, as well.

# sphinx_gallery_start_ignore
# Due to a quirk of sphinx-gallery, we must re-generate the plot in each
# cell.
my_bivariate_plotter = vr.plotting.SpherePlotter(my_bivariate_meshes, off_screen=True)
my_bivariate_plotter.produce_plot()
# sphinx_gallery_end_ignore

my_bivariate_plotter.rotate_to_view(phi=160, theta=300)

my_bivariate_plotter.produce_shells_video(
    "nested_rotated.gif",
    fps=8,
    boomerang=True,
    add_shell_text=True
)

# sphinx_gallery_start_ignore
my_bivariate_plotter.close()
# sphinx_gallery_end_ignore

# %%
# Now we can see the different frequencies across the shells from a
# different perspective. But what about flying across the sphere?

# %%
# Rotation Animations
# ^^^^^^^^^^^^^^^^^^^
# Rotation animations can be used for any spherical or nested spherical
# histogram plot. In a rotation animation, the camera orbits around the
# spherical histogram, saving frames to produce a video.
#
# Nested sphere animations are produced using the
# :meth:`.SpherePlotter.produce_rotating_video` method.
#
# This method has parameters that control the number of frames generated.
# A higher number of frames results in a smoother but larger video.
#
# Let's see some examples using ``my_orientation_plotter``, which shows the
# marginal orientation distribution for our data.
#
# First, we'll use a lower frame rate of 10 frames per second and produce
# 36 frames.

# sphinx_gallery_start_ignore
# Due to a quirk of sphinx-gallery, we must re-generate the plot in each
# cell.
my_orientation_plotter = vr.plotting.SpherePlotter(my_orientation_mesh, off_screen=True)
my_orientation_plotter.produce_plot()
# sphinx_gallery_end_ignore

my_orientation_plotter.rotate_to_view(phi=160, theta=300)

my_orientation_plotter.produce_rotating_video(
    "orientation_10_fps.gif",
    fps=10,
    number_of_frames=36,
)

# sphinx_gallery_start_ignore
my_orientation_plotter.close()
# sphinx_gallery_end_ignore

# %%
# Now, let's create 72 frames and have a video that is 20 fps.

# sphinx_gallery_start_ignore
# Due to a quirk of sphinx-gallery, we must re-generate the plot in each
# cell.
my_orientation_plotter = vr.plotting.SpherePlotter(my_orientation_mesh, off_screen=True)
my_orientation_plotter.produce_plot()
# sphinx_gallery_end_ignore

my_orientation_plotter.rotate_to_view(phi=160, theta=300)

my_orientation_plotter.produce_rotating_video(
    "orientation_20_fps.gif",
    fps=20,
    number_of_frames=72,
)

# sphinx_gallery_start_ignore
my_orientation_plotter.close()
# sphinx_gallery_end_ignore

# %%
# Notice the difference? We can also control the zoom and add a vertical
# shift using some of the other parameters.

# sphinx_gallery_start_ignore
# Due to a quirk of sphinx-gallery, we must re-generate the plot in each
# cell.
my_orientation_plotter = vr.plotting.SpherePlotter(my_orientation_mesh, off_screen=True)
my_orientation_plotter.produce_plot()
# sphinx_gallery_end_ignore

my_orientation_plotter.rotate_to_view(phi=160, theta=300)

my_orientation_plotter.produce_rotating_video(
    "orientation_zoom_shift.gif",
    fps=10,
    number_of_frames=36,
    zoom_factor=1.5,
    vertical_shift=0.5
)

# sphinx_gallery_start_ignore
my_orientation_plotter.close()
# sphinx_gallery_end_ignore

# %%
# So, we can now make videos orbiting around the sphere. Except, there's a
# small limit. We can't orbit around arbitrary axes. To do this, we need to
# use a bit of a different approach...

# %%
# Custom Animations
# ^^^^^^^^^^^^^^^^^
# In addition to using the pre-made animations, :class:`.SpherePlotter`
# offers functions to construct custom animations.
#
# For example, let's say we want to create an animation going from the top
# of the sphere to the bottom along a single meridian. We can't do this
# with the built-in functions, but we can with a custom animation.
#
# The key methods involved are:
#
# * :meth:`.SpherePlotter.open_movie_file` - Start a new movie or GIF file.
# * :meth:`.SpherePlotter.write_frame` - Add a new frame to the movie.
# * :meth:`.SpherePlotter.close_movie` - Finish the movie.
#
# .. warning::
#    The movie writer does **not** interpolate between frames. If you want
#    A smooth animation, you need to make a lot of frames.
#
# Let's combine these methods with :meth:`.SpherePlotter.rotate_to_view` to
# create a custom sphere rotating animation.

# sphinx_gallery_start_ignore
# Due to a quirk of sphinx-gallery, we must re-generate the plot in each
# cell.
my_orientation_plotter = vr.plotting.SpherePlotter(my_orientation_mesh, off_screen=True)
my_orientation_plotter.produce_plot()
# sphinx_gallery_end_ignore

phi_values = range(0, 181, 10)
theta_value = 45

# Hide the sliders
my_orientation_plotter.hide_sliders()

my_orientation_plotter.open_movie_file("custom_anim.gif", fps=10)

for phi in phi_values:
    # Get the camera into the new position.
    my_orientation_plotter.rotate_to_view(phi, theta_value)

    # Write the new frame.
    my_orientation_plotter.write_frame()

# Let's boomerang it manually
for phi in reversed(phi_values):
    # Get the camera into the new position.
    my_orientation_plotter.rotate_to_view(phi, theta_value)

    # Write the new frame.
    my_orientation_plotter.write_frame()

my_orientation_plotter.close_movie()

# Re-show the sliders
my_orientation_plotter.show_sliders()

# sphinx_gallery_start_ignore
my_orientation_plotter.close()
# sphinx_gallery_end_ignore

# %%
# By combining these methods, we've now created a new animation.
#
# There are many other possibilities for custom animations involving custom
# paths and activating shells in the nested histogram case.
#

# %%
# Summary
# -------
# That brings us to the end of this example. Good luck making animations
# with VectoRose!
