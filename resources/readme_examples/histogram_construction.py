"""Example usage for histogram construction."""

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
my_sphere_plotter.produce_plot(add_sliders=False)
my_sphere_plotter.produce_shells_video(
    "histogram_construction_shells.gif",
    fps=3,
    inward_direction=True,
    boomerang=True,
    add_shell_text=True
)
