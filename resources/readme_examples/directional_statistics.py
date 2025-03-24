"""Example usage for directional statistics."""

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
