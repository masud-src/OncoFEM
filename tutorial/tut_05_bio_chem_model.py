import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def gaussian_distribution(domain, min_value, max_value, inner_potato_radius, outer_potato_radius):
    # Calculate the centroid of the inner potato region
    inner_centroid = np.array([np.mean(coord) for coord in np.nonzero(domain)])

    # Calculate the distance of each point from the inner centroid
    inner_distances = np.linalg.norm(np.indices(domain.shape) - inner_centroid[:, np.newaxis, np.newaxis, np.newaxis], axis=0)

    # Calculate the Gaussian distribution for the inner potato region
    inner_gaussian = max_value * np.exp(-0.5 * (inner_distances / inner_potato_radius) ** 2)

    # Calculate the distance of each point from the outer potato region
    outer_distances = np.linalg.norm(np.indices(domain.shape), axis=0) - outer_potato_radius

    # Calculate the Gaussian distribution for the outer potato region
    outer_gaussian = min_value + (max_value - min_value) * np.exp(-0.5 * (outer_distances / outer_potato_radius) ** 2)

    # Combine the inner and outer Gaussian distributions
    gaussian = np.where(domain == 1, inner_gaussian, outer_gaussian)

    return gaussian


# Define the total domain size
total_size = 100

# Create the total domain as a numpy ndarray
domain = np.zeros((total_size, total_size, total_size))

# Define the inner and outer potato regions
inner_potato_radius = 40
outer_potato_radius = 60
potato_center = (total_size - 1) // 2
indices = np.indices((total_size, total_size, total_size))
distances = np.sqrt(np.sum((indices - potato_center) ** 2, axis=0))
domain[distances <= outer_potato_radius] = 1
domain[distances <= inner_potato_radius] = 0

# Define the minimum and maximum values
min_value = 1.0
max_value = 100.0

# Compute the Gaussian distribution
gaussian = gaussian_distribution(domain, min_value, max_value, inner_potato_radius, outer_potato_radius)

# Visualize the Gaussian distribution
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create coordinate arrays
x, y, z = np.indices(domain.shape)

# Plot the potato-shaped domain
ax.scatter(x[domain == 1], y[domain == 1], z[domain == 1], color='gray', alpha=0.2)

# Plot the Gaussian distribution
ax.scatter(x.flatten(), y.flatten(), z.flatten(), c=gaussian.flatten(), cmap='viridis', alpha=0.5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

x = 10