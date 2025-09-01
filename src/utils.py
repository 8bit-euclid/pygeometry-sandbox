"""
Reusable utilities extracted for Jupyter notebook tutorials.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Set up plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)


def plot_triangulation(vertices, triangles, edges_in=None, title="Triangulation",
                       show_vertices=True, show_edges=True, show_triangles=True, input_vertices=None):
    """
    Visualize triangulation results with customizable display options.
    Shows side-by-side comparison when input_vertices is provided.
    """
    # Determine if we should show side-by-side comparison
    show_comparison = input_vertices is not None and len(
        input_vertices) != len(vertices)

    if show_comparison:
        # Side-by-side comparison layout
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Convert input_vertices to numpy array if it isn't already
        input_vertices = np.array(input_vertices)

        # Left plot: Original input (point cloud or input geometry)
        ax1.scatter(input_vertices[:, 0], input_vertices[:,
                    1], c='red', s=50, alpha=0.7, zorder=5)

        # Highlight input boundary edges if provided
        if edges_in is not None:
            boundary_edges = []
            for edge in edges_in:
                boundary_edges.append(
                    [input_vertices[edge[0]], input_vertices[edge[1]]])
            boundary_collection = LineCollection(
                boundary_edges, colors='red', linewidths=2)
            ax1.add_collection(boundary_collection)

        ax1.set_title(f'Input Geometry\\n{len(input_vertices)} points')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)

        # Right plot: Triangulated result
        ax = ax2  # Use ax2 for the triangulation plot
        ax.set_title(f'Triangulated Result\\n{len(triangles)} triangles')
    else:
        # Single plot layout (original behavior)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title(title)

    # Plot triangles on the main/right axis
    if show_triangles and len(triangles) > 0:
        for tri in triangles:
            triangle = plt.Polygon(vertices[tri], alpha=0.3,
                                   facecolor='lightblue', edgecolor='blue', linewidth=0.5)
            ax.add_patch(triangle)

    # Plot triangle edges
    if show_edges and len(triangles) > 0:
        edges = []
        for tri in triangles:
            for i in range(3):
                edge = [vertices[tri[i]], vertices[tri[(i+1) % 3]]]
                edges.append(edge)

        line_collection = LineCollection(
            edges, colors='blue', linewidths=0.5, alpha=0.7)
        ax.add_collection(line_collection)

    # Plot vertices
    if show_vertices:
        ax.scatter(vertices[:, 0], vertices[:, 1], c='red', s=30, zorder=5)

        # Label original vertices if we have input edges (only in single plot mode)
        if edges_in is not None and not show_comparison:
            unique_indices = np.unique(edges_in.flatten())
            verts_for_labels = input_vertices if input_vertices is not None else vertices
            for i in unique_indices:
                if i < len(verts_for_labels):  # Safety check
                    ax.annotate(f'{i}', (verts_for_labels[i, 0], verts_for_labels[i, 1]),
                                xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Set overall title
    if show_comparison:
        plt.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()


def create_circle_points(center, radius, num_points):
    """Create points on a circle."""
    angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    points = np.column_stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles)
    ])
    return points


def create_star(center, outer_radius, inner_radius, num_points=5):
    """Create a star shape with alternating outer and inner radii."""
    angles = np.linspace(0, 2*np.pi, 2*num_points, endpoint=False)
    radii = np.tile([outer_radius, inner_radius], num_points)

    points = np.column_stack([
        center[0] + radii * np.cos(angles),
        center[1] + radii * np.sin(angles)
    ])
    return points


def generate_point_cloud(n_points=50, shape='random', noise_level=0.1):
    """
    Generate different types of point clouds for triangulation.

    Args:
        n_points: Number of points to generate
        shape: Type of point cloud ('random', 'circle', 'grid', 'spiral')
        noise_level: Amount of random noise to add

    Returns:
        numpy array of 2D points
    """
    np.random.seed(42)  # For reproducible results

    if shape == 'random':
        # Completely random points in a unit square
        points = np.random.rand(n_points, 2) * 4.0

    elif shape == 'circle':
        # Points roughly arranged in a circle with noise
        angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        radius = 1.5 + noise_level * np.random.randn(n_points)
        points = np.column_stack([
            2.0 + radius * np.cos(angles),
            2.0 + radius * np.sin(angles)
        ])

    elif shape == 'grid':
        # Grid-like arrangement with noise
        grid_size = int(np.sqrt(n_points))
        x = np.linspace(0, 3, grid_size)
        y = np.linspace(0, 3, grid_size)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel()])[:n_points]
        # Add noise
        points += noise_level * np.random.randn(n_points, 2)

    elif shape == 'spiral':
        # Spiral pattern
        t = np.linspace(0, 4*np.pi, n_points)
        r = 0.1 + 0.3 * t
        points = np.column_stack([
            2.0 + r * np.cos(t),
            2.0 + r * np.sin(t)
        ])
        # Add noise
        points += noise_level * np.random.randn(n_points, 2)

    return points
