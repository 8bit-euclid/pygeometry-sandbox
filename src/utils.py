"""
Reusable utilities extracted for Jupyter notebook tutorials.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Set up plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)


def vertex_array(triangulation):
    """Convert triangulation vertices to NumPy array."""
    return np.array([[v.x, v.y] for v in triangulation.vertices])


def triangle_array(triangulation):
    """Convert triangulation triangles to NumPy array."""
    return np.array([t.vertices for t in triangulation.triangles])


def plot_triangulation(vertices, triangles, edges_in=None, title="Triangulation",
                       show_vertices=True, show_edges=True, show_triangles=True, input_vertices=None,
                       show_vertex_labels=True, show_triangle_labels=True, show_constraint_edges=True, show_legend=True,
                       vertex_label_offset=(4, 2), interactive=True):
    """
    Visualize triangulation results with customizable display options.
    Shows side-by-side comparison when input_vertices is provided.

    Args:
        vertices: Array of vertex coordinates (N, 2)
        triangles: Array of triangle indices (M, 3)
        edges_in: Input constraint edges (K, 2) (optional)
        title: Plot title
        show_vertices: Whether to show vertex points
        show_edges: Whether to show triangle edges
        show_triangles: Whether to show triangle fills
        input_vertices: Original input vertices for comparison (N, 2) (optional)
        show_vertex_labels: Whether to show vertex index labels
        show_triangle_labels: Whether to show triangle index labels
        show_constraint_edges: Whether to highlight constraint edges
        show_legend: Whether to show legend when constraints are present
        vertex_label_offset: Tuple (x, y) for vertex label offset in points (default: (4, 2))
        interactive: Enable interactive zooming and panning (default: True)
    """
    # Set up appropriate backend for display
    import matplotlib
    current_backend = matplotlib.get_backend()

    # Check if we're in a Jupyter environment
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        in_jupyter = ipython is not None
    except ImportError:
        in_jupyter = False
        ipython = None

    if in_jupyter:
        # In Jupyter, choose backend based on interactivity needs
        if interactive:
            # For interactive plots, try notebook backends first
            if current_backend not in ['notebook', 'nbagg', 'widget']:
                for backend in ['notebook', 'nbagg']:
                    try:
                        matplotlib.use(backend)
                        if ipython and hasattr(ipython, 'run_line_magic'):
                            try:
                                ipython.run_line_magic('matplotlib', backend)
                            except:
                                pass
                        break
                    except (ImportError, ValueError):
                        continue
                else:
                    # Fallback to inline if interactive backends fail
                    try:
                        matplotlib.use('inline')
                        if ipython and hasattr(ipython, 'run_line_magic'):
                            ipython.run_line_magic('matplotlib', 'inline')
                    except:
                        pass
        else:
            # For non-interactive plots, use inline backend
            if current_backend not in ['inline']:
                try:
                    matplotlib.use('inline')
                    if ipython and hasattr(ipython, 'run_line_magic'):
                        ipython.run_line_magic('matplotlib', 'inline')
                except:
                    pass
    elif interactive:
        # For non-Jupyter interactive mode, try GUI backends
        if current_backend not in ['qt5agg', 'tkagg', 'notebook', 'nbagg']:
            for backend in ['qt5agg', 'tkagg']:
                try:
                    matplotlib.use(backend)
                    break
                except (ImportError, ValueError):
                    continue

    # Determine if we should show side-by-side comparison
    show_comparison = input_vertices is not None and len(
        input_vertices) != len(vertices)

    if show_comparison:
        # Side-by-side comparison layout
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Convert input_vertices to numpy array if it isn't already
        input_vertices = np.array(input_vertices)

        # Left plot: Original input (point cloud or input geometry)
        ax1.plot(input_vertices[:, 0], input_vertices[:, 1], 'o', markersize=8,
                 markerfacecolor='black', markeredgecolor='black', markeredgewidth=0,
                 alpha=0.8, zorder=5)

        # Highlight input boundary edges if provided
        if edges_in is not None and show_constraint_edges:
            for edge in edges_in:
                v1, v2 = input_vertices[edge[0]], input_vertices[edge[1]]
                ax1.plot([v1[0], v2[0]], [v1[1], v2[1]],
                         'r-', linewidth=3, alpha=0.8)

        ax1.set_title(
            f'Input Geometry\\n{len(input_vertices)} points', fontweight='bold')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')

        # Right plot: Triangulated result
        ax = ax2  # Use ax2 for the triangulation plot
        ax.set_title(
            f'Triangulated Result\\n{len(triangles)} triangles', fontweight='bold')
    else:
        # Single plot layout (original behavior)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title(title, fontsize=14, fontweight='bold')

    # Plot triangulation using triplot for cleaner rendering
    if (show_triangles or show_edges) and len(triangles) > 0:
        if show_triangles:
            # Plot filled triangles with transparency
            ax.triplot(vertices[:, 0], vertices[:, 1], triangles, 'b-',
                       alpha=0.5, linewidth=1)
            # Add transparent fill
            for tri in triangles:
                verts = [vertices[i] for i in tri]
                triangle = plt.Polygon(verts, alpha=0.1, facecolor='red',
                                       edgecolor='none')
                ax.add_patch(triangle)
        elif show_edges:
            # Just show edges without fill
            ax.triplot(vertices[:, 0], vertices[:, 1], triangles, 'b-',
                       alpha=0.7, linewidth=0.8)

    # Highlight constraint edges if provided
    if edges_in is not None and show_constraint_edges:
        constraint_vertices = input_vertices if input_vertices is not None else vertices
        for edge in edges_in:
            if edge[0] < len(constraint_vertices) and edge[1] < len(constraint_vertices):
                v1, v2 = constraint_vertices[edge[0]
                                             ], constraint_vertices[edge[1]]
                ax.plot([v1[0], v2[0]], [v1[1], v2[1]],
                        'r-', linewidth=3, alpha=0.8)

    # Plot vertices with enhanced styling
    if show_vertices:
        ax.plot(vertices[:, 0], vertices[:, 1], 'o', markersize=8,
                markerfacecolor='black', markeredgecolor='black', markeredgewidth=0, zorder=5)

        # Add vertex labels with enhanced styling
        if show_vertex_labels:
            if edges_in is not None and not show_comparison:
                # Only label constraint vertices in single plot mode
                unique_indices = np.unique(edges_in.flatten())
                verts_for_labels = input_vertices if input_vertices is not None else vertices
                for i in unique_indices:
                    if i < len(verts_for_labels):  # Safety check
                        ax.annotate(str(i), (verts_for_labels[i, 0], verts_for_labels[i, 1]),
                                    xytext=vertex_label_offset, textcoords='offset points',
                                    fontsize=10, color='darkblue', weight='bold')
            elif not show_comparison:
                # Label all vertices if no constraints or in single mode
                for i, vertex in enumerate(vertices):
                    ax.annotate(str(i), (vertex[0], vertex[1]), xytext=vertex_label_offset,
                                textcoords='offset points', fontsize=9,
                                color='darkblue', weight='bold')

    # Add triangle labels with enhanced styling
    if show_triangle_labels and len(triangles) > 0 and not show_comparison:
        for i, tri in enumerate(triangles):
            # Calculate centroid of triangle
            centroid = np.mean([vertices[j] for j in tri], axis=0)
            ax.annotate(str(i), (centroid[0], centroid[1]), xytext=(0, 0),
                        textcoords='offset points', fontsize=9,
                        color='darkgreen', weight='bold', ha='center', va='center')

    # Set axis properties
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Add legend if there are constraints and legend is requested
    if edges_in is not None and show_legend and show_constraint_edges:
        legend_elements = []
        if show_triangles or show_edges:
            legend_elements.append(plt.Line2D([0], [0], color='blue', alpha=0.5,
                                              label='Triangulation edges'))
        if show_constraint_edges:
            legend_elements.append(plt.Line2D([0], [0], color='red', linewidth=3,
                                              alpha=0.8, label='Constraint edges'))
        if show_vertices:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                              markerfacecolor='black', markeredgecolor='black', markeredgewidth=0,
                                              markersize=8, label='Vertices'))
        if legend_elements:
            ax.legend(handles=legend_elements)

    # Set overall title for comparison plots
    if show_comparison:
        plt.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Check if we're in Jupyter for proper display handling
    try:
        from IPython import get_ipython
        in_jupyter = get_ipython() is not None
    except ImportError:
        in_jupyter = False

    if in_jupyter:
        # In Jupyter, use plt.show() to display inline
        plt.show()
        # Don't return figure to avoid <Figure> text display
        return None
    else:
        # In regular Python, show the plot
        plt.show()
        # Return figure if interactive mode is requested
        if interactive:
            return fig
        return None


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
