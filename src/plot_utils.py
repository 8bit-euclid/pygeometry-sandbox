import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from cvt_utils import SpaceIterMesh


def display_plots(plot_configs: list[dict], title: str, xlabel: str, ylabel: str, show_grid: bool = True, ylog=False, xlog=False, xlims=None, ylims=None):
    """Display a list of plots in a Jupyter notebook."""

    plt.figure(figsize=(10, 6))
    for config in plot_configs:
        plt.plot(*config["args"], **config["kwargs"])
    plt.title(title)  # type: ignore
    plt.xlabel(xlabel)  # type: ignore
    plt.ylabel(ylabel)  # type: ignore
    if xlims is not None:
        plt.xlim(xlims)
    if ylims is not None:
        plt.ylim(ylims)
    if xlog:
        plt.xscale("log")  # type: ignore
    if ylog:
        plt.yscale("log")  # type: ignore
    plt.legend()
    plt.grid(show_grid, alpha=0.3)  # type: ignore
    plt.show()


def plot_cell_size_and_density(mesh: SpaceIterMesh):
    """Plot cell size and density (static plot)."""
    assert mesh.cell_size is not None, "Cell size function not set"
    x_min = mesh.x_min
    x_max = mesh.x_max

    x = np.linspace(x_min, x_max, 1000)
    plt.figure(figsize=(10, 6))
    plt.plot(x, mesh.cell_size(x))  # type: ignore
    plt.title("Cell Size")  # type: ignore
    plt.xlabel("x")  # type: ignore
    plt.ylabel("Size")  # type: ignore
    plt.xlim(x_min, x_max)
    plt.grid(True, alpha=0.3)  # type: ignore
    plt.show()

    # Plot cell density
    x = np.linspace(x_min, x_max, 1000)
    plt.figure(figsize=(10, 6))
    plt.plot(x, mesh.cell_density(x))  # type: ignore
    plt.title("Cell Density")  # type: ignore
    plt.xlabel("x")  # type: ignore
    plt.ylabel("Density")  # type: ignore
    plt.xlim(x_min, x_max)
    plt.show()


def plot_computed_cell_size(mesh: SpaceIterMesh):
    """Plot cell size at the centroids (static plot)."""
    assert mesh.cell_size is not None, "Cell size function not set"
    x_min = mesh.x_min
    x_max = mesh.x_max
    n_cells = mesh.n_cells

    # Get cell sizes at centroids
    centroids = np.array([mesh.cell_centroid(ic, mesh.n_iters - 1)
                          for ic in range(n_cells)])
    bounds = mesh.bound_matrix[mesh.n_iters - 1, :]
    cell_sizes = bounds[1:] - bounds[:-1]

    # Plot cell sizes
    plt.figure(figsize=(10, 6))
    plt.plot(centroids, cell_sizes, '-', markersize=1, markerfacecolor='white',  # type: ignore
             markeredgecolor='white', markeredgewidth=0, alpha=0.8)
    # pyright: ignore[reportCallIssue]
    plt.title(f'1D Cells Plot - Iter: {mesh.n_iters - 1}')  # type: ignore
    plt.xlim(x_min, x_max)
    plt.xlabel('X')  # type: ignore
    plt.ylabel('Y')  # type: ignore
    plt.show()


def plot_seeds_and_cells(mesh: SpaceIterMesh, iter: int):
    """Plot cells for a single iteration (static plot)."""
    seeds = mesh.seed_matrix[iter, :]
    bounds = mesh.bound_matrix[iter, :]

    assert len(bounds) == len(seeds) + \
        1, "cell_bounds must have one more entry than cell_seeds"
    assert np.all(seeds[:-1] <= seeds[1:]), "Ensure that cell_seeds is sorted"
    assert np.all(bounds[:-1] <= bounds[1:]
                  ), "Ensure that cell_bounds is sorted"

    x_min = bounds[0]
    x_max = bounds[-1]

    # Plot cell seeds
    plt.figure(figsize=(10, 0.5))
    plt.plot(seeds, np.zeros_like(seeds), 'o', markersize=1, markerfacecolor='white',  # type: ignore
             markeredgecolor='white', markeredgewidth=0, alpha=0.8)
    # pyright: ignore[reportCallIssue]
    plt.title(f'1D Cells Plot - Iter: {iter}')  # type: ignore
    plt.xlim(x_min, x_max)
    plt.ylim(-1, 1)
    plt.xlabel('X')  # type: ignore
    plt.ylabel('Y')  # type: ignore

    # Plot interior cell boundaries
    for bound in bounds[1:-1]:
        plt.axvline(x=bound, color='red', linestyle='-',  # type: ignore
                    linewidth=1, alpha=0.5)  # type: ignore

    plt.show()


def plot_bound_paths(mesh: SpaceIterMesh, start_iter: int = 0, end_iter: int | None = None):
    """Plot cell boundary paths over iterations (static plot)."""
    # Get the iteration range
    if end_iter is None:
        end_iter = mesh.n_iters - 1
    iterations = np.arange(start_iter, end_iter + 1)
    n_bounds = mesh.n_cells + 1

    # Create figure with same width as static plot
    plt.figure(figsize=(10, 8))

    # Plot each seed path as a separate line
    for bound_idx in range(n_bounds):
        # Extract seed positions for this seed across all iterations
        bound_positions = mesh.bound_matrix[start_iter: end_iter + 1, bound_idx]

        # Plot with negative iterations (so iteration 0 is at top) but use white color
        plt.plot(bound_positions, -iterations, "-",  # type: ignore
                 color="red", linewidth=0.6, alpha=0.8)

    # Set labels and title
    plt.xlabel("Cell Boundary Positions")  # type: ignore
    plt.ylabel("Iteration")  # type: ignore
    plt.title(f"Paths Over Iters: [{start_iter}, {end_iter}]")  # type: ignore

    # Set axis limits
    plt.xlim(mesh.x_min, mesh.x_max)
    plt.ylim(-end_iter, -start_iter + 1)

    # Customize y-axis to show positive iteration numbers
    y_ticks = np.arange(-end_iter, -start_iter + 1,
                        max(1, (end_iter - start_iter) // 10))
    # Convert back to positive for display
    y_labels = [-tick for tick in y_ticks]
    plt.yticks(y_ticks, y_labels)

    # Show the plot
    plt.show()


def animate_cells(
    mesh: SpaceIterMesh,
    start_iter: int = 0,
    end_iter: int | None = None,
    num_frames: int = 100,
    interval: int = 50,  # delay between frames in ms
):
    """Create an animated plot showing cell evolution over iterations."""
    if end_iter is None:
        end_iter = mesh.n_iters - 1
    else:
        assert end_iter >= start_iter and end_iter < mesh.n_iters, \
            "end_iter must be in [start_iter, n_iters)"

    n_iters = end_iter - start_iter + 1
    assert 1 <= num_frames <= n_iters, "num_frames must be in [1, n_iters]"
    iters_per_frame = n_iters / num_frames

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 0.25), dpi=300)

    # Get global bounds for consistent axis limits
    x_min = mesh.bound_matrix[:, 0].min()
    x_max = mesh.bound_matrix[:, -1].max()

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Initialize empty plot elements
    (seeds_plot,) = ax.plot(
        [],
        [],
        "o",
        markersize=1.5,
        markerfacecolor="white",
        markeredgecolor="white",
        markeredgewidth=0,
        alpha=1,
    )
    boundary_lines = []

    def animate(frame):
        delta_iter = int(frame * iters_per_frame)
        iter = min(start_iter+delta_iter, mesh.n_iters-1)
        seeds = mesh.seed_matrix[iter, :]
        bounds = mesh.bound_matrix[iter, :]

        # Clear previous boundary lines
        for line in boundary_lines:
            line.remove()
        boundary_lines.clear()

        # Update seeds
        seeds_plot.set_data(seeds, np.zeros_like(seeds))

        # Add boundary lines
        for bound in bounds[1:-1]:
            line = ax.axvline(
                x=bound, color="red", linestyle="-", linewidth=1, alpha=0.5
            )
            boundary_lines.append(line)

        # Update title
        ax.set_title(f"1D Cells Plot - Iteration {iter}")

        return [seeds_plot] + boundary_lines

    # Create animation
    anim = FuncAnimation(
        fig, animate, frames=num_frames, interval=interval, blit=False, repeat=True
    )

    # Close the figure to prevent it from being displayed as a static plot
    plt.close(fig)

    return anim


def display_animation(mesh: SpaceIterMesh, duration: float = 3.0, fps: int = 60) -> HTML:
    n_frames = int(fps * duration)
    total_time = duration * 1000  # convert to ms
    interval = int(total_time / n_frames)
    anim = animate_cells(mesh, num_frames=n_frames, interval=interval)

    # Print animation info
    print(f"Duration: {duration}s")
    print(f"fps: {fps}")
    print(f"# of frames: {n_frames}")
    print(f"Interval: {interval}ms")

    return HTML(anim.to_jshtml())


def plot_errors(min_errors, max_errors, rms_errors, logscale=True):
    plt.figure(figsize=(10, 6))
    plt.plot(min_errors, label="Min Error",  # type: ignore
             linewidth=1.2, color="green")
    plt.plot(max_errors, label="Max Error",  # type: ignore
             linewidth=1, color="red")
    plt.plot(rms_errors, label="RMS Error",  # type: ignore
             linewidth=1, color="cyan")
    plt.title("Convergence of Cell Boundaries")  # type: ignore
    plt.xlabel("Iteration")  # type: ignore
    plt.ylabel(f"Error {'(log scale)' if logscale else ''}")  # type: ignore
    if logscale:
        plt.yscale("log")  # type: ignore
    plt.legend()
    plt.grid(True, alpha=0.3)  # type: ignore
    plt.show()

# Plot log(error_{i+1}) vs log(error_i)


def plot_log_errors(min_errors, max_errors, rms_errors):
    plt.figure(figsize=(8, 8.3))
    plt.plot(
        np.log(max_errors[:-1]),  # type: ignore
        np.log(max_errors[1:]),
        "-",
        label="Max Error",
        linewidth=1,
        color="red",
    )
    plt.plot(
        np.log(rms_errors[:-1]),  # type: ignore
        np.log(rms_errors[1:]),
        "-",
        label="RMS Error",
        linewidth=1,
        color="cyan",
    )
    plt.title("Convergence Rate of Cell Boundaries")  # type: ignore
    plt.xlabel(r"$\log(\epsilon_i)$")  # type: ignore
    plt.ylabel(r"$\log(\epsilon_{i+1})$")  # type: ignore
    plt.legend()
    plt.grid(True, alpha=0.3)  # type: ignore
    plt.show()
