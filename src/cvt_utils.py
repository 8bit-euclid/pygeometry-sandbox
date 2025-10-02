import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from scipy.integrate import quad, fixed_quad, quad_vec
from functools import lru_cache
from joblib import Parallel, delayed
# import cupy as cp

# Set higher limits for the HTML writer
# Default is 20 MB, increase as needed
plt.rcParams['animation.embed_limit'] = 500  # MB


class SpaceIterMesh:
    def __init__(self, x_min: float, x_max: float, n_cells: int, n_iters: int):
        self.x_min = x_min
        self.x_max = x_max
        self.n_cells = n_cells
        self.n_iters = n_iters
        self.cell_size: Callable[[float | np.ndarray],
                                 float | np.ndarray] | None = None
        self.seed_matrix = np.zeros((n_iters, n_cells))
        self.bound_matrix = np.zeros((n_iters, n_cells + 1))

    @property
    def length(self):
        return self.x_max - self.x_min

    def cell_density(self, x):
        sz = self.cell_size(x)  # type: ignore
        return self.length / sz  # type: ignore

    def normalised_cell_density(self, x):
        return self.cell_density(x) / self.total_mass() * self.n_cells

    @lru_cache(maxsize=None)
    def total_mass(self):
        def integrand(x): return self.cell_density(x)
        mass, err = quad(integrand, self.x_min, self.x_max)
        assert err < 1.0e-10, "Integration error is too large"
        return mass

    def cell_centroid(self, i_cell: int, iter: int) -> float:
        bounds = self.bound_matrix[iter, :]
        x_l = bounds[i_cell]
        x_r = bounds[i_cell + 1]

        def integrand_nu(x): return x * self.cell_density(x)
        def integrand_de(x): return self.cell_density(x)

        nu, e_nu = fixed_quad(integrand_nu, x_l, x_r, n=5)
        de, e_de = fixed_quad(integrand_de, x_l, x_r, n=5)

        # err = (de*e_nu + nu*e_de) / (de*(de + e_de))
        return nu / de

    def cell_energy(self, i_cell: int, iter: int, order: int = 5) -> float:
        x_i = self.seed_matrix[iter, i_cell]
        x_l = self.bound_matrix[iter, i_cell]
        x_r = self.bound_matrix[iter, i_cell + 1]
        def integrand(x): return self.cell_density(x) * (x - x_i) ** 2
        energy, err = fixed_quad(integrand, x_l, x_r, n=order)
        return energy

    def total_energy(self) -> np.ndarray:
        total_energy = np.zeros(self.n_iters)
        for iter in range(self.n_iters):
            for ic in range(self.n_cells):
                total_energy[iter] += self.cell_energy(ic, iter)
        return total_energy

    def set_initial_conditions(self):
        self.seed_matrix[0, :] = np.sort(
            np.random.uniform(self.x_min, self.x_max, self.n_cells))

    def set_boundary_conditions(self):
        self.bound_matrix[:, 0] = self.x_min   # First column
        self.bound_matrix[:, -1] = self.x_max  # Last column

    def update_cell_bounds(self, iter: int):
        seeds = self.seed_matrix[iter, :]
        bounds = self.bound_matrix[iter, :]
        bounds[1:-1] = 0.5 * (seeds[:-1] + seeds[1:])  # Interior bounds

    def update_cell_seeds(self, iter: int):
        assert iter < self.n_iters - 1, "Cannot update beyond the last iteration"
        seeds_new = self.seed_matrix[iter + 1, :]

        # Centroid calculation (not worth parallelizing)
        for ic in range(self.n_cells):
            seeds_new[ic] = self.cell_centroid(ic, iter)

    def update_cell_seeds_aitken(self, iter: int, order: int = 2) -> bool:
        assert iter < self.n_iters - 1, "Cannot update beyond the last iteration"
        assert iter > order, "Not enough iterations to apply Aitken's acceleration"

        # for ic in range(self.n_cells):
        #     x_0 = self.seed_matrix[iter - 2, ic]
        #     x_1 = self.seed_matrix[iter - 1, ic]
        #     x_2 = self.seed_matrix[iter, ic]

        #     dx = x_1 - x_0
        #     # dx = x_2 - x_1
        #     d2x = x_2 - 2*x_1 + x_0

        #     if np.abs(d2x) > 1.0e-12:
        #         x_new = x_0 - (dx**2) / d2x
        #         # x_new = x_2 - (dx**2) / d2x
        #         self.seed_matrix[iter + 1, ic] = x_new
        #     else:
        #         self.seed_matrix[iter + 1, ic] = self.cell_centroid(ic, iter)

        x_0 = self.seed_matrix[iter - 2, :]
        x_1 = self.seed_matrix[iter - 1, :]
        x_2 = self.seed_matrix[iter, :]

        dx = x_1 - x_0
        d2x = x_2 - 2*x_1 + x_0

        x_new = self.seed_matrix[iter + 1, :]
        x_new[:] = x_0 - (dx**2) / d2x

        if np.any(np.abs(d2x) > 1.0e-12):
            return False

        return True
