import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from scipy.integrate import quad, fixed_quad
from functools import lru_cache
from itertools import combinations
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
        self.cell_size: Callable[[np.ndarray], np.ndarray] | None = None
        self.cell_masses = np.zeros(n_cells)
        self.cell_moments = np.zeros(n_cells)
        self.cell_centroids = np.zeros(n_cells)
        self.seed_matrix = np.zeros((n_iters, n_cells))
        self.bound_matrix = np.zeros((n_iters, n_cells + 1))

    # Domain properties ---------------------------------------------------------------------
    @property
    def length(self):
        return self.x_max - self.x_min

    def density(self, x):
        sz = self.cell_size(x)  # type: ignore
        return self.length / sz  # type: ignore

    def normalised_density(self, x):
        return self.density(x) / self.total_mass() * self.n_cells

    @lru_cache(maxsize=None)
    def total_mass(self):
        def integrand(x): return self.density(x)
        mass, err = quad(integrand, self.x_min, self.x_max)
        assert err < 1.0e-10, "Integration error is too large"
        return mass

    # Cell properties -----------------------------------------------------------------------
    def cell_mass(self, i_cell: int, iter: int) -> float:
        bounds = self.bound_matrix[iter, :]
        x_l = bounds[i_cell]
        x_r = bounds[i_cell + 1]
        def integrand(x): return self.density(x)
        mass, err = fixed_quad(integrand, x_l, x_r)
        return mass

    def cell_moment(self, i_cell: int, iter: int) -> float:
        bounds = self.bound_matrix[iter, :]
        x_l = bounds[i_cell]
        x_r = bounds[i_cell + 1]
        def integrand(x): return x * self.density(x)
        moment, err = fixed_quad(integrand, x_l, x_r)
        return moment

    def cell_centroid(self, i_cell: int, iter: int) -> float:
        mass = self.cell_masses[i_cell]
        moment = self.cell_moments[i_cell]
        return moment / mass

    def cell_energy(self, i_cell: int, iter: int, order: int = 5) -> float:
        x_i = self.seed_matrix[iter, i_cell]
        x_l = self.bound_matrix[iter, i_cell]
        x_r = self.bound_matrix[iter, i_cell + 1]
        def integrand(x): return self.density(x) * (x - x_i) ** 2
        e, _ = fixed_quad(integrand, x_l, x_r, n=order)
        return e

    def total_energy(self) -> np.ndarray:
        total_energy = np.zeros(self.n_iters)
        for iter in range(self.n_iters):
            for ic in range(self.n_cells):
                total_energy[iter] += self.cell_energy(ic, iter)
        return total_energy

    def energy_gradient(self, iter: int) -> np.ndarray:
        xs = self.seed_matrix[iter, :]
        masses = self.cell_masses
        moments = self.cell_moments
        return 2 * (xs * masses - moments)

    def energy_gradient_2(self, iter: int) -> np.ndarray:
        x_is = self.seed_matrix[iter, :]
        x_ls = self.bound_matrix[iter, :-1]
        x_rs = self.bound_matrix[iter, 1:]
        masses = self.cell_masses
        moments = self.cell_moments

        def integrand(x):
            return self.density(x) * (x - x_is) ** 2

        return \
            0.5 * integrand(x_rs) - \
            0.5 * integrand(x_ls) + \
            2 * (x_is * masses - moments)

    def energy_hessian(self, iter: int) -> np.ndarray:
        return 2 * self.cell_masses

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

    def update_cell_masses(self, iter: int):
        for ic in range(self.n_cells):
            self.cell_masses[ic] = self.cell_mass(ic, iter)

    def update_cell_moments(self, iter: int):
        for ic in range(self.n_cells):
            self.cell_moments[ic] = self.cell_moment(ic, iter)

    def update_cell_centroids(self, iter: int):
        for ic in range(self.n_cells):
            self.cell_centroids[ic] = self.cell_centroid(ic, iter)

    def update_cell_seeds(self, iter: int):
        assert iter < self.n_iters - 1, "Cannot update beyond the last iteration"
        seeds_new = self.seed_matrix[iter + 1, :]

        # Centroid calculation (not worth parallelizing)
        for ic in range(self.n_cells):
            seeds_new[ic] = self.cell_centroids[ic]

    def update_cell_seeds_1st_Ord(self, iter: int, iter_step: int = 10) -> bool:
        assert iter < self.n_iters - 1, "Cannot update beyond the last iteration"

        x_0 = self.seed_matrix[iter, :]
        x_1 = self.seed_matrix[iter - 1, :]

        dx = x_0 - x_1

        x_new = self.seed_matrix[iter + 1, :]
        step = iter_step * dx
        x_new[:] = x_0 + step

        if np.any(np.abs(step) > 1.0e-13):
            return False

        return True

    def update_cell_seeds_2nd_Ord(self, iter: int, iter_step: int = 10) -> bool:
        assert iter < self.n_iters - 1, "Cannot update beyond the last iteration"

        x_0 = self.seed_matrix[iter, :]
        x_1 = self.seed_matrix[iter - 1, :]
        x_2 = self.seed_matrix[iter - 2, :]
        x_3 = self.seed_matrix[iter - 3, :]

        # dx = x_0 - x_1
        # dx = 0.5 * (x_0 - x_2)
        dx = 0.5 * (3*x_0 - 4*x_1 + x_2)
        # d2x = x_0 - 2*x_1 + x_2
        d2x = 2*x_0 - 5*x_1 + 4*x_2 - x_3

        # coeffs = np.polyfit(
        #     [0, -1, -2, -3],
        #     [x_0, x_1, x_2, x_3],
        #     deg=2
        # )

        x_new = self.seed_matrix[iter + 1, :]
        step = iter_step * dx + 0.5 * (iter_step ** 2) * d2x
        x_new[:] = x_0 + step
        # x_new[:] = x_1 + step
        # x_new[:] = np.polyval(coeffs, iter_step)

        # if np.any(np.abs(step) > 1.0e-13):
        #     return False

        return False
        # return True

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

        if np.any(np.abs(d2x) > 1.0e-10):
            return False

        return True


class HalfPlane:
    def __init__(self, point: np.ndarray = None, normal=None):
        self.point: np.ndarray = point
        self.tangent: np.ndarray
        self._normal: np.ndarray
        self._compute_normal(normal)
        self._compute_tangent()

    @property
    def normal(self):
        return self._normal

    @normal.setter
    def normal(self, n):
        if n is None:
            return
        self._compute_normal(n)
        self._compute_tangent()

    def _compute_normal(self, n):
        if n is None:
            return
        norm_value = np.linalg.norm(n)
        self._normal = np.divide(n, norm_value)

    def _compute_tangent(self):
        if self.normal is None:
            return
        self.tangent = np.array([-self.normal[1], self.normal[0]])

    def signed_distance(self, point: np.ndarray) -> float:
        return np.dot(self.normal, np.subtract(point, self.point))

    def contains(self, point: np.ndarray, approx=False) -> bool:
        if approx:
            return self.signed_distance(point) > -1.0e-10
        else:
            return self.signed_distance(point) >= 0

    def is_parallel(self, other: "HalfPlane") -> bool:
        return np.isclose(np.dot(self.normal, other.normal), 1.0)

    def intersection(self, other: "HalfPlane") -> np.ndarray:
        if self.is_parallel(other):
            raise ValueError("Half-planes are parallel - not intersection")

        d1 = self.tangent
        d2 = other.tangent
        p1 = self.point
        p2 = other.point

        # 2D cross product: a × b = a[0]*b[1] - a[1]*b[0]
        cross_d1_d2 = d1[0] * d2[1] - d1[1] * d2[0]
        delta_p = p2 - p1  # type: ignore
        cross_delta_d2 = delta_p[0] * d2[1] - delta_p[1] * d2[0]

        t1 = cross_delta_d2 / cross_d1_d2

        return p1 + t1 * d1


class Rectangle:
    def __init__(self, p_sw: np.ndarray, p_ne: np.ndarray):
        self.p_sw = p_sw
        self.p_ne = p_ne
        N = np.array([0, 1])
        S = np.array([0, -1])
        E = np.array([1, 0])
        W = np.array([-1, 0])
        self.half_planes = [
            HalfPlane(p_sw, E),
            HalfPlane(p_ne, W),
            HalfPlane(p_sw, N),
            HalfPlane(p_ne, S),
        ]

    def signed_distance(self, point: np.ndarray) -> float:
        return np.min([hp.signed_distance(point) for hp in self.half_planes])

    def contains(self, point: np.ndarray, approx=False) -> bool:
        return np.all([hp.contains(point, approx) for hp in self.half_planes]).item()

    def intersections(self, hp: HalfPlane) -> list[np.ndarray]:
        pts = []
        for hp in self.half_planes:
            try:
                pts.append(hp.intersection(hp))
            except ValueError:
                continue

        for pt in pts:
            if not self.contains(pt):
                pts.remove(pt)

        assert len(pts) == 2, f"Expected 2 intersections, got {len(pts)}"
        return pts


class SpaceIterMesh2D:
    def __init__(self, p_sw: np.ndarray, p_ne: np.ndarray, n_cells: int, n_iters: int):
        self.boundary = Rectangle(p_sw, p_ne)
        self.n_cells = n_cells
        self.n_iters = n_iters
        self.cell_size: Callable[[np.ndarray], float] | None = None
        self.seed_matrix = np.zeros((n_iters, n_cells, 2))
        self.edges = np.array(list(combinations(range(n_cells), 2)))
        self.half_planes = [HalfPlane() for _ in range(len(self.edges))]

    def set_initial_conditions(self):
        self.seed_matrix[0, :] = np.random.uniform(
            self.boundary.p_sw, self.boundary.p_ne, (self.n_cells, 2))

    def update_half_planes(self, iter: int):
        for i, edge in enumerate(self.edges):
            p0 = self.seed_matrix[iter, edge[0], :]
            p1 = self.seed_matrix[iter, edge[1], :]
            hp = self.half_planes[i]
            hp.point = 0.5 * (p0 + p1)
            hp.normal = p1 - p0

    def compute_intersections(self, iter: int):
        for i, edge in enumerate(self.edges):
            hp = self.half_planes[i]
            pts = self.boundary.intersections(hp)
            assert len(pts) == 2, f"Expected 2 intersections, got {len(pts)}"
            # self.seed_matrix[iter, edge, :] = pts
