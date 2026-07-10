"""
Support handling utilities for simulation regularization.

The canonical Minerals workflow simulates on a finer support and then
regularizes to reporting blocks by arithmetic averaging.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    origin_xyz: tuple[float, float, float]
    dims: tuple[int, int, int]
    cell_size_m: tuple[float, float, float]

    @property
    def dx(self) -> float:
        return self.cell_size_m[0]

    @property
    def dy(self) -> float:
        return self.cell_size_m[1]

    @property
    def dz(self) -> float:
        return self.cell_size_m[2]

    @property
    def nx(self) -> int:
        return self.dims[0]

    @property
    def ny(self) -> int:
        return self.dims[1]

    @property
    def nz(self) -> int:
        return self.dims[2]

    def axes(self) -> dict[str, np.ndarray]:
        x0, y0, z0 = self.origin_xyz
        return {
            "x": x0 + np.arange(self.nx) * self.dx,
            "y": y0 + np.arange(self.ny) * self.dy,
            "z": z0 + np.arange(self.nz) * self.dz,
        }

    def to_meta(self) -> dict:
        axes = self.axes()
        return {
            "origin_xyz": list(self.origin_xyz),
            "dx": self.dx,
            "dy": self.dy,
            "dz": self.dz,
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
            "x_min": float(axes["x"][0]),
            "x_max": float(axes["x"][-1]),
            "y_min": float(axes["y"][0]),
            "y_max": float(axes["y"][-1]),
            "z_min": float(axes["z"][0]),
            "z_max": float(axes["z"][-1]),
            "block_volume_m3": float(self.dx * self.dy * self.dz),
        }


def grid_spec_from_runtime(grid_def: dict) -> GridSpec:
    return GridSpec(
        origin_xyz=(
            float(grid_def["x"][0]),
            float(grid_def["y"][0]),
            float(grid_def["z"][0]),
        ),
        dims=(int(grid_def["nx"]), int(grid_def["ny"]), int(grid_def["nz"])),
        cell_size_m=(float(grid_def["dx"]), float(grid_def["dy"]), float(grid_def["dz"])),
    )


def grid_spec_from_config(grid_cfg: dict) -> GridSpec:
    return GridSpec(
        origin_xyz=tuple(float(v) for v in grid_cfg["origin_xyz"]),
        dims=(int(grid_cfg["nx"]), int(grid_cfg["ny"]), int(grid_cfg["nz"])),
        cell_size_m=(float(grid_cfg["dx"]), float(grid_cfg["dy"]), float(grid_cfg["dz"])),
    )


def reporting_grid_from_config(config: dict | None) -> GridSpec | None:
    if not config:
        return None
    grid_cfg = config.get("reporting_grid")
    if not grid_cfg:
        return None
    return grid_spec_from_config(grid_cfg)


def regularization_factors(sim_grid: GridSpec, reporting_grid: GridSpec) -> tuple[int, int, int]:
    if tuple(round(v, 8) for v in sim_grid.origin_xyz) != tuple(round(v, 8) for v in reporting_grid.origin_xyz):
        raise ValueError("Simulation and reporting grids must share the same origin")

    fx = reporting_grid.dx / sim_grid.dx
    fy = reporting_grid.dy / sim_grid.dy
    fz = reporting_grid.dz / sim_grid.dz
    factors = (fx, fy, fz)
    if any(abs(v - round(v)) > 1e-8 for v in factors):
        raise ValueError(f"Reporting grid must be an integer multiple of simulation support, got {factors}")

    out = tuple(int(round(v)) for v in factors)
    if (
        reporting_grid.nx * out[0] != sim_grid.nx
        or reporting_grid.ny * out[1] != sim_grid.ny
        or reporting_grid.nz * out[2] != sim_grid.nz
    ):
        raise ValueError("Reporting grid dimensions are inconsistent with the simulation grid")
    return out


def regularize_realizations(
    realizations: np.ndarray,
    sim_grid_def: dict,
    config: dict | None,
) -> tuple[np.ndarray, dict, dict] | tuple[None, None, None]:
    reporting_grid = reporting_grid_from_config(config)
    if reporting_grid is None:
        return None, None, None

    sim_grid = grid_spec_from_runtime(sim_grid_def)
    fx, fy, fz = regularization_factors(sim_grid, reporting_grid)
    n_real = realizations.shape[0]

    reshaped = realizations.reshape(
        n_real,
        reporting_grid.nx,
        fx,
        reporting_grid.ny,
        fy,
        reporting_grid.nz,
        fz,
    )
    regularized = reshaped.mean(axis=(2, 4, 6), dtype=np.float64).astype(np.float32)
    reporting_axes = reporting_grid.axes()
    grid_def = {
        "x": reporting_axes["x"],
        "y": reporting_axes["y"],
        "z": reporting_axes["z"],
        "dx": reporting_grid.dx,
        "dy": reporting_grid.dy,
        "dz": reporting_grid.dz,
        "nx": reporting_grid.nx,
        "ny": reporting_grid.ny,
        "nz": reporting_grid.nz,
    }
    meta = {
        **reporting_grid.to_meta(),
        "derived_from_simulation_support_m": [sim_grid.dx, sim_grid.dy, sim_grid.dz],
        "regularization_factors": [fx, fy, fz],
        "method": "arithmetic_block_average",
    }
    return regularized, grid_def, meta
