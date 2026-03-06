"""
utils/geo.py - Geospatial utilities
"""

import numpy as np
import pandas as pd


def calculate_bounding_box(df, buffer=50):
    """Calculate bounding box with optional buffer."""
    return {
        'x_min': df['x'].min() - buffer,
        'x_max': df['x'].max() + buffer,
        'y_min': df['y'].min() - buffer,
        'y_max': df['y'].max() + buffer,
        'z_min': df['z'].min() - buffer,
        'z_max': df['z'].max() + buffer
    }


def snap_to_grid(value, step):
    """Snap value to grid."""
    return np.floor(value / step) * step


def create_grid_definition(df, dx, dy, dz, buffer_x=50, buffer_y=50, buffer_z=10):
    """Create grid definition from data bounds."""
    bbox = calculate_bounding_box(df, buffer=0)

    x_min = snap_to_grid(bbox['x_min'] - buffer_x, dx)
    y_min = snap_to_grid(bbox['y_min'] - buffer_y, dy)
    z_min = snap_to_grid(bbox['z_min'] - buffer_z, dz)

    nx = int(np.ceil((bbox['x_max'] + buffer_x - x_min) / dx))
    ny = int(np.ceil((bbox['y_max'] + buffer_y - y_min) / dy))
    nz = int(np.ceil((bbox['z_max'] + buffer_z - z_min) / dz))

    return {
        'x': [x_min, x_min + nx * dx, dx],
        'y': [y_min, y_min + ny * dy, dy],
        'z': [z_min, z_min + nz * dz, dz],
        'nx': nx,
        'ny': ny,
        'nz': nz
    }


def get_grid_coordinates(grid_def):
    """Get grid coordinate arrays."""
    x = np.arange(grid_def['x'][0], grid_def['x'][1], grid_def['x'][2])
    y = np.arange(grid_def['y'][0], grid_def['y'][1], grid_def['y'][2])
    z = np.arange(grid_def['z'][0], grid_def['z'][1], grid_def['z'][2])
    return x, y, z


def cell_size_from_grid(grid_def):
    """Get cell size tuple."""
    return (grid_def['x'][2], grid_def['y'][2], grid_def['z'][2])


def grid_volume(dx, dy, dz):
    """Calculate cell volume."""
    return dx * dy * dz
