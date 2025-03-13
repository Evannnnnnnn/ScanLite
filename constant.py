import numpy as np

width = 0.216
length = 0.279

marker_size = 0.021

marker_position = {
    23:
    np.array([
        [0, marker_size, 0],    # Top-left
        [marker_size, marker_size, 0],     # Top-right
        [marker_size, 0, 0],    # Bottom-right
        [0, 0, 0]
    ]),
    75:
    np.array([
        [0, length, 0],    # Top-left
        [marker_size, length, 0],     # Top-right
        [marker_size, length - marker_size, 0],    # Bottom-right
        [0, length - marker_size, 0]
    ]),
    127:
    np.array([
        [width, 0, 0],    # Top-left
        [width - marker_size, 0, 0],     # Top-right
        [width - marker_size, marker_size, 0],    # Bottom-right
        [width, marker_size, 0]
    ]),
    200:
    np.array([
        [width - marker_size, length, 0],    # Top-left
        [width, length, 0],     # Top-right
        [width, length - marker_size, 0],    # Bottom-right
        [width - marker_size, length - marker_size, 0]
    ])
}