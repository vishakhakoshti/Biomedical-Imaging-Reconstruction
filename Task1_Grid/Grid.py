import numpy as np
from interpolate import interpolate


class Grid:
    def __init__(self, height, width, spacing):
        self.origin = np.array([0, 0])
        self.height = height
        self.width = width
        self.spacing = spacing
        self.buffer = np.zeros((height, width))
        self.x = 0
        self.y = 0
        self.physical_origin = (-0.5 * (self.height - 1) * self.spacing[0], -0.5 * (self.width - 1) * self.spacing[1])

    def set_buffer(self, buffer):
        self.buffer = buffer

    def get_buffer(self):
        return self.buffer

    def get_spacing(self):
        return self.spacing

    def set_origin(self, x, y):
        self.physical_origin = (x, y)

    def get_origin(self):
        return self.origin

    def get_size(self):
        return (self.height, self.width)

    def index_to_physical(self, i, j):
        x = self.physical_origin[0] + (i * self.spacing[0])
        y = self.physical_origin[1] + (j * self.spacing[1])
        return (x, y)

    def physical_to_index(self, x, y):
        i = (x-self.physical_origin[0]) / self.spacing[0]
        j = (y-self.physical_origin[1]) / self.spacing[1]
        return (i, j)

    def set_at_index(self, i, j, val):
        self.buffer[i, j] = val

    def get_at_index(self, i, j):
        return self.buffer[i, j]

    def get_at_physical(self, i, j):
        i_x, i_y = self.physical_to_index(i, j)
        # Interpolate the value at the point
        return interpolate(self, i_x, i_y)