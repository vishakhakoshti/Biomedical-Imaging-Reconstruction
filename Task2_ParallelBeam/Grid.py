import numpy as np
from interpolate import interpolate

class Grid:
    def __init__(self, height, width, spacing):
        self.height = height
        self.width = width
        self.spacing = spacing
        self.origin = [-0.5 * height * spacing[0], -0.5 * width * spacing[1]]
        self.buffer = np.zeros((height, width))

    def set_at_index(self, i, j, val):
        self.buffer[i, j] = val

    def get_at_index(self, i, j):
        return self.buffer[i, j]

    def get_at_physical(self, x, y):
        # Convert physical coordinates to index space
        i, j = self.physical_to_index(x, y)
        return interpolate(self, i, j)

    def index_to_physical(self, i, j):
        x = self.origin[0] + i * self.spacing[0]
        y = self.origin[1] + j * self.spacing[1]
        return (x, y)

    def physical_to_index(self, x, y):
        i = (x - self.origin[0]) / self.spacing[0]
        j = (y - self.origin[1]) / self.spacing[1]
        return (i, j)

    def get_size(self):
        return (self.height, self.width)

    def set_origin(self, origin):
        self.origin = origin

    def get_origin(self):
        return self.origin

    def set_spacing(self, spacing):
        self.spacing = spacing

    def get_spacing(self):
        return self.spacing

    def get_buffer(self):
        return self.buffer

    def set_buffer(self, buffer):
        self.buffer = buffer
