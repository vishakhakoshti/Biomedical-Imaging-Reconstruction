import numpy as np
import math
from Grid import Grid
from interpolate import interpolate


def create_sinogram(phantom, number_of_projections, detector_spacing, detector_size, scan_range):
    """
    Create a sinogram using a ray-driven Radon transform.
    For each projection angle, this function simulates rays that pass through the image.
    It integrates the image values along each ray path and stores the result in a sinogram.
    """
    angular_increment = scan_range / number_of_projections  # angle step between projections
    sinogram = Grid(number_of_projections, detector_size, [angular_increment, detector_spacing])
    sinogram.set_origin([0, -0.5 * detector_size * detector_spacing])  # origin in the center of detector

    delta_t = 0.1  # small step size for sampling along the ray

    for p in range(number_of_projections):
        theta = math.radians(p * angular_increment)  # convert angle to radians
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        for d in range(detector_size):
            s = sinogram.index_to_physical(p, d)[1]  # physical detector coordinate

            ray_sum = 0.0
            for t in np.arange(-phantom.height, phantom.height, delta_t):
                # compute coordinates along the ray at angle theta
                x = s * cos_theta - t * sin_theta
                y = s * sin_theta + t * cos_theta
                ray_sum += phantom.get_at_physical(x, y)

            ray_sum *= delta_t  # approximate integration
            sinogram.set_at_index(p, d, ray_sum)

    return sinogram


def backproject(sinogram, size_x, size_y, grid_spacing):
    """
    Perform pixel-driven backprojection.
    For each pixel in the reconstruction grid, it projects the point back onto each sinogram
    and adds the interpolated value. This reconstructs the image from the sinogram.
    """
    reco = Grid(size_x, size_y, grid_spacing)
    reco.set_origin([-0.5 * size_x * grid_spacing[0], -0.5 * size_y * grid_spacing[1]])  # center the origin

    for i in range(size_x):
        for j in range(size_y):
            x, y = reco.index_to_physical(i, j)
            value = 0.0

            for p in range(sinogram.height):
                theta = math.radians(p * sinogram.spacing[0])
                cos_theta = math.cos(theta)
                sin_theta = math.sin(theta)

                # project (x, y) onto the detector coordinate s
                s = x * cos_theta + y * sin_theta
                _, j_s = sinogram.physical_to_index(0, s)
                value += interpolate(sinogram, p, j_s)  # interpolate value from sinogram

            reco.set_at_index(i, j, value)

    return reco


def ramp_filter(sinogram, detector_spacing):
    """
    Apply a ramp filter in the frequency domain.
    It enhances high-frequency details and sharpens the result.
    This is done by multiplying each sinogram row with a ramp function in Fourier space.
    """
    filtered = Grid(sinogram.height, sinogram.width, sinogram.get_spacing())
    filtered.set_origin(sinogram.get_origin())

    N = sinogram.width
    freqs = np.fft.fftfreq(N, d=detector_spacing)
    ramp = np.abs(freqs)  # ramp = |f|

    for i in range(sinogram.height):
        projection = sinogram.buffer[i, :]
        projection_fft = np.fft.fft(projection)
        filtered_fft = projection_fft * ramp
        filtered_projection = np.fft.ifft(filtered_fft).real  # take real part
        filtered.buffer[i, :] = filtered_projection

    return filtered


def ramlak_filter(sinogram, detector_spacing):
    """
    Apply Ram-Lak filter in the spatial domain.
    This filter also sharpens the image, implemented by convolution in real space.
    """
    filtered = Grid(sinogram.height, sinogram.width, sinogram.get_spacing())
    filtered.set_origin(sinogram.get_origin())

    N = sinogram.width
    kernel = np.zeros(N)

    # Define the Ram-Lak kernel (from lecture formula)
    for n in range(-N//2, N//2):
        if n == 0:
            kernel[n % N] = 1 / (4 * detector_spacing ** 2)
        elif n % 2 == 0:
            kernel[n % N] = 0
        else:
            kernel[n % N] = -1 / (math.pi ** 2 * detector_spacing ** 2 * n ** 2)

    for i in range(sinogram.height):
        projection = sinogram.buffer[i, :]
        filtered_projection = np.convolve(projection, kernel, mode='same')
        filtered.buffer[i, :] = filtered_projection

    return filtered


def next_power_of_two(value):
    """
    Return the next power of two greater than the value, then double it.
    This is useful when zero-padding signals for FFT.
    """
    if is_power_of_two(value):
        return value * 2
    else:
        i = 2
        while i <= value:
            i *= 2
        return i * 2


def is_power_of_two(k):
    """
    Check if a number is a power of two.
    """
    return k and not k & (k - 1)
