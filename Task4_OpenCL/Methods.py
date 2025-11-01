import numpy as np
import math
from Grid import Grid
from interpolate import interpolate
import pyopencl as cl
import pyopencl.array


def create_sinogram(phantom, number_of_projections, detector_spacing, detector_size, scan_range):
    angular_increment = scan_range / number_of_projections
    sinogram = Grid(number_of_projections, detector_size, [angular_increment, detector_spacing])
    sinogram.set_origin([0, -(detector_size / 2 - 0.5) * detector_spacing])

    ray_length = math.sqrt(
        (phantom.height * phantom.spacing[0]) ** 2 +
        (phantom.width * phantom.spacing[1]) ** 2
    )
    delta_t = 0.5
    num_of_samples = int(ray_length / delta_t)
    t_values = np.linspace(-0.5 * ray_length, 0.5 * ray_length, num_of_samples)

    for p in range(number_of_projections):
        theta = math.radians(p * angular_increment)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        for d in range(detector_size):
            s = sinogram.get_origin()[1] + d * detector_spacing
            x = -t_values * cos_theta + s * sin_theta
            y = t_values * sin_theta + s * cos_theta
            values = [phantom.get_at_physical(xi, yi) for xi, yi in zip(x, y)]
            ray_sum = np.sum(values) * delta_t
            sinogram.set_at_index(p, d, ray_sum)

    return sinogram


def backproject(sinogram, size_x, size_y, grid_spacing):
    reco = Grid(size_x, size_y, grid_spacing)
    reco.set_origin([-0.5 * size_x * grid_spacing[0], -0.5 * size_y * grid_spacing[1]])
    num_projections = sinogram.height
    angular_increment = sinogram.get_spacing()[0]
    detector_spacing = sinogram.get_spacing()[1]

    for i in range(size_x):
        for j in range(size_y):
            x, y = reco.index_to_physical(i, j)
            val = 0.0
            for p in range(num_projections):
                theta = math.radians(p * angular_increment)
                cos_theta = math.cos(theta)
                sin_theta = math.sin(theta)
                s = x * sin_theta - y * cos_theta
                s_index = (s - sinogram.get_origin()[1]) / detector_spacing
                val += interpolate(sinogram, p, s_index)
            reco.set_at_index(i, j, val)
    return reco


def ramp_filter(sinogram, detector_spacing):
    projections, detectors = sinogram.get_size()
    result = Grid(projections, detectors, sinogram.get_spacing())
    result.set_origin(sinogram.get_origin())

    for i in range(projections):
        row = sinogram.buffer[i]
        n = len(row)
        padded_len = next_power_of_two(n)
        padded = np.zeros(padded_len)
        padded[:n] = row

        freqs = np.fft.fftfreq(padded_len, d=detector_spacing)
        ramp = np.abs(freqs)
        filtered = np.fft.ifft(np.fft.fft(padded) * ramp).real
        result.buffer[i, :] = filtered[:n]

    return result


def ramlak_filter(sinogram, detector_spacing):
    projections, detectors = sinogram.get_size()
    result = Grid(projections, detectors, sinogram.get_spacing())
    result.set_origin(sinogram.get_origin())

    for i in range(projections):
        row = sinogram.buffer[i]
        n = len(row)
        kernel = np.zeros(n)
        for k in range(n):
            if k == n // 2:
                kernel[k] = 1 / (4 * detector_spacing ** 2)
            elif (k - n // 2) % 2 == 1:
                kernel[k] = -1 / (math.pi ** 2 * detector_spacing ** 2 * (k - n // 2) ** 2)
        filtered = np.convolve(row, kernel, mode='same')
        result.buffer[i] = filtered

    return result


def next_power_of_two(value):
    if is_power_of_two(value):
        return value * 2
    i = 2
    while i <= value:
        i *= 2
    return i * 2


def is_power_of_two(k):
    return k and not k & (k - 1)


def addGrids_texture(g1, g2):
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)

    g1_np = g1.buffer.astype(np.float32)
    g2_np = g2.buffer.astype(np.float32)
    shape = g1_np.shape
    r_np = np.empty_like(g1_np)

    fmt = cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT)
    g1_img = cl.Image(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, fmt, shape=shape[::-1], hostbuf=g1_np)
    g2_img = cl.Image(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, fmt, shape=shape[::-1], hostbuf=g2_np)
    r_img = cl.Image(ctx, cl.mem_flags.WRITE_ONLY, fmt, shape=shape[::-1])

    with open("addGridscl.sec", "r") as f:
        kernel_src = f.read()
    prg = cl.Program(ctx, kernel_src).build()

    prg.add_texture(queue, shape[::-1], None, g1_img, g2_img, r_img)

    origin = (0, 0, 0)
    region = shape[::-1] + (1,)
    cl.enqueue_copy(queue, r_np, r_img, origin=origin, region=region)

    result = Grid(*shape, g1.spacing)
    result.set_buffer(r_np)
    return result


def backproject_cl(sinogram, reco_sizeX, reco_sizeY, reco_spacing):
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)

    sinogram_np = sinogram.buffer.astype(np.float32)
    reco_np = np.zeros((reco_sizeX, reco_sizeY), dtype=np.float32)

    fmt = cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT)
    sinogram_img = cl.Image(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, fmt, shape=sinogram_np.shape[::-1], hostbuf=sinogram_np)
    reco_img = cl.Image(ctx, cl.mem_flags.WRITE_ONLY, fmt, shape=reco_np.shape[::-1])

    with open("backprojectioncl.sec", "r") as f:
        kernel_src = f.read()
    prg = cl.Program(ctx, kernel_src).build()

    prg.backproject(
        queue, reco_np.shape[::-1], None,
        sinogram_img, reco_img,
        np.int32(sinogram.height),
        np.int32(sinogram.width),
        np.float32(sinogram.spacing[0]),
        np.float32(sinogram.spacing[1]),
        np.float32(sinogram.get_origin()[1]),
        np.int32(reco_sizeX),
        np.int32(reco_sizeY),
        np.float32(-0.5 * reco_sizeX * reco_spacing[0]),
        np.float32(-0.5 * reco_sizeY * reco_spacing[1]),
        np.float32(reco_spacing[0]),
        np.float32(reco_spacing[1])
    )

    origin = (0, 0, 0)
    region = reco_np.shape[::-1] + (1,)
    cl.enqueue_copy(queue, reco_np, reco_img, origin=origin, region=region)

    result = Grid(reco_sizeX, reco_sizeY, reco_spacing)
    result.set_buffer(reco_np)
    return result

def addGrids_texture(grid1, grid2):
    import pyopencl as cl
    import numpy as np

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    height, width = grid1.height, grid1.width
    mf = cl.mem_flags

    arr1 = grid1.buffer.astype(np.float32)
    arr2 = grid2.buffer.astype(np.float32)

    arr1_rgba = np.zeros((height, width, 4), dtype=np.float32)
    arr2_rgba = np.zeros((height, width, 4), dtype=np.float32)
    arr1_rgba[..., 0] = arr1
    arr2_rgba[..., 0] = arr2

    fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT)
    im1 = cl.Image(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, fmt, shape=(width, height), hostbuf=arr1_rgba)
    im2 = cl.Image(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, fmt, shape=(width, height), hostbuf=arr2_rgba)
    result = cl.Image(ctx, mf.WRITE_ONLY, fmt, shape=(width, height))

    with open("addGridscl.sec", "r") as f:
        kernel_src = f.read()
    prg = cl.Program(ctx, kernel_src).build()

    prg.add_texture(queue, (width, height), None, im1, im2, result, np.int32(width), np.int32(height))

    out_rgba = np.empty((height, width, 4), dtype=np.float32)
    origin = (0, 0, 0)
    region = (width, height, 1)
    try:
        cl.enqueue_read_image(queue, result, origin, region, out_rgba).wait()
    except AttributeError:
        cl._enqueue_read_image(queue, result, origin, region, out_rgba).wait()
    out = out_rgba[..., 0]

    grid_out = Grid(height, width, [1, 1])
    grid_out.set_buffer(out)
    return grid_out