import numpy as np
from Grid import Grid
from interpolate import interpolate
from scipy.signal import convolve

def create_sinogram(phantom, num_projections, detector_spacing, detector_sizeInPixels, angular_scan_range):
    global num_proj, det_size
    det_size = detector_sizeInPixels
    num_proj = num_projections

    theta_spacing = angular_scan_range / num_projections
    sinogram = Grid(num_projections, detector_sizeInPixels, (theta_spacing, detector_spacing))
    sinogram.set_origin(0, -0.5 * (detector_sizeInPixels - 1) * detector_spacing)

    for i in range(num_projections):
        for j in range(detector_sizeInPixels):
            theta, s = sinogram.index_to_physical(i, j)
            line_integral = 0
            p = (s * np.cos(np.deg2rad(theta)), s * np.sin(np.deg2rad(theta)))
            u = (-np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta)))
            delta_t = 0.5
            for t in np.arange(-0.5 * np.sqrt((phantom.get_size()[0] ** 2) + (phantom.get_size()[1] ** 2)) *
                               phantom.get_spacing()[0], 0.5 * np.sqrt(2 * (phantom.get_size()[0] ** 2) +
                               (phantom.get_size()[1] ** 2)) * phantom.get_spacing()[1], 0.5):
                m = (p[0] + t * u[0], p[1] + t * u[1])
                line_integral += phantom.get_at_physical(m[0], m[1]) * delta_t
            sinogram.set_at_index(i, j, line_integral)

    return sinogram

def backproject(sinogram, size_x, size_y, spacing):
    reco = Grid(size_x, size_y, (spacing[0], spacing[1]))
    # reco.set_origin(-0.5 * (size_x - 1) * spacing[0], -0.5 * (size_y - 1) * spacing[1])

    for i in range(size_x):
        for j in range(size_y):
            # Flip the vertical index
            # flip_j = size_y - 1 - j

            phy_x, phy_y = reco.index_to_physical(i, j)
            val = 0
            for k in range(sinogram.get_size()[0]):
                theta, s = sinogram.index_to_physical(k, 0)
                s_world = (phy_x * np.cos(np.deg2rad(theta))) + (phy_y * np.sin(np.deg2rad(theta))) 
                
                s = sinogram.physical_to_index(k, s_world)
                
               # val += interpolate(sinogram, k, s[1])

                reco.set_at_index(i, j, reco.get_at_index(i, j) + interpolate(sinogram, k, s[1]))
                
           # reco.set_at_index(i, flip_j, val)

    return reco

def ramp_filter(sinogram, detector_spacing):
    k = (sinogram.get_size()[1])
    delta_f = 1 / (detector_spacing * k)
    ramp = np.abs(np.fft.fftfreq(k, delta_f))
    ramp[0:k // 2 + 1] = np.linspace(0, 1, k // 2 + 1)
    ramp[k // 2 + 1:] = np.linspace(1 - 1 / k, 0, k // 2 - 1)

    sinogram_padded = np.pad(sinogram.get_buffer(), ((0, 0), (0, k - sinogram.get_size()[1])), 'constant')
    sinogram_fft = np.fft.fft(sinogram_padded, axis=1)
    filtered_sinogram_fft = sinogram_fft * ramp
    filtered_sinogram = np.real(np.fft.ifft(filtered_sinogram_fft, axis=1))

    sinogram.set_buffer(filtered_sinogram[:, :sinogram.get_size()[1]])
    return sinogram

def ramlak_filter(sinogram, detector_spacing):
    sinogram_buffer = sinogram.get_buffer()
    num_projections, detector_size = sinogram.get_size()
    n = detector_size // 2

    # Create the RamLak filter kernel
    kernel_size = detector_size + 1
    kernel = np.zeros(kernel_size)
    j_vals = np.arange(-n, n + 1)
    kernel[n] = 0.25 / (detector_spacing ** 2)
    kernel[1::2] = -1 / (np.pi ** 2 * j_vals[1::2] ** 2 * detector_spacing ** 2)

    # Apply the filter to each projection using convolution
    filtered_sinogram = np.zeros_like(sinogram_buffer)
    for i in range(num_projections):
        projection = sinogram_buffer[i, :]
        filtered_projection = convolve(projection, kernel, mode='same')
        filtered_sinogram[i, :] = filtered_projection

    sinogram.set_buffer(filtered_sinogram)
    return sinogram

def next_power_of_two(value):
    if is_power_of_two(value):
        return value * 2
    else:
        i = 2
        while i <= value:
            i *= 2
        return i * 2

def is_power_of_two(k):
    return k and not k & (k - 1)

def create_fanogram(phantom, number_of_projections, detector_spacing, detector_size, angular_increment, d_si, d_sd):
    fanogram = Grid(number_of_projections, detector_size, (angular_increment, detector_spacing))
    fanogram.set_origin([0, (-0.5 * (detector_size - 1) * detector_spacing)])

    for beta_idx in range(number_of_projections):
        beta = beta_idx * angular_increment
        cos_beta = np.cos(np.deg2rad(beta))
        sin_beta = np.sin(np.deg2rad(beta))
        source_position = (-d_si * sin_beta, d_si * cos_beta)
        M = (d_sd * sin_beta, -d_sd * cos_beta)

        for s in range(detector_size):
            s_world = -0.5 * (detector_size - 1) * detector_spacing + s * detector_spacing

            point_on_detector = (source_position[0] + M[0] + s_world * cos_beta,
                                 source_position[1] + M[1] + s_world * sin_beta)

            ray_SP = (point_on_detector[0] - source_position[0], point_on_detector[1] - source_position[1])
            SP_length = np.sqrt(ray_SP[0] ** 2 + ray_SP[1] ** 2)
            ray_direction = (ray_SP[0] / SP_length, ray_SP[1] / SP_length)

            step_size = 0.5
            number_of_steps = int(np.ceil(SP_length / step_size))
            ray_sum = 0.0

            for i in range(number_of_steps):
                curr_point = (source_position[1] + i * step_size * ray_direction[1],
                              source_position[0] + i * step_size * ray_direction[0])

                val = phantom.get_at_physical(curr_point[0], curr_point[1])
                ray_sum += val * step_size

            fanogram.set_at_index(beta_idx, s, ray_sum)

    return fanogram

def backproject_fanbeam(fanogram, size_x, size_y, image_spacing, d_si, d_sd, rotation_angle=90):
    reco = Grid(size_x, size_y, (image_spacing, image_spacing))
   
    angular_increment = fanogram.get_spacing()[0]
    print('angular increment: ' + str(angular_increment))

    for i in range(fanogram.get_size()[0]):
        for j in range(fanogram.get_size()[1]):
            t = -0.5 * (fanogram.get_size()[1] - 1) * fanogram.get_spacing()[1] + j * fanogram.get_spacing()[1]
            cos_weight = d_sd / np.sqrt(d_sd ** 2 + t ** 2)
            fanogram.set_at_index(i, j, fanogram.get_at_index(i, j) * cos_weight)

    rotation_angle_rad = np.deg2rad(rotation_angle)
    for ix in range(reco.get_size()[0]):
        for iy in range(reco.get_size()[1]):
            x, y = reco.index_to_physical(ix, iy)
            pixel_value = 0.0

            for beta_index in range(fanogram.get_size()[0]):
                beta_degree = beta_index * fanogram.get_spacing()[0] + fanogram.get_origin()[0]
                beta = np.deg2rad(beta_degree)

                # Source position
                source = np.array([-d_si * np.sin(beta), d_si * np.cos(beta)])
                
                source_direction = np.array([np.sin(beta), -np.cos(beta)])
                
                # Pixel position (no rotation for now)
                pixel = np.array([x, y])
                SX = pixel - source

                SQ = np.dot(SX, source_direction)

                # Calculate gamma for this pixel and projection
                #gamma = np.arctan2(SX[0] * np.cos(beta) + SX[1] * np.sin(beta), d_sd)
                #s = d_sd * np.tan(gamma)
                #s_index = (s / fanogram.get_spacing()[1]) + 0.5 * (fanogram.get_size()[1] - 1)


                ratio_alpha = d_sd / SQ
                SP = ratio_alpha * SX
                
                t = SP[0] * np.cos(beta) + SP[1] * np.sin(beta)
                
                value = fanogram.get_at_physical(beta_degree, t)
                
                # value = interpolate(fanogram, beta_index, s_index)
                U = SQ / d_si
                if U != 0:
                    pixel_value += value / (U * U)

            reco.set_at_index(ix, iy, pixel_value)

    return reco

def rebinning(fanogram, d_si, d_sd):
    sinogram = Grid(180, fanogram.get_size()[1], (1, fanogram.get_spacing()[1]))

    for p in range(sinogram.get_size()[0]):
        theta_degree = p * sinogram.get_spacing()[0] + sinogram.get_origin()[0]
        theta = np.deg2rad(theta_degree)

        for s in range(sinogram.get_size()[1]):
            s_world = sinogram.index_to_physical(p, s)[1]
            gamma = np.arctan(s_world / d_si)
            beta = theta - gamma

            # --- Redundancy correction START ---
            if beta < 0:
                beta = beta + 2 * gamma + np.pi
                gamma = -gamma
            # --- Redundancy correction END ---

            s_fan_world = d_sd * np.tan(gamma)
            beta_degrees = np.rad2deg(beta)

            val = fanogram.get_at_physical(beta_degrees, s_fan_world)
            sinogram.set_at_index(p, s, val)
    return sinogram
