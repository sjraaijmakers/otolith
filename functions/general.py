# General functions

import numpy as np
import os
import cv2
import shutil
import glob
from bresenham import bresenham


def trunc(f, n):
    return np.floor(f * 10 ** n) / 10 ** n


def count_files_in_folder(folder, file_extension):
    return len(glob.glob1(folder, "*.%s" % file_extension))


def get_file_prefix(folder, z_amount):
    blocks = len(str(z_amount))  # count digits in int

    filename = glob.glob1(folder, "*.tif")[int(z_amount / 2)]
    prefix = os.path.splitext(filename)[0]
    return prefix[:-blocks]


# Generate sphere 3d np array
# https://stackoverflow.com/questions/46626267/how-to-generate-a-sphere-in-3d-numpy-array
def sphere(r, val=255, padding=0):
    shape = (r*2 + padding*2, r*2 + padding*2, r*2 + padding*2)
    position = ((r*2 + padding*2)/2, (r*2 + padding*2)/2, (r*2 + padding*2)/2)

    semisizes = (r,) * 3
    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
    position = np.ogrid[grid]

    arr = np.zeros(shape, dtype=float)
    for x_i, semisize in zip(position, semisizes):
        arr += (x_i / semisize) ** 2

    return (arr <= 1.0) * val


# Transform numpy array to image stack
def arr_to_folder(arr, output_folder, img_format="tif", verbose=False):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        shutil.rmtree(output_folder)
        os.makedirs(output_folder)

    for i in range(arr.shape[2]):
        name = output_folder + "/img_" + str(f'{i:04}' + "." + img_format)
        if verbose:
            print("Writing " + name + "...")
        cv2.imwrite(name, arr[:, :, i])

    print("Finished writing %s slices to %s" % (arr.shape[2], output_folder))


# Get y_val for x_val
# xs must be sorted, and xs -> ys must be injective
def get_y_val(x_val, xs, ys):
    if len(xs) != len(ys):
        raise ValueError()

    xs = np.array(xs)
    x_index = np.searchsorted(xs, x_val)

    if x_index >= len(xs):
        return ys[-1]

    return ys[x_index]


# Check if "p1" is within radius r of "p2"
def in_radius(p1, p2, r):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 <= r**2


# Returns set of points representing a line of pixels between "points"
def get_line(points):
    ps = []
    for i in range(len(points) - 1):
        line = list(bresenham(points[i][0], points[i][1], points[i + 1][0],
                              points[i + 1][1]))
        ps += line
    return ps


# Checks distance of x1 to range (x2 - padding, x2 + padding)
def distance_to_range(x1, x2, padding, include_border=False):
    r = 1 if include_border else 0
    ran = (x2 - padding - r, x2 + padding + r)

    if x1 < ran[0]:
        return x1 - ran[0]
    elif x1 > ran[1]:
        return x1 - ran[1]
    return 0


# Get top-edge (x, max(y)) within img
def get_top_edge(img, reverse=False):
    edge_xs = []
    edge_ys = []

    for x in range(img.shape[1]):
        if np.any(img[:, x]):
            if reverse:
                y = np.max(np.nonzero(img[:, x]))
            else:
                y = np.min(np.nonzero(img[:, x]))

                if(len(edge_ys) > 0):
                    inc = -1 if edge_ys[-1] > y else 1
                    for t in range(edge_ys[-1], y, inc):
                        edge_xs.append(x)
                        edge_ys.append(t)

            edge_xs.append(x)
            edge_ys.append(y)

    return np.array(edge_xs), np.array(edge_ys)


# Converts non-zero pixels in 2D array to list of coordinates
def arr_to_points(arr):
    xs, ys = np.nonzero(arr.T)
    return list(zip(xs, ys))


# Transform folder (image stack) to 3D numpy array
def folder_to_arr(folder, file_format="tif", verbose=False):
    files = sorted(glob.glob(folder + "/*." + file_format))

    first_img = cv2.imread(files[0], cv2.IMREAD_UNCHANGED)
    y, x = first_img.shape
    z = len(files)

    dimensions = (x, y, z)

    # init
    slices = np.empty((dimensions[1], dimensions[0], dimensions[2]),
                      dtype=first_img.dtype)

    # Construct 3D array
    for z in range(dimensions[2]):
        image = cv2.imread(files[z], cv2.IMREAD_UNCHANGED)
        slices[:, :, z] = image

        if verbose:
            print("loaded %d" % files[z])

    return slices
