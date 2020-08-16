import sys
sys.path.insert(1, '../functions')

import general
import vtk_functions

import numpy as np
import matplotlib.pyplot as plt

import cv2
import glob

from scipy import signal
from peakdetect import peakdet
from pickle_functions import save_pickle
from itertools import chain


# import peakutils as pu


"""
    Otolith class:
    NOTE: origin of images is top left, origin of plots is botom left

    TODO:
        - create cmakelists kind of build??
        - general.top_edge to list of points instead of seperate coordinates?
        - verify result of measurements!!!!
            - volume: check
            - surface: check (almost, still need proximal)
        - merge get/detect peaks!
"""


def linear_interpolation(x1, x2, distance_left, distance_right):
    return x1 + (x2 - x1) * (distance_left / (distance_left + distance_right))


# Returns a map containing distances between [points] and line spanned
# by [tops]
def get_distance_map(points, tops):
    line_xs, line_ys = zip(*general.get_line(tops))

    distance_map = np.zeros(len(points))

    for i, p in enumerate(points):
        p_x, p_y = p
        y_val_line = general.get_y_val(p_x, line_xs, line_ys)

        if p_y < y_val_line:
            distance_map[i] = y_val_line - p_y

    return distance_map


# Returns highest point and remaining points
def get_highest(points, tops):
    distance_map = get_distance_map(points, tops)

    if np.any(distance_map):
        highest_point_index = np.argmax(distance_map)
        highest_point = tuple(points[highest_point_index])

        distance_map[highest_point_index] = 0
        remaining_points = points[np.argwhere(distance_map > 0).reshape(-1)]

        return highest_point, remaining_points

    return None, []


class Otolith():
    def __init__(self, name=None, slices=[]):
        self.sulcus = None

        if len(slices) > 0:
            self.set_slices(slices)
        else:
            self.slices = None

        self.peaks = {}
        self.name = name
        self.isosurface = None

        # PDA params
        self.wl = 31
        self.p_margin = 0
        self.max_distance = 10

    def save_peaks(self, filename):
        save_pickle(self.peaks, filename)

    # Read slices from folder
    def read_from_folder(self, folder, file_format="tif", verbose=False):
        self.name = folder.split("/")[-1]
        self.peaks = {}

        files = sorted(glob.glob(folder + "/*." + file_format))

        y, x = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE).shape
        z = len(files)

        dimensions = (x, y, z)

        slices = np.empty((dimensions[1], dimensions[0], dimensions[2]),
                          dtype=np.uint8)

        # Construct 3D array
        for z in range(dimensions[2]):
            image = cv2.imread(files[z], cv2.IMREAD_GRAYSCALE)
            slices[:, :, z] = image

            if verbose:
                print("loaded %d" % files[z])

        self.set_slices(slices)

    def set_peaks(self, peaks):
        if type(peaks) is dict:
            self.peaks = peaks
        else:
            raise Exception("File is not a dictionary")

    def set_slices(self, slices):
        self.slices = slices

    # Function gets peaks from dictionary (if any)
    # otherwise; interpolate peaks
    def get_peaks_2d(self, z):
        if self.peaks.get(z) and len(self.peaks[z]) >= 2:
            # Ignore innerpeaks
            x1 = self.peaks[z][0][0]
            x2 = self.peaks[z][-1][0]

            top_edge_xs, top_edge_ys = general.get_top_edge(
                self.slices[:, :, z])

            return [(x1, general.get_y_val(x1, top_edge_xs, top_edge_ys)),
                    (x2, general.get_y_val(x2, top_edge_xs, top_edge_ys))]
        elif self.max_distance > 0:
            return self.get_interpolated_peaks(z)
        return None

    # Get distance d to closest neighbor (containing peaks) in "direction"
    # within max_distance
    # TODO: check max_distance (seems to stop earlier...)
    def get_neighbor_distance(self, z, direction):
        z += direction
        distance = 1

        while not self.peaks.get(z) and distance < self.max_distance:
            z += direction
            distance += 1

        if distance == self.max_distance:
            return None

        return distance

    # Generates linear interpolated peaks for z, if z has left- and
    # right neighbor within max_distance """
    def get_interpolated_peaks(self, z):
        d_left = self.get_neighbor_distance(z, -1)
        d_right = self.get_neighbor_distance(z, 1)

        if not d_left or not d_right:
            return None

        top_edge_xs, top_edge_ys = general.get_top_edge(self.slices[:, :, z])

        x1_left = self.peaks[z - d_left][0][0]
        x1_right = self.peaks[z + d_right][0][0]
        p1x = linear_interpolation(x1_left, x1_right, d_left, d_right)
        p1x = int(round(p1x))
        p1y = general.get_y_val(p1x, top_edge_xs, top_edge_ys)

        x2_left = self.peaks[z - d_left][-1][0]
        x2_right = self.peaks[z + d_right][-1][0]
        p2x = linear_interpolation(x2_left, x2_right, d_left, d_right)
        p2x = int(round(p2x))
        p2y = general.get_y_val(p2x, top_edge_xs, top_edge_ys)

        return [(p1x, p1y), (p2x, p2y)]

    def detect_peaks_matlab(self, top_edge_xs, top_edge_ys, top_edge_ys_i):
        max_tab, min_tab = peakdet(top_edge_ys_i, self.delta)

        peaks = []
        valleys = []

        if np.any(max_tab):
            max_x_indices = max_tab[:, 0].astype(int)
            peaks = list(zip(top_edge_xs[max_x_indices],
                             top_edge_ys[max_x_indices]))

        if np.any(min_tab):
            min_x_indices = min_tab[:, 0].astype(int)
            valleys = list(zip(top_edge_xs[min_x_indices], top_edge_ys[min_x_indices]))

        return peaks, []

    def detect_peaks_scipy(self, top_edge_xs, top_edge_ys, top_edge_ys_i):
        peaks, _ = signal.find_peaks(top_edge_ys_i, width=10, distance=10,
                                     prominence=1)
        peaks = list(zip(top_edge_xs[peaks], top_edge_ys[peaks]))
        return peaks, []

    # def detect_peaks_pu(self, top_edge_xs, top_edge_ys, top_edge_ys_i):
    #     indices = pu.indexes(top_edge_ys_i, thres=0.8, min_dist=10)

    #     peaks = list(zip(top_edge_xs[indices], top_edge_ys[indices]))

    #     return peaks, []

    """ PEAK DETECTION """
    # Find peaks desribing the sulcus in slice z
    # wl        =   window length to smoothen otolith's edge
    # p_margin  =   max distance allowed between old and new peaks
    # delta     =   parameter for peak detection algorithm
    # old_peaks =   peaks found in previous slice
    def detect_peaks_2d(self, z, old_peaks=[], verbose=False):
        poly_order = 3

        img = self.slices[:, :, z]
        # self.slices[:, :, z] = img

        top_edge_xs, top_edge_ys = general.get_top_edge(img)

        # Smoothen top_edge_ys
        if self.wl > poly_order:
            if len(top_edge_ys) < self.wl:
                return []
            top_edge_ys_hat = signal.savgol_filter(top_edge_ys, self.wl, poly_order)
        else:
            top_edge_ys_hat = top_edge_ys
        top_edge_ys_i = (top_edge_ys_hat * -1)

        """ Find peaks in top edge """
        # todo: different algorithm?

        peaks, valleys = self.detect_peaks_scipy(top_edge_xs, top_edge_ys,
                                                 top_edge_ys_i)

        # If old_peaks where given check if new peaks are within range
        if old_peaks and self.p_margin > 0:
            if verbose:
                plt.axvline(x=old_peaks[0][0] - self.p_margin, alpha=0.3)
                plt.axvline(x=old_peaks[0][0] + self.p_margin, alpha=0.3)

                plt.axvline(x=old_peaks[-1][0] - self.p_margin, alpha=0.3)
                plt.axvline(x=old_peaks[-1][0] + self.p_margin, alpha=0.3)

                peaks_xs, peaks_ys = zip(*peaks)

                plt.plot(peaks_xs, peaks_ys, "o", c="m", alpha=0.7)
                plt.plot([old_peaks[0][0], old_peaks[-1][0]],
                         [old_peaks[0][1], old_peaks[-1][1]],
                         "o", c="b", alpha=0.7)

            # check distance of lm peak wrt old_peak[0] +- range
            distance_left = general.distance_to_range(peaks[0][0],
                                                      old_peaks[0][0],
                                                      self.p_margin)
            if distance_left != 0:
                new_peak = (old_peaks[0][0], general.get_y_val(old_peaks[0][0],
                                                               top_edge_xs,
                                                               top_edge_ys))

                if distance_left < 0:
                    peaks[0] = new_peak
                else:
                    peaks.insert(0, new_peak)
                    peaks = sorted(peaks, key=lambda p: p[0])

            # check distance of rm peak wrt old_peak[-1] +- range
            distance_right = general.distance_to_range(peaks[-1][0],
                                                       old_peaks[-1][0],
                                                       self.p_margin)
            if distance_right != 0:
                new_peak = (old_peaks[-1][0], general.get_y_val(
                    old_peaks[-1][0], top_edge_xs, top_edge_ys))

                if distance_right > 0:
                    peaks[-1] = new_peak
                else:
                    peaks.insert(-1, new_peak)
                    peaks = sorted(peaks, key=lambda p: p[0])

        if verbose:
            self.peaks[z] = peaks

            plt.plot(top_edge_xs, top_edge_ys_hat, ".", markersize=2,
                     color="orange")
            plt.imshow(img, cmap="gray")

            if peaks:
                peaks_xs, peaks_ys = zip(*peaks)
                plt.axis("off")
                plt.tight_layout()
                plt.plot(peaks_xs, peaks_ys, "o", c="red")

            plt.show()

        return peaks

    # Find sulcus peaks in consecutive slices
    def detect_peaks_loop(self, z_range, inc, max_nps, verbose=False):
        z_start, z_end = z_range
        z_inc = 1 * inc if z_start <= z_end else -1 * inc

        z = z_start
        peaks = []
        nps = 0

        # Terminate search process if "max_nps" consecutive slices have
        # no peaks or end is reached
        while nps < max_nps and \
                ((z_inc > 0 and z < z_end) or (z_inc < 0 and z >= z_end)):

            if not np.any(self.slices[:, :, z]):
                old_peaks = []
            else:
                old_peaks = self.detect_peaks_2d(z, peaks)

            if verbose:
                print("Slice %d: %d peak(s) found." % (z, len(old_peaks)))

            if len(old_peaks) < 2:
                nps += 1
            else:
                peaks = old_peaks

                if nps > 0:
                    nps = 0

            yield z, old_peaks

            z += z_inc

        if verbose:
            print("STOPPED")

    # Functions find peaks in all slices
    def detect_peaks(self, inc=1, max_nps=2, verbose=False):
        middle_z = int(np.ceil(self.slices.shape[2] / 2))

        # Run from middle slice to end
        first = self.detect_peaks_loop((middle_z, self.slices.shape[2]), inc,
                                       max_nps, verbose=verbose)

        # Run from middle slice - 1 to begin
        second = self.detect_peaks_loop((middle_z - 1, 0), inc, max_nps,
                                        verbose=verbose)

        return chain(first, second)

    """ RECONSTRUCT SULCUS """
    # Convert outerpeaks to two dimensional image of sulcus
    def get_sulcus_2d(self, z, img=[]):
        if not np.any(img):
            img = self.slices[:, :, z]

        peaks_new = self.get_peaks_2d(z)

        if not peaks_new:
            return None

        top_edge_xs, top_edge_ys = general.get_top_edge(img)

        # Get all edge-pixels in between peaks
        points_all_xs = np.arange(peaks_new[0][0] + 1, peaks_new[-1][0])
        points_all_ys = [general.get_y_val(points_all_x, top_edge_xs,
                         top_edge_ys) for points_all_x in points_all_xs]
        points_all = np.array(list(zip(points_all_xs, points_all_ys)))

        points_highest, points_remaining = get_highest(points_all,
                                                       peaks_new)

        # Keep on looking if any pixels will cross the line
        while points_highest:
            peaks_new.insert(-1, points_highest)
            peaks_new = sorted(peaks_new, key=lambda p: p[0])
            points_highest, points_remaining = \
                get_highest(points_remaining, peaks_new)

        img_sul = np.zeros(img.shape, dtype=np.uint8)

        # Fill surface between line of peaks and otolith edge
        line = general.get_line(peaks_new)

        for p_x, p_y in line:
            edge_y = general.get_y_val(p_x, top_edge_xs, top_edge_ys)
            img_sul[p_y + 1: edge_y, p_x] = 1

        # p = np.pad(img + img_sul*255, [(20, 0), (0, 0)])
        # plt.imshow(p, cmap="gray")
        # line = general.get_line(peaks_new)
        # line_xs, line_ys = zip(*line)
        # plt.plot(line_xs, np.array(line_ys) + 20, ".", markersize=3, c="orange")
        # peaks_new_xs, peaks_new_ys = zip(*peaks_new)
        # plt.plot(peaks_new_xs, np.array(peaks_new_ys) + 20, "o", markersize=10, c="red")
        # # plt.plot(points_highest[0], points_highest[1], "o", markersize=10, c="blue")
        # plt.xlim(peaks_new[0][0] - 50, peaks_new[-1][0] + 50)


        # plt.show()

        return img_sul

    # Convert known peaks to volume describing the sulcus
    def get_sulcus(self, verbose=False):
        sulcus = np.zeros((self.slices.shape), dtype=np.uint8)

        first_peakful_slice = min(list(self.peaks.keys()))
        last_peakful_slice = max(list(self.peaks.keys()))

        for z in range(first_peakful_slice, last_peakful_slice + 1):
            sulcus_2d = self.get_sulcus_2d(z)

            if np.any(sulcus_2d):
                sulcus[:, :, z] = sulcus_2d

            if verbose:
                print("Peaks to sulcus for slice %s" % z)

        return sulcus

    def get_isosurface_otolith(self):
        if not self.isosurface:
            polydata = vtk_functions.arr_to_pd(self.slices)
            self.isosurface = polydata

        return self.isosurface

    def get_isosurface_sulcus(self, sulcus_slices=[]):
        if not np.any(sulcus_slices):
            sulcus_slices = self.get_sulcus()

        polydata = vtk_functions.arr_to_pd(sulcus_slices)
        return polydata

    """ MEASUREMENTS """
    def get_nzp_otolith(self):
        for z in range(self.slices.shape[2]):
            slice_nzp = general.count_non_empty_pixels(self.slices[:, :, z])
            yield z, slice_nzp

    def get_nzp_sulcus(self):
        first_peakful_slice = min(list(self.peaks.keys()))
        last_peakful_slice = max(list(self.peaks.keys()))

        for z in range(first_peakful_slice, last_peakful_slice + 1):
            sulcus_2d = self.get_sulcus_2d(z)
            slice_nzp = general.count_non_empty_pixels(sulcus_2d)
            yield z, slice_nzp

    def get_proximal_surface_otolith(self):
        for z in range(self.slices.shape[2]):
            img = self.slices[:, :, z]

            if not np.any(img):
                line_length = 0
            else:
                edge_xs, edge_ys = general.get_top_edge(img)
                edge = list(zip(edge_xs, edge_ys))
                line_length = general.get_line_length(edge)

            yield z, line_length

    def get_volume_otolith_vtk(self):
        polydata = self.get_isosurface_otolith()
        return vtk_functions.get_volume(polydata)

    def get_surface_area_otolith(self):
        polydata = self.get_isosurface_otolith()
        return vtk_functions.get_surface_area(polydata)

    def get_surface_area_sulcus(self):
        if self.peaks:
            polydata = self.get_isosurface_sulcus()

            return vtk_functions.get_surface_area(polydata)
        return 0

    # # TODO: THIS IS NOT RIGHT YET
    # def get_proximal_surface_sulcus(self):
    #     for z in range(self.slices.shape[2]):
    #         img = self.slices[:, :, z]

    #         sulcus_2d = self.get_sulcus_2d(z)

    #         if not np.any(img):
    #             line_length = 0
    #         else:
    #             edge_xs, edge_ys = general.get_top_edge(img, True)
    #             edge = list(zip(edge_xs, edge_ys))
    #             line_length = general.get_line_length(edge)

    #         yield z, line_length

    def get_dims(self):
        return (self.slices.shape[1], self.slices.shape[0],
                self.slices.shape[2])

    def __repr__(self):
        s = "OTOLITH\n"
        s += "Dimensions: %d\n" % self.slices.shape
        s += "PDA parameters:\n"
        s += " - WL: %d\n" % self.wl
        return s
