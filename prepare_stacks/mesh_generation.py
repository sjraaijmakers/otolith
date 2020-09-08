import sys
sys.path.insert(1, '../functions')

import vtk_functions
import general

import vtk
import numpy as np
import os
import general
import matplotlib.pyplot as plt
import sys


def run(input_folder, output_folder):
    imgdata = vtk_functions.folder_to_imgdata(input_folder)

    print("Read folder...")

    sigma = 5

    if sigma > 0:
        padding = sigma * 2

        imgdata = vtk_functions.pad_imgdata(imgdata, padding)
        imgdata = vtk_functions.gauss_imgdata(imgdata, sigma=sigma)

        print("Applied Gaussian smoothing")

    polydata = vtk_functions.imgdata_to_pd(imgdata)

    print("Created polydata")

    basename = os.path.basename(input_folder)
    output_file = "%s/%s_g_%s.ply" % (output_folder, basename, sigma)
    vtk_functions.write_ply(polydata, output_file)

    print("Wrote file to %s" % output_file)


if __name__ == "__main__":
    args = sys.argv[1:]
    input_folder = str(args[0])
    output_folder = args[1]

    run(input_folder, output_folder)



    # sigma = 5
    # padding = sigma * 2

    # s = general.sphere(100)

    # imgdata = general.arr_to_imgdata(s)
    # imgdata = vtk_constantpad(imgdata, padding)
    # imgdata = vtk_functions.gauss(imgdata, sigma=sigma)
    # arr = general.imgdata_to_arr(imgdata)
    # general.arr_to_imgseq(arr.astype(np.uint8), "../INPUTS/sphere_kut")
    # polydata = create_polydata(imgdata)
