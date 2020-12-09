# Derive H values for image stack

import sys
sys.path.insert(1, '../functions')

import vtk_functions
from vtk.util import numpy_support

import vtk
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()


if __name__ == "__main__":
    input_folder = "/home/steven/scriptie/inputs/sphere_r_100_p_20"

    imgdata = vtk_functions.folder_to_imgdata(input_folder)

    sigma = 3

    if sigma > 0:
        padding = sigma * 2
        imgdata = vtk_functions.pad_imgdata(imgdata, padding)
        imgdata = vtk_functions.gauss_imgdata(imgdata, sigma=sigma)

    polydata = vtk_functions.imgdata_to_pd(imgdata)
    polydata = vtk_functions.normals(polydata)
    polydata = vtk_functions.mean_curvature(polydata)

    k  = numpy_support.vtk_to_numpy(polydata.GetPointData().GetArray("Mean_Curvature"))

    print(len(k))

    ax = sns.distplot(k)
    plt.show()



