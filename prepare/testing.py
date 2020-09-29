import sys
sys.path.insert(1, '../functions')
sys.path.insert(1, '../sulcus')

import vtk_functions
import general

import vtk
import numpy as np
import os
import general
import matplotlib.pyplot as plt
import sys
from Otolith import Otolith



if __name__ == "__main__":
    folder = "/home/steven/scriptie/inputs/sphere_r_100"
    imgdata = vtk_functions.folder_to_imgdata(folder)
    gauss = vtk_functions.gauss_imgdata(imgdata, sigma=10)

    arr = vtk_functions.imgdata_to_arr(gauss)
    general.arr_to_imgseq(arr, "/home/steven/scriptie/inputs/sphere_r_100_g_10_vtk")