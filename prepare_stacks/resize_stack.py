import sys
sys.path.insert(1, '../functions')

import vtk_functions
import general


if __name__ == "__main__":
    max_z = 2000

    input_folder = "/home/steven/inputs/scans/original/otoF75"
    imgdata = vtk_functions.folder_to_imgdata(input_folder, verbose=True)

    print("Loaded files...")

    dims = imgdata.GetDimensions()

    omega = 1

    if dims[2] > max_z:
        omega = general.trunc(max_z / dims[2], 1)
        print("Omega: %s" % omega)

    resized_imgdata = vtk_functions.resize_imgdata(imgdata, omega)

    arr = vtk_functions.imgdata_to_arr(resized_imgdata)
    general.arr_to_imgseq(arr, "/home/steven/inputs/scans/resized/otoF75_%s" % omega)