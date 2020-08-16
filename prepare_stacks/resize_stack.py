import sys
sys.path.insert(1, '../functions')

import vtk_functions
import general
import os


if __name__ == "__main__":
    args = sys.argv[1:]
    input_folder = str(args[0])
    output_folder = args[1]

    max_z = 2000
    imgdata = vtk_functions.folder_to_imgdata(input_folder, verbose=True)

    print("Loaded files...")

    dims = imgdata.GetDimensions()

    omega = 1

    if dims[2] > max_z:
        omega = general.trunc(max_z / dims[2], 1)
        print("Omega: %s" % omega)

    resized_imgdata = vtk_functions.resize_imgdata(imgdata, omega)

    arr = vtk_functions.imgdata_to_arr(resized_imgdata)

    basename = os.path.basename(input_folder)
    general.arr_to_imgseq(arr, "%s/%s_%s" % (output_folder, basename, omega))