import sys
sys.path.insert(1, '../functions')

import vtk_functions
import general

import numpy as np

if __name__ == "__main__":
    np_data = general.folder_to_arr("/home/steven/Documents/otoI47_0.9", file_format="tif", verbose=False)

    np_data = np.flip(np_data, axis=1)
    general.arr_to_imgseq(np_data, "/home/steven/Documents/otoI47_0.9_flipped")

