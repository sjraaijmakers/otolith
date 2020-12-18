import sys
sys.path.insert(1, '../functions')

import vtk_functions
import general
import numpy as np

if __name__ == "__main__":
    args = sys.argv[1:]
    input_folder = str(args[0])
    output_folder = args[1]

    np_data = general.folder_to_arr(input_folder, file_format="tif",
                                    verbose=False)

    np_data = np.rot90(np_data, k=2, axes=(1, 2))
    general.arr_to_folder(np_data, output_folder)
