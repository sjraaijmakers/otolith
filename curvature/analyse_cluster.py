import vtk
from vtk.util import numpy_support
import pandas as pd
import os


if __name__ == "__main__":
    input_folder = "/home/steven/scriptie/inputs/meshes/oto/clusters"
    for file_short in os.listdir(input_folder):
        if file_short.endswith(".vtk"):
            file_name = os.path.join(input_folder, file_short)

            reader = vtk.vtkPolyDataReader()
            reader.SetFileName(file_name)
            reader.Update()
            polydata = reader.GetOutput()

            df = pd.DataFrame()

            df["cluster"] =  numpy_support.vtk_to_numpy(polydata.GetPointData().GetArray("Cluster"))

            print(file_short, len(df.groupby("cluster").size()))