import sys
sys.path.insert(1, '../functions')

import vtk_functions
import vtk
from vtk.util import numpy_support


if __name__ == "__main__":
    polydata = vtk_functions.read_ply("/home/steven/scriptie/inputs/meshes/oto/otoF176_0.5_g_3_MLX.ply")

    minCurv_filter = vtk.vtkCurvatures()
    minCurv_filter.SetCurvatureTypeToMinimum()
    minCurv_filter.SetInputData(polydata)
    minCurv_filter.Update()

    polydata = minCurv_filter.GetOutput()

    maxCurv_filter = vtk.vtkCurvatures()
    maxCurv_filter.SetCurvatureTypeToMaximum()
    maxCurv_filter.SetInputData(polydata)
    maxCurv_filter.Update()

    polydata = maxCurv_filter.GetOutput()

    min_curv_vals = numpy_support.vtk_to_numpy(polydata.GetPointData().GetArray("Minimum_Curvature"))
    max_curv_vals = numpy_support.vtk_to_numpy(polydata.GetPointData().GetArray("Maximum_Curvature"))

    mean_curv_vals = 0.5 * (min_curv_vals + max_curv_vals)

    vtk_arr = numpy_support.numpy_to_vtk(mean_curv_vals)
    vtk_arr.SetName("Mean_Curvature_2")

    polydata.GetPointData().AddArray(vtk_arr)

    vtk_functions.write_vtk(polydata, "TOST.vtk")
