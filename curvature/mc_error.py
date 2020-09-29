import sys
sys.path.insert(1, '../functions')

import vtk_functions

import vtk
import numpy as np
from vtk.util import numpy_support


if __name__ == "__main__":
    r = 100

    polydata = vtk_functions.read_ply("/home/steven/scriptie/inputs/meshes/sphere_r_100_p_20_MLX.ply")
    polydata = vtk_functions.normals(polydata)

    mcFilter = vtk.vtkCurvatures()
    mcFilter.SetCurvatureTypeToMean()
    mcFilter.SetInputData(polydata)
    mcFilter.Update()

    points = mcFilter.GetOutput().GetPoints()
    vals = mcFilter.GetOutput().GetPointData().GetScalars()
    mc_errors = []

    for i in range(points.GetNumberOfPoints()):
        p = points.GetPoint(i)
        val = vals.GetComponent(i, 0)

        mc_error = abs(val - 1/r) / (1/r) * 100
        mc_errors.append(mc_error)

    # add mc_erros to data of vtk object
    vtk_da = numpy_support.numpy_to_vtk(mc_errors)
    vtk_da.SetName("mc_error")

    mcFilter.GetOutput().GetPointData().AddArray(vtk_da)

    vtk_functions.write_vtk(mcFilter.GetOutput(), "mc_error_MLX.vtk")
