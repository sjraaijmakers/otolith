import sys
sys.path.insert(1, '../functions')


import vtk_functions
from pickle_functions import open_pickle, save_pickle

from scipy.sparse import csr_matrix
import numpy as np
from scipy.sparse import dok_matrix
import vtk
from sklearn.cluster import DBSCAN
from vtk.util import numpy_support
import pandas as pd


def get_connect_vertices(pd, id):
    connectedVertices = []

    cellIdList = vtk.vtkIdList()
    pd.GetPointCells(id, cellIdList)

    for i in range(cellIdList.GetNumberOfIds()):
        # print("id: %s" % cellIdList.GetId(i))

        pointIdList = vtk.vtkIdList()
        pd.GetCellPoints(cellIdList.GetId(i), pointIdList)

        # print("End points are %s and %s" % (pointIdList.GetId(0), pointIdList.GetId(1)))

        if pointIdList.GetId(0) != id:
            # print("Connected to %s" % pointIdList.GetId(0))
            connectedVertices.append(pointIdList.GetId(0))
        else:
            # print("Connected to %s" % pointIdList.GetId(1))
            connectedVertices.append(pointIdList.GetId(1))

    return connectedVertices

def get_distance_matrix(pd):
    N = pd.GetPoints().GetNumberOfPoints()
    d = dok_matrix((N, N))

    for i in range(N):
        neighbors = get_connect_vertices(pd, i)
        d[i, neighbors] = 1

    return d


def func(x, y):
    print(x, y)

if __name__ == "__main__":
    cluster = True

    polydata = vtk_functions.read_ply("/home/steven/scriptie/inputs/meshes/oto/otoF73_0.6_g_3_MLX.ply")
    polydata = vtk_functions.normals(polydata)

    print("Loaded mesh (%s points)" % polydata.GetPoints().GetNumberOfPoints())

    curvatureFilter = vtk.vtkCurvatures()
    curvatureFilter.SetCurvatureTypeToMean()
    curvatureFilter.SetInputData(polydata)
    curvatureFilter.Update()
    polydata = curvatureFilter.GetOutput()

    print("Got curvature...")

    threshold = vtk.vtkThreshold()
    threshold.SetInputData(polydata)
    threshold.ThresholdByUpper(0.01)
    threshold.Update()
    polydata = threshold.GetOutput()

    print("Got threshold (%s points)" % polydata.GetPoints().GetNumberOfPoints())

    normals = list(map(tuple, numpy_support.vtk_to_numpy(polydata.GetPointData().GetArray("Normals"))))

    c = vtk.vtkLongLongArray()
    c.SetName("c")

    for n in normals:
        x, y, z = n
        if abs(x) > abs(y):
            c.InsertNextTuple1(1)
        # elif abs(z) > abs(y):
        #     c.InsertNextTuple1(2)
        else:
            c.InsertNextTuple1(0)

    polydata.GetPointData().AddArray(c)

    threshold = vtk.vtkThreshold()
    threshold.SetInputData(polydata)
    threshold.SetInputArrayToProcess(0, 0, 0, 0, "c")
    threshold.ThresholdByUpper(1)
    threshold.Update()
    polydata = threshold.GetOutput()

    df = pd.DataFrame()

    df["coord"] = list(map(tuple, numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())))

    if cluster:
        d = get_distance_matrix(polydata)
        print("Got distance matrix (%sx%s)" % (d.shape[0], d.shape[1]))

        model = DBSCAN(eps=1, min_samples=5, metric="precomputed")
        model.fit(d)
        df["cluster"] = model.labels_

        df = df[df["cluster"] != -1]

        df = df.groupby("cluster").filter(lambda x: x["cluster"].count() > 50)



        new_cluster_labels, _ = pd.factorize(df["cluster"])
        df["cluster"] = new_cluster_labels

        print("Clustered DF")
        print(df.groupby("cluster").size().sort_values(ascending=False))

    # DF TO VTK

    print("DF to vtk")

    vtk_points = vtk.vtkPoints()

    if cluster:
        cluster_labels = vtk.vtkLongLongArray()
        cluster_labels.SetName("Cluster")

    for _, row in df.iterrows():
        vtk_points.InsertNextPoint(row["coord"])

        if cluster:
            cluster_labels.InsertNextTuple1(row["cluster"])

    points_poly = vtk.vtkPolyData()
    points_poly.SetPoints(vtk_points)

    if cluster:
        points_poly.GetPointData().AddArray(cluster_labels)

    vertexGlyphFilter = vtk.vtkVertexGlyphFilter()
    vertexGlyphFilter.AddInputData(points_poly)
    vertexGlyphFilter.Update()

    vtk_functions.write_vtk(vertexGlyphFilter.GetOutput(), "/home/steven/scriptie/inputs/meshes/oto/clusters/tost/tost.vtk")
