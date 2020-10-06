import sys
sys.path.insert(1, '../functions')

import vtk_functions

import vtk
import seaborn as sns
from sklearn.cluster import DBSCAN
from vtk.util import numpy_support
import pandas as pd
import os
from scipy.sparse import dok_matrix


# Get all direct neighbors of vertex_id
def get_connect_vertices(polydata, vertex_id):
    connectedVertices = []

    cellIdList = vtk.vtkIdList()
    polydata.GetPointCells(vertex_id, cellIdList)

    for i in range(cellIdList.GetNumberOfIds()):
        pointIdList = vtk.vtkIdList()
        polydata.GetCellPoints(cellIdList.GetId(i), pointIdList)

        if pointIdList.GetId(0) != vertex_id:
            connectedVertices.append(pointIdList.GetId(0))
        else:
            connectedVertices.append(pointIdList.GetId(1))

    return connectedVertices


# Get (sparse) distance matrix of polydata
def get_distance_matrix(polydata):
    N = polydata.GetPoints().GetNumberOfPoints()
    d = dok_matrix((N, N))

    for i in range(N):
        neighbors = get_connect_vertices(polydata, i)
        d[i, neighbors] = 1
    return d


class CurvatureSegmentation():
    def __init__(self, pd):
        self.polydata = pd

        self.set_normals()
        self.set_mean_curvature()

        # Params
        self.min_H_value = 0
        self.min_cluster_count = 40
        self.min_cr_ring = 500
        self.cluster = True
        self.eps_2 = 10

    def set_normals(self):
        ns = vtk_functions.normals(self.polydata)
        self.polydata = ns

    def set_mean_curvature(self):
        curvatureFilter = vtk.vtkCurvatures()
        curvatureFilter.SetCurvatureTypeToMean()
        curvatureFilter.SetInputData(self.polydata)
        curvatureFilter.Update()
        self.polydata = curvatureFilter.GetOutput()

    def get_cluster_labels(self, points, eps):
        model = DBSCAN(eps=eps)
        model.fit_predict(points)
        return model.labels_

    def threshold_pd(self):
        # threshold on mean curvature value
        threshold = vtk.vtkThreshold()
        threshold.SetInputData(self.polydata)
        threshold.ThresholdByUpper(self.min_H_value)
        threshold.Update()

        polydata = threshold.GetOutput()

        # threshold on normal vector
        normals = list(map(tuple, numpy_support.vtk_to_numpy(polydata.GetPointData().GetArray("Normals"))))

        c = vtk.vtkLongLongArray()
        c.SetName("c")

        for n in normals:
            x, y, z = n
            if abs(x) > abs(y):
                c.InsertNextTuple1(1)
            elif abs(z) > abs(y):
                c.InsertNextTuple1(2)
            else:
                c.InsertNextTuple1(0)

        polydata.GetPointData().AddArray(c)

        threshold = vtk.vtkThreshold()
        threshold.SetInputData(polydata)
        threshold.SetInputArrayToProcess(0, 0, 0, 0, "c")
        threshold.ThresholdByUpper(1)
        threshold.Update()
        return threshold.GetOutput()


    def run(self):
        polydata = self.threshold_pd()

        coords = polydata.GetPoints()

        # PD to
        df = pd.DataFrame()
        df["coord"] = list(map(tuple, numpy_support.vtk_to_numpy(coords.GetData())))

        if self.cluster:
            d = get_distance_matrix(polydata)
            print("Got distance matrix (%sx%s)" % (d.shape[0], d.shape[1]))

            model = DBSCAN(eps=1, min_samples=5, metric="precomputed")
            model.fit(d)
            df["cluster"] = model.labels_

            # remove noise cluster
            df = df[df["cluster"] != -1]

            df = df.groupby("cluster").filter(lambda x: x["cluster"].count() > self.min_cluster_count)

            cluster_labels2 = self.get_cluster_labels(df["coord"].to_list(), self.eps_2)
            df["cluster_ring"] = cluster_labels2

            print(df.groupby("cluster_ring").size().sort_values(ascending=False))

            df = df.groupby("cluster_ring").filter(lambda x: x["cluster"].count() > self.min_cr_ring)

            new_cluster_labels, _ = pd.factorize(df["cluster"])
            df["cluster"] = new_cluster_labels

            print("Clustered DF")
            print(df.groupby("cluster").size().sort_values(ascending=False))

            cluster_labels = vtk.vtkLongLongArray()
            cluster_labels.SetName("Cluster")

        # covert df to vtkpoints
        vtk_points = vtk.vtkPoints()

        for _, row in df.iterrows():
            vtk_points.InsertNextPoint(row["coord"])

            if self.cluster:
                cluster_labels.InsertNextTuple1(row["cluster"])

        points_poly = vtk.vtkPolyData()
        points_poly.SetPoints(vtk_points)

        if self.cluster:
            points_poly.GetPointData().AddArray(cluster_labels)

        vertexGlyphFilter = vtk.vtkVertexGlyphFilter()
        vertexGlyphFilter.AddInputData(points_poly)
        vertexGlyphFilter.Update()

        print("Created polydata of cluster points")

        return vertexGlyphFilter.GetOutput()


if __name__ == "__main__":
    args = sys.argv[1:]
    input_file = str(args[0])
    output_folder = args[1]

    polydata = vtk_functions.read_ply(input_file)

    cs = CurvatureSegmentation(polydata)
    cps = cs.run()

    basename = os.path.splitext(os.path.basename(input_file))[0]
    output_file = "%s/%s_clusters.vtk" % (output_folder, basename)
    vtk_functions.write_vtk(cps, output_file)
    print("Wrote polydata to %s" % output_file)
