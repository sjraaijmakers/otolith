import sys
sys.path.insert(1, '../functions')

import vtk_functions

import vtk
import seaborn as sns
from sklearn.cluster import DBSCAN
from vtk.util import numpy_support
import pandas as pd
import os


class CurvatureSegmentation():
    def __init__(self, pd):
        self.polydata = pd
        self.set_normals()
        self.set_mean_curvature()

        # Params
        self.min_H_value = 0.001
        self.min_cluster_count = 50
        self.min_cr_ring = 969
        self.cluster = True
        self.eps_2 = 8

    def set_normals(self):
        self.polydata = vtk_functions.normals(self.polydata)

    def set_mean_curvature(self):
        self.polydata = vtk_functions.mean_curvature(self.polydata)

    def get_cluster_labels(self, X, eps):
        model = DBSCAN(eps=eps)
        model.fit_predict(X)
        return model.labels_

    def threshold_pd(self):
        threshold = vtk.vtkThreshold()
        threshold.SetInputData(self.polydata)
        threshold.ThresholdByUpper(self.min_H_value)
        threshold.Update()
        polydata = threshold.GetOutput()

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


    def run(self, verbose=True):
        # threshold on mean curvature and normal vector
        polydata = self.threshold_pd()

        # transform pd to df to apply clustering

        df = pd.DataFrame()
        df["coord"] = list(map(tuple, numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())))

        if self.cluster:
            d = vtk_functions.get_distance_matrix(polydata)

            if verbose:
                print("Got distance matrix (%sx%s)" % (d.shape[0], d.shape[1]))

            # Cluster on distance in mesh
            model = DBSCAN(eps=1, min_samples=1, metric="precomputed")
            model.fit(d)
            df["cluster"] = model.labels_

            df = df[df["cluster"] != -1]
            df = df.groupby("cluster").filter(lambda x: x["cluster"].count() > self.min_cluster_count)

            # Cluster on euclidean distance (ring)
            cluster_labels2 = self.get_cluster_labels(df["coord"].to_list(), self.eps_2)
            df["cluster_ring"] = cluster_labels2

            if verbose:
                print(df.groupby("cluster_ring").size().sort_values(ascending=False))

            df = df.groupby("cluster_ring").filter(lambda x: x["cluster"].count() > self.min_cr_ring)

            new_cluster_labels, _ = pd.factorize(df["cluster"])
            df["cluster"] = new_cluster_labels

            if verbose:
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

        if verbose:
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
