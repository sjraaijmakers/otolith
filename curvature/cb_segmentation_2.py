import sys
sys.path.insert(1, '../functions')

import vtk_functions

import vtk
import seaborn as sns
from sklearn.cluster import DBSCAN
from vtk.util import numpy_support
import pandas as pd
import os

sns.set()


class CurvatureSegmentation():
    def __init__(self, pd, eps_1=4, eps_2=8):
        self.polydata = pd

        self.set_normals()
        self.set_mean_curvature()

        self.eps_1 = eps_1
        self.eps_2 = eps_2
        self.min_H_value = 0.005
        self.min_cluster_count = 50
        self.min_cr_ring = 1000
        self.cluster = True

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

    # Filter vertices in df based on heuristics
    def filter_df(self, df):
        # check if normal vector points to side
        df = df[df["n"].apply(lambda x: abs(x[0]) > abs(x[1]) or abs(x[2]) > abs(x[1]))]

        # check if mean curvature is positive
        df = df[df["H"] > self.min_H_value]
        return df

    def cluster_df(self, df):
        # cluster remaining vertices
        cluster_labels = self.get_cluster_labels(df["coord"].to_list(), self.eps_1)
        df["cluster"] = cluster_labels

        # remove noise cluster (-1)
        df = df[df["cluster"] != -1]

        # df = df.groupby("cluster").filter(lambda x: x["cluster"].count() >
        #                                   self.min_cluster_count)

        # cluster_labels2 = self.get_cluster_labels(df["coord"].to_list(), self.eps_2)
        # df["cluster_ring"] = cluster_labels2

        # print(df.groupby("cluster_ring").size().sort_values(ascending=False))

        # df = df.groupby("cluster_ring").filter(lambda x: x["cluster"].count() > self.min_cr_ring)

        # filter out cluster smaller than min_cluster_count

        # get average normal vec per cluster
        # df["normal_vec_numpy"] = df["point"].apply(lambda x: np.array(x))
        # df = df.join(df.groupby("cluster")["normal_vec_numpy"].apply(np.mean), on="cluster", rsuffix='_avg')
        # df = df[df["normal_vec_numpy_avg"].apply(lambda x: not abs(x[1]) > abs(x[0]))]

        new_cluster_labels, _ = pd.factorize(df["cluster"])
        df["cluster"] = new_cluster_labels

        return df


    def get_clusters(self):
        xyzs = self.polydata.GetPoints()
        normals = self.polydata.GetPointData().GetArray("Normals")
        mean_curvatures = self.polydata.GetPointData().GetArray("Mean_Curvature")

        # convert polydata info to pandas dataframe
        df = pd.DataFrame()

        df["coord"] = list(map(tuple, numpy_support.vtk_to_numpy(xyzs.GetData())))
        df["n"] = list(map(tuple, numpy_support.vtk_to_numpy(normals)))
        df["H"] = numpy_support.vtk_to_numpy(mean_curvatures)

        print("Polydata to pandas DataFrame (%s points)" % xyzs.GetNumberOfPoints())

        df = self.filter_df(df)

        print("Filtered DF")

        if self.cluster:
            df = self.cluster_df(df)
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

    eps_1 = int(args[2])
    eps_2 = int(args[3])

    polydata = vtk_functions.read_ply(input_file)

    cs = CurvatureSegmentation(polydata, eps_1=eps_1, eps_2=eps_2)
    cps = cs.get_clusters()

    basename = os.path.splitext(os.path.basename(input_file))[0]
    output_file = "%s/%s_clusters.vtk" % (output_folder, basename)
    vtk_functions.write_vtk(cps, output_file)
    print("Wrote polydata to %s" % output_file)
