import sys

import vtk_functions

import vtk
import seaborn as sns
from sklearn.cluster import DBSCAN
from vtk.util import numpy_support
import pandas as pd
import os

sys.path.insert(1, '../functions')
sns.set()


class CurvatureSegmentation():
    def __init__(self, pd):
        self.polydata = pd

        self.set_normals()
        self.set_mean_curvature()

        self.eps = 2
        self.min_cluster_count = 100

    def set_normals(self):
        normals = vtk.vtkPolyDataNormals()
        normals.SplittingOff()
        normals.SetInputData(self.polydata)
        normals.ComputePointNormalsOn()
        normals.Update()
        self.polydata = normals.GetOutput()

    def set_mean_curvature(self):
        curvatureFilter = vtk.vtkCurvatures()
        curvatureFilter.SetCurvatureTypeToMean()
        curvatureFilter.SetInputData(self.polydata)
        curvatureFilter.Update()
        self.polydata = curvatureFilter.GetOutput()

    def get_cluster_labels(self, points):
        model = DBSCAN(eps=self.eps)
        model.fit_predict(points)
        return model.labels_

    # Filter vertices in df based on heuristics
    def filter_df(self, df):
        # check if normal vector points to side
        df = df[df["normal_vec"].apply(lambda x: not abs(x[1]) > abs(x[0]))]

        # check if mean curvature is positive
        df = df[df["mean_curvature"] > 0]

        # cluster remaining vertices points
        cluster_labels = self.get_cluster_labels(df["point"].to_list())
        df["cluster"] = cluster_labels

        # remove noise cluster (-1)
        df = df[df["cluster"] != -1]

        # filter out cluster smaller than min_cluster_count
        df = df.groupby("cluster").filter(lambda x: x["cluster"].count() >
                                          self.min_cluster_count)

        # get average normal vec per cluster
        # df["normal_vec_numpy"] = df["point"].apply(lambda x: np.array(x))
        # df = df.join(df.groupby("cluster")["normal_vec_numpy"].apply(np.mean), on="cluster", rsuffix='_avg')
        # df = df[df["normal_vec_numpy_avg"].apply(lambda x: not abs(x[1]) > abs(x[0]))]

        new_cluster_labels, _ = pd.factorize(df["cluster"])
        df["cluster"] = new_cluster_labels

        return df

    def get_clusters(self):
        xyzs = self.polydata.GetPoints()
        normal_vecs = self.polydata.GetPointData().GetArray("Normals")
        mc_vals = self.polydata.GetPointData().GetArray("Mean_Curvature")

        # convert polydata info to pandas dataframe
        df = pd.DataFrame()

        df["point"] = list(map(tuple, numpy_support.vtk_to_numpy(xyzs.GetData())))
        df["normal_vec"] = list(map(tuple,
                                    numpy_support.vtk_to_numpy(normal_vecs)))
        df["mean_curvature"] = numpy_support.vtk_to_numpy(mc_vals)

        print("Polydata info to pandas DataFrame")

        df = self.filter_df(df)

        print("Filtered DF. Clusters: %s " % df["cluster"].unique())

        # covert df to vtkpoints
        vtk_points = vtk.vtkPoints()

        cluster_labels = vtk.vtkLongLongArray()
        cluster_labels.SetName("Cluster")

        for _, row in df.iterrows():
            vtk_points.InsertNextPoint(row["point"])
            cluster_labels.InsertNextTuple1(row["cluster"])

        points_poly = vtk.vtkPolyData()
        points_poly.SetPoints(vtk_points)
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
    cps = cs.get_clusters()

    basename = os.path.splitext(os.path.basename(input_file))[0]
    output_file = "%s/%s_clusters.vtk" % (output_folder, basename)
    vtk_functions.write_vtk(cps, output_file)
    print("Wrote polydata to %s" % output_file)
