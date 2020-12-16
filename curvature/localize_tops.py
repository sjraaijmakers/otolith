# Localizes tops of protuberance in otolith mesh (.ply)

import sys
sys.path.insert(1, '../functions')

import vtk_functions

import vtk
import seaborn as sns
from sklearn.cluster import DBSCAN
from vtk.util import numpy_support
from scipy.sparse.csgraph import connected_components
import pandas as pd
import os
import numpy as np


class CurvatureSegmentation():
    def __init__(self, pd):
        self.polydata = pd
        self.set_normals()
        self.set_mean_curvature()

        # Parameters (described in thesis)
        self.min_H_value = 0
        self.min_cluster_count = 50  # N_min
        self.min_cr_ring = 100
        self.cluster = True
        self.eps_2 = 9

    def set_normals(self):
        self.polydata = vtk_functions.normals(self.polydata)

    def set_mean_curvature(self):
        self.polydata = vtk_functions.mean_curvature(self.polydata)

    def get_cluster_labels(self, X, eps):
        model = DBSCAN(eps=eps)
        model.fit_predict(X)
        return model.labels_

    # Filter polydata via diretion of n and value of H
    def threshold_filter_pd(self):
        threshold = vtk.vtkThreshold()
        threshold.SetInputData(self.polydata)
        threshold.ThresholdByUpper(self.min_H_value)
        threshold.Update()
        polydata = threshold.GetOutput()

        normals = list(map(tuple,
                           numpy_support.vtk_to_numpy(
                               polydata.GetPointData().GetArray("Normals"))))

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

        polydata = threshold.GetOutput()
        polydata.GetPointData().RemoveArray("c")
        return polydata

    def run(self, verbose=True):
        polydata = self.threshold_filter_pd()

        # Create distance matrix from polydata
        d = vtk_functions.get_distance_matrix(polydata)

        if verbose:
            print("Got distance matrix (%sx%s)" % (d.shape[0], d.shape[1]))

        # Transform polydata to dataframe to apply operations
        df = pd.DataFrame()
        df["coord"] = list(map(tuple, numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())))
        _, labels = connected_components(csgraph=d, directed=False, return_labels=True)
        df["cluster"] = labels

        # Remove cluster == -1 (noise cluster)
        df["approved"] = df.apply(lambda x: 0 if x['cluster'] == -1 else 1, axis=1)

        # Remove clusters where size < N_min
        g = df.groupby('cluster')['cluster'].transform("size")
        df.loc[g <= self.min_cluster_count, 'approved'] = 0

        # ring cluster 2
        cluster_labels2 = self.get_cluster_labels(df["coord"].to_list(), self.eps_2)
        df["cluster_ring"] = cluster_labels2

        if verbose:
            print(df.groupby("cluster_ring").size().sort_values(ascending=False))

        g = df.groupby('cluster_ring')['cluster_ring'].transform("size")
        df.loc[g <= self.min_cr_ring), 'approved'] = 0

        # Remap values
        df['cluster_new'] = df.apply(lambda x: x["cluster"] if x['approved'] == 1 else -1, axis=1)

        d = list(df["cluster_new"].unique())
        d.remove(-1)

        new_map = {}

        for i in range(len(d)):
            new_map[d[i]] = i

        new_map[-1] = -1

        df["cluster_new"] = df["cluster_new"].map(new_map)

        if verbose:
            print(df.groupby("cluster_new").size().sort_values(ascending=False))

        cluster_labels = vtk.vtkLongLongArray()
        cluster_labels.SetName("Cluster")

        for _, row in df.iterrows():
            if row["approved"] == 1:
                cluster_labels.InsertNextTuple1(row["cluster_new"])
            else:
                cluster_labels.InsertNextTuple1(-1)

        polydata.GetPointData().AddArray(cluster_labels)

        threshold = vtk.vtkThreshold()
        threshold.SetInputData(polydata)
        threshold.SetInputArrayToProcess(0, 0, 0, 0, "Cluster")
        threshold.ThresholdByUpper(0)
        threshold.Update()

        polydata = threshold.GetOutput()
        polydata.GetPointData().RemoveArray("Mean_Curvature")
        polydata.GetPointData().RemoveArray("Normals")

        return polydata


if __name__ == "__main__":
    args = sys.argv[1:]
    input_file = str(args[0])
    output_folder = args[1]

    polydata = vtk_functions.read_ply(input_file)

    cs = CurvatureSegmentation(polydata)
    cps = cs.run()

    basename = os.path.splitext(os.path.basename(input_file))[0]
    output_file = "%s/%s_clusters_TOST.vtk" % (output_folder, basename)
    vtk_functions.write_vtk(cps, output_file)
    print("Wrote polydata to %s" % output_file)
