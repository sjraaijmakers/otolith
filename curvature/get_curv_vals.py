import sys
sys.path.insert(1, '../functions')

import vtk_functions
from pickle_functions import save_pickle, open_pickle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import seaborn as sns

sns.set()

if __name__ == "__main__":
    gm = pd.read_csv("geometric_measurements.csv")

    gender_group = "I"

    juveniles = gm[gm["gender"] == gender_group]

    mc = pd.DataFrame()

    for index, row in juveniles.iterrows():
        folder_name = row["label"]

        if row["omega"] != 1:
            folder_name += "_%s" % row["omega"]

        file_name = "/home/steven/scriptie/inputs/meshes/oto/og/%s_g_3.ply" % folder_name
        polydata = vtk_functions.read_ply(file_name)
        arr = vtk_functions.pd_to_curv_arr(polydata)

        mc_temp = pd.DataFrame()

        mc_temp["mean_curvature"] = arr
        mc_temp["label"] = row["label"]

        mc = pd.concat([mc, mc_temp])

        print("Finished %s" % row["label"])

    save_pickle(mc, "mc_vals_og_%s.pkl" % gender_group)
