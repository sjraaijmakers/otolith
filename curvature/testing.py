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

    juveniles = gm[gm["gender"] == "F"]

    mc = pd.DataFrame()

    for index, row in juveniles.iterrows():
        folder_name = row["label"]

        if row["omega"] != 1:
            folder_name += "_%s" % row["omega"]

        arr = vtk_functions.folder_to_curv_arr("/home/steven/scriptie/inputs/edited_scans/%s" % folder_name)

        mc_temp = pd.DataFrame()

        mc_temp["mean_curvature"] = arr
        mc_temp["label"] = row["label"]

        mc = pd.concat([mc, mc_temp])

        print("Finished %s" % row["label"])

        print(mc)



    save_pickle(mc, "mc_vals_F.pkl")
