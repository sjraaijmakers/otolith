# Turns .ply file into dataframe with corresponding mean curvature values

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
    args = sys.argv[1:]
    input_file = str(args[0])  # == .ply
    output_file = args[1]  # == .pkl (will contain dataframe)

    mc = pd.DataFrame()

    polydata = vtk_functions.read_ply(input_file)

    df = pd.DataFrame()
    df["mean_curvature"] = vtk_functions.pd_to_curv_arr(polydata)

    save_pickle(df, "%s" % output_file)
