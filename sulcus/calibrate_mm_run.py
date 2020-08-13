import sys
sys.path.insert(1, '../functions')

import vtk_functions
from pickle_functions import save_pickle

import numpy as np
from vtk.util import numpy_support
import vtk
import pandas as pd


def get_volume(arr):
    vol = 0

    for z in range(arr.shape[2]):
        vol += np.count_nonzero(arr[:, :, z])

    return vol


def get_surface_area(arr):
    imgdata = vtk_functions.arr_to_imgdata(arr)
    imgdata = vtk_functions.gauss_imgdata(imgdata, 10)
    pd = vtk_functions.imgdata_to_pd(imgdata)

    return vtk.get_surface_area(pd)


if __name__ == "__main__":
    radi = [100, 150, 200, 250, 300, 350, 400, 450, 500]

    df = pd.DataFrame()

    vols = []
    sas = []

    for r in radi:
        s = sphere(r, padding=20)

        vol = get_volume(s)
        sa = get_surface_area(s)

        vols.append(vol)
        sas.append(sa)

        print("R: %s" % r)

    real_vol = [4/3 * np.pi * r**3 for r in radi]
    real_sa = [4 * np.pi * r**2 for r in radi]

    df["r"] = radi
    df["vol_estimate"] = vols
    df["sa_estimate"] = sas
    df["vol_real"] = real_vol
    df["sa_real"] = real_sa

    save_pickle(df, "sphere_measurements.pkl")


