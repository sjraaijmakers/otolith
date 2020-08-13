import sys
sys.path.insert(1, '../functions')

import os
from Otolith import Otolith
from pickle_functions import open_pickle, save_pickle
import pandas as pd


def has_subdirectories(dir):
    for f in os.scandir(s):
        if f.is_dir():
            return True

    return False


def get_volume(oto, t="oto"):
    if t == "oto":
        vol_gen = oto.get_nzp_otolith()
    elif t == "sulcus":
        vol_gen = oto.get_nzp_sulcus()

    total_vol = 0
    for z, vol in vol_gen:
        total_vol += vol

    return total_vol


def get_surface_area(oto, res):
    surface_otolith = oto.get_surface_area_otolith() * res**2
    surface_sulcus = oto.get_surface_area_sulcus() * res**2

    print("Surface area otolith: \t %s" % surface_otolith)
    print("Surface area sulcus: \t %s" % surface_sulcus)


def get_volumes(oto, res):
    volume_otolith = get_volume(oto, t="oto") * res**3
    volume_sulcus = get_volume(oto, t="sulcus") * res**3
    print("Volume otolith: \t %s" % volume_otolith)
    print("Volume sulcus: \t %s" % volume_sulcus)


if __name__ == "__main__":
    # input_folder = "/home/steven/Documents/school/scriptie/SCANS/edits/edits_new"
    # subs = [x[0] for x in os.walk(input_folder)]
    # subs = subs[1:]

    # for s in subs:
    #     if has_subdirectories(s):
    #         continue

    #     name = s.replace(input_folder, "")
    #     print("Measuring %s" % name)

    #     oto = Otolith()
    #     oto.read_from_folder(s)

    #     print("dims: (%s, %s, %s)" % oto.get_dims())

    #     surface = get_volume(oto, t="oto")
    #     print("surface: %s" % oto.get_surface_area_otolith())
    #     print("")

        # print("Preparing %s" % name)
        # basename = os.path.basename(s)
        # out = (output_folder + name).replace(basename, "")

        # prepare_stack.run(s, out)

    labels = [
        "otoF73L",
    ]

    omegas = [
        1
    ]

    for i in range(len(labels)):
        print("Measuring %s" % labels[i])

        if omegas[i] != 1:
            att = "_" + str(omegas[i])
        else:
            att = ""

        folder = "/home/steven/Documents/school/scriptie/SCANS/edits/" + labels[i] + "_prepared" + att
        peaks_file = "/home/steven/Documents/school/scriptie/SULCUS/peaks_map/" + labels[i] + "_prepared" + att + "_PEAKSMAP.pkl"

        oto = Otolith()
        oto.read_from_folder(folder)
        oto.peaks = open_pickle(peaks_file)


        vol = get_volume(oto, t="oto")
        # sa = oto.get_surface_area_otolith()

        print("volume: %s" % vol)
        # print("sa: %s" % sa)
        print()
