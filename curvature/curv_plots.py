import sys
sys.path.insert(1, '../functions')

from pickle_functions import open_pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sns.set(font_scale=1.3)

m = sns.color_palette("hls", 9)

sns.set_palette(m)


if __name__ == "__main__":
    df_gm = pd.read_csv("geometric_measurements.csv")

    gender_group = "F"
    df = open_pickle("mc_vals_%s.pkl" % gender_group)
    df_info = df_gm[df_gm["gender"] == gender_group]

    for index, row in df_info.iterrows():
        print(row["label"])
        oto_df = df[df["label"] == row["label"]]
        g = sns.kdeplot(oto_df["mean_curvature"].to_numpy(), gridsize=1000000, label="%s" % (row["label"]))

    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.xlim(-0.1, 0.1)
    plt.ylim(0, 45)
    plt.xlabel("mean curvature")
    plt.ylabel("density")

    plt.savefig("/home/steven/scriptie/code/analysis/figures/%s_curvature.pdf" % gender_group, bbox_inches="tight", pad_inches=0)

    plt.xlim(-0.02, 0.02)
    plt.savefig("/home/steven/scriptie/code/analysis/figures/%s_curvature_zoom.pdf" % gender_group, bbox_inches="tight", pad_inches=0)
