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

def gender_plot(gender_group):
    df_gm = pd.read_csv("geometric_measurements.csv")

    df = open_pickle("mc_vals_og_%s.pkl" % gender_group)
    df_info = df_gm[df_gm["gender"] == gender_group]

    for _, row in df_info.iterrows():
        print(row["label"])
        oto_df = df[df["label"] == row["label"]]

        g = sns.kdeplot(oto_df["mean_curvature"].to_numpy(), gridsize=1000000, label="%s" % (row["label"]))

    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.xlim(-0.1, 0.1)
    plt.ylim(0, 65)
    plt.xlabel("mean curvature")
    plt.ylabel("density")

    # plt.savefig("/home/steven/scriptie/code/analysis/figures/%s_curvature.pdf" % gender_group, bbox_inches="tight", pad_inches=0)

    plt.xlim(-0.02, 0.02)
    plt.show()
    # plt.savefig("/home/steven/scriptie/code/analysis/figures/%s_curvature_zoom.pdf" % gender_group, bbox_inches="tight", pad_inches=0)


def gender_comp():
    df_i = open_pickle("mc_vals_I.pkl")
    df_m = open_pickle("mc_vals_M.pkl")
    df_f = open_pickle("mc_vals_F.pkl")

    for i in range(len(df_f["label"].unique())):
        f_label = df_f["label"].unique()[i]
        m_label = df_m["label"].unique()[i]

        i = df_m[df_m["label"] == m_label]
        sns.kdeplot(i["mean_curvature"].to_numpy(), gridsize=1000000, label=i["label"][0], shade=True, color="green")

        i = df_f[df_f["label"] == f_label]
        sns.kdeplot(i["mean_curvature"].to_numpy(), gridsize=1000000, label=i["label"][0], shade=True, color="orange")

        plt.ylim(0, 40)
        plt.xlim(-0.1, 0.1)
        plt.xlabel("mean curvature")
        plt.ylabel("density")
        plt.tight_layout()
        plt.savefig("/home/steven/scriptie/code/analysis/figures/kde_%s_%s.png" % (f_label, m_label), bbox_inches="tight", pad_inches=0)
        plt.show()
        plt.clf()


if __name__ == "__main__":
    gender_plot("I")