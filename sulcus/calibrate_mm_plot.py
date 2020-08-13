import sys
sys.path.insert(1, '../functions')

from pickle_functions import open_pickle
import matplotlib.pyplot as plt
import seaborn as sns


sns.set()


if __name__ == "__main__":
    df = open_pickle("sphere_measurements.pkl")
    print(df)
    df["vol_pe"] = (abs(df["vol_estimate"] - df["vol_real"])) / df["vol_real"] * 100
    df["sa_pe"] = (abs(df["sa_estimate"] - df["sa_real"])) / df["sa_real"] * 100

    plt.plot(df["r"], df["vol_pe"], color="red")
    plt.xlabel("r")
    plt.ylabel("% error")
    plt.savefig("../figures/pe_volume.pdf")
    plt.show()