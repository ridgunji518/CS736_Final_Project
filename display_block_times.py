
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



def main():
    filename: str = "outputs/yesBackoff_1024ele_10iters.csv"

    df = pd.read_csv(filename, skiprows=4)

    blocks = df["Blocks"].to_numpy()
    threads = df["Threads"].to_numpy()
    times = df["Avg_Time_ms"].to_numpy()
    # ==== 3D Plot ====
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(blocks, threads, times)

    ax.set_xlabel("Blocks")
    ax.set_ylabel("Threads")
    ax.set_zlabel("Time (ms)")

    plt.show()

    


if __name__ == "__main__":
    main()