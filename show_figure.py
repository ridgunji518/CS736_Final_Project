import pandas as pd
import matplotlib.pyplot as plt

no_bo = pd.read_csv("outputs/no_hash_backoff_10iters.csv", skiprows=2)
bo    = pd.read_csv("outputs/noada_hash_backoff_10iters.csv",     skiprows=2)
ada_bo = pd.read_csv("outputs/ada_hash_backoff_10iters.csv", skiprows=2)
# 1) 按 Total_Threads 聚合，算平均时间
no_bo_g = no_bo.groupby("Total_Threads", as_index=False)["Avg_Time_ms"].mean()
bo_g    = bo.groupby("Total_Threads", as_index=False)["Avg_Time_ms"].mean()
ada_bo_g = ada_bo.groupby("Total_Threads", as_index=False)["Avg_Time_ms"].mean()

# 2) 按 Total_Threads 排序（从小到大）
no_bo_g = no_bo_g.sort_values("Total_Threads")
bo_g    = bo_g.sort_values("Total_Threads")
ada_bo_g = ada_bo_g.sort_values("Total_Threads")

# 3) 画折线
plt.figure(figsize=(8,6))
plt.plot(no_bo_g["Total_Threads"], no_bo_g["Avg_Time_ms"], label="No Backoff")
plt.plot(bo_g["Total_Threads"],    bo_g["Avg_Time_ms"],    label="Backoff")
plt.plot(ada_bo_g["Total_Threads"], ada_bo_g["Avg_Time_ms"], label="Adaptive Backoff")
plt.xlabel("Total Threads")
plt.ylabel("Average Time (ms)")
plt.title("Average CAS Time vs Total Threads")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figure/Robin_avg_time_by_total_threads.png")
plt.close()

