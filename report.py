import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# file="exp-data/epoch-10/yolov5s-10.csv"
# file="exp-data/epoch-25/yolov5s-25.csv"
# file="exp-data/epoch-50/yolov5s-50.csv"
# file="exp-data/epoch-75/yolov5s-75.csv"
# file="exp-data/epoch-100/yolov5s-100.csv"
# file="exp-data/yolov5n/yolov5n-100.csv"
# file="exp-data/yolov5s/yolov5s-100.csv"
file="exp-data/yolov5m/yolov5m-100.csv"
print("DIR: ",file)
best=True
save_dir = Path(file).parent
fig, ax = plt.subplots(2, 5, figsize=(24, 8), tight_layout=True)
ax = ax.ravel()
files = list(save_dir.glob("yolov5*.csv"))
assert len(files), f"No yolov5*.csv files found in {save_dir.resolve()}, nothing to plot."
best_pt = []
for f in files:
    try:
        data = pd.read_csv(f)
        index = np.argmax(0.9 * data.values[:, 7] + 0.1 * data.values[:, 6])
        best_pt.append(data.values[index, :])

        s = [x.strip() for x in data.columns]
        x = data.values[:, 0]
        for i, j in enumerate([1, 2, 3, 4, 5, 8, 9, 10, 6, 7 ]):
            y = data.values[:, j]
            # y[y == 0] = np.nan  # don't show zero values
            ax[i].plot(x, y, marker=".", label=f.stem, linewidth=1, markersize=4)
            if best:
                # best
                ax[i].scatter(index, y[index], color="r", label=f"best:{index}", marker="*", linewidth=3)
                # ax[i].set_title(s[j] + f"\n{round(y[index], 5)}")
                # best_pt.append(y[index])
                # ax[i].set_title(s[j])
            else:
                # last
                ax[i].scatter(x[-1], y[-1], color="r", label="last", marker="*", linewidth=3)
                ax[i].set_title(s[j] + f"\n{round(y[-1], 5)}")
            # if j in [8, 9, 10]:  # share train and val loss y axes
            #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
    except Exception as e:
        print(f"Warning: Plotting error for {f}: {e}")

datas = pd.DataFrame(best_pt)
best_all = np.argmax(0.9 * datas.values[:, 7] + 0.1 * datas.values[:, 6])

for i,j in enumerate([1, 2, 3, 4, 5, 8, 9, 10, 6, 7 ]):
    y = datas.values[:, j]
    ax[i].set_title(s[j] + f"\n{round(y[best_all], 5)}")

ax[2].legend()
fig.savefig(save_dir / "results.png", dpi=200)
plt.close()