import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from scipy import signal
from scipy.signal import find_peaks
from sklearn.decomposition import PCA

plt.style.use("/Users/sunyuyao/Documents/PhDinGermany/courses/neural data science/CL1/matplotlib_style.txt")
fs = 30000.0  # sampling rate of the signal in Hz
dt = 1 / fs
cols = ["Ch1", "Ch2", "Ch3", "Ch4"]
x = pd.read_csv("/Users/sunyuyao/Documents/PhDinGermany/courses/neural data science/CL1/data/nds_cl_1.csv", header=0,
                names=cols)
x.describe()


def filter_signal(
        x: pd.DataFrame, fs: float, low: float, high: float, order: int = 3
) -> pd.DataFrame:
    from scipy.signal import butter, filtfilt

    # Define the filter
    # Other Low-cut Filter: Bessel,Elliptic )
    nyquist = 0.5 * fs
    low_cutoff = low / nyquist
    high_cutoff = high / nyquist
    b, a = butter(order, [low_cutoff, high_cutoff], btype="band")

    # Apply the filter to each channel
    y: DataFrame = pd.DataFrame(index=x.index, columns=x.columns)
    for col in x.columns:
        y[col] = filtfilt(b, a, x[col])

    return y
    pass


xf = filter_signal(x, fs, 500, 4000)
mosaic = [
    ["raw: Ch1", "filtered: Ch1"],
    ["raw: Ch2", "filtered: Ch2"],
    ["raw: Ch3", "filtered: Ch3"],
    ["raw: Ch4", "filtered: Ch4"],
]
fig, ax = plt.subplot_mosaic(
    mosaic=mosaic, figsize=(8, 6), layout="constrained", dpi=100
)

for i, m in enumerate(mosaic):
    raw = m[0]
    filtered = m[1]

    # Plot raw signal
    ax[raw].plot(x.index * dt, x[raw.split(": ")[1]], color='blue')  # Plot raw signal
    ax[raw].set_xlim((0, 3))
    ax[raw].set_ylim((-1000, 1000))
    ax[raw].set_ylabel("voltage")
    ax[raw].set_title(m[0], loc="left")
    if i != 3:
        ax[raw].set_xticklabels([])
    else:
        ax[raw].set_xlabel("time (s)")

    # Plot filtered signal
    ax[filtered].plot(x.index * dt, xf[raw.split(": ")[1]], color='red')  # Plot filtered signal
    ax[filtered].set_xlim((0, 3))
    ax[filtered].set_ylim((-400, 250))
    ax[filtered].set_title(m[1], loc="left")
    if i != 3:
        ax[filtered].set_xticklabels([])
    else:
        ax[filtered].set_xlabel("time (s)")


plt.show()

def detect_spikes(
    x: np.ndarray, fs: float, N: int = 5, lockout: int = 10) -> tuple[np.ndarray, np.ndarray, np.float64]:
    # Compute the robust standard deviation
    robust_std = np.median(np.abs(x - np.median(x))) / 0.6745

    # Calculate the threshold
    threshold = -N * robust_std

    # Find all local minima
    local_minima = find_peaks(-x.values.flatten())[0]

    return local_minima, local_minima / fs * 1000, threshold

mosaic = [
        ["Ch1"],
        ["Ch2"],
        ["Ch3"],
        ["Ch4"],
]
fig, ax = plt.subplot_mosaic(
        mosaic=mosaic, figsize=(8, 8), layout="constrained", dpi=100
    )
# Call the detect_spikes function to get local minima and the threshold
spikes, t, threshold = detect_spikes(x, fs)

# Plot the threshold and detected spikes
for i, col in enumerate(cols):
    # Set the axis labels and title
    ax[col].set_ylim((-400, 400))
    ax[col].set_xlim((0.025, 0.075))
    ax[col].set_ylabel("voltage")
    ax[col].set_title(col, loc="left")
    # Plot the threshold line
    ax[col].axhline(y=threshold, color='red', linestyle='--', label='Threshold')
    # Mark the detected spikes
    if len(x.values) > 0 and len(spikes) > 0:
        spikes_in_bounds = spikes[spikes < len(x)]
        ax[col].scatter(t[spikes_in_bounds], x.values[spikes_in_bounds, i], color='green', label='Detected Spikes')

    # Set the x-axis label
    if col != "Ch4":
        ax[col].set_xticklabels([])
    else:
        ax[col].set_xlabel("time (s)")

plt.show()
