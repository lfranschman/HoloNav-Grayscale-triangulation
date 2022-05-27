import numpy as np
import pandas as pd

from config import config
from DataAcquisition import DataAcquisition, ACQUISITIONS_HOLOLENS

data = DataAcquisition()
data.load_data(config.get_filename("optical_sphere"))

df = data.acquisitions['probe'].copy()
df["time1"] = df.index
df = df.shift(1)
diff_time = np.array([x.total_seconds() for x in (df.index - df["time1"])[1:]]) * 1000
print(
    f"previous time mean {np.mean(diff_time)} median {np.median(diff_time)} 99.5 percentile {np.percentile(diff_time, 99.5)}")

for acquisition in ACQUISITIONS_HOLOLENS:
    if not data.acquisitions[acquisition].empty:
        data.acquisitions[acquisition].index = data.acquisitions[acquisition].index + pd.Timedelta(
            seconds=config.temporal_shift_hololens)

count = 0
count2 = 0
count3 = 0
nb_frames = 0
diff = []
# for frame_id in range(len(data.acquisitions["ahat_depth_cam_frames"])):
for frame_id in range(76, 799):
    timestamp = data.acquisitions["ahat_depth_cam"].index[frame_id]

    optical_index = data.acquisitions["probe"].index.get_loc(timestamp, method='nearest')
    optical_timestamp = data.acquisitions["probe"].index[optical_index]

    diff.append(abs((timestamp - optical_timestamp).total_seconds()))
    if diff[-1] < 0.02:
        count += 1
    if diff[-1] < 0.025:
        count2 += 1
    if diff[-1] < 0.1:
        count3 += 1
    nb_frames += 1

diff = np.array(diff)

print(f"{count} of {nb_frames}")
print(f"{count2} of {nb_frames}")
print(f"{count3} of {nb_frames}")
print(f"diff mean {np.mean(diff)} median {np.median(diff)} 75 percentile {np.percentile(diff, 75)}")
