import os
import pandas as pd
import numpy as np


trigger_data = None
trigger_data2 = None

if os.path.exists("trigger_metrics.csv"):
    trigger_data = pd.read_csv("trigger_metrics.csv", index_col=False).values

ignored_trigger=2
target_trigger=1
trigger_data = trigger_data[trigger_data[:, 0] != ignored_trigger, :]

trigger_data = trigger_data[trigger_data[:, 1].argsort()]

tp_s_trigger = 0
fp_s_trigger = 0
fn_s_trigger = 0

total_cam_triggers = 0
total_sen_triggers = 0

log_size = trigger_data.shape[0]
time_window = 3

for i in range(log_size):

    b_distance = 6
    a_distance = 6

    if trigger_data[i, 0] == 0:
        total_cam_triggers += 1
        if i - 1 >= 0 and trigger_data[i - 1, 0] == target_trigger:
            b_distance = abs(trigger_data[i - 1, 1] - trigger_data[i, 1])

        if i + 1 < log_size and trigger_data[i + 1, 0] == target_trigger:
            a_distance = abs(trigger_data[i + 1, 1] - trigger_data[i, 1])

        if b_distance <= time_window or a_distance < time_window:
            tp_s_trigger += 1
        else:
            fn_s_trigger += 1

    else:
        total_sen_triggers += 1
        if i - 1 >= 0 and trigger_data[i - 1, 0] == 0:
            b_distance = abs(trigger_data[i - 1, 1] - trigger_data[i, 1])

        if i + 1 < log_size and trigger_data[i + 1, 0] == 0:
            a_distance = abs(trigger_data[i + 1, 1] - trigger_data[i, 1])

        if b_distance > time_window and a_distance > time_window:
            fp_s_trigger += 1


print("Total cam triggers: {}".format(total_cam_triggers))
print("Total sen triggers: {}".format(total_sen_triggers))

print("TP: {}".format(tp_s_trigger))
print("FP: {}".format(fp_s_trigger))
print("FN: {}".format(fn_s_trigger))


