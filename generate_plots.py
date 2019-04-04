import human_detection as hd
import experiment_functions as hdu
import face_match as fm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import pandas as pd
import cv2 as cv
import experiment_functions as ef
from pathlib import Path


face_model_path = "face_id_models/20170512-110547.pb"
hd_model_path = "object_detection_models/frozen_inference_graph.pb"
hd_threshold = 0.7
iterations = 50
map_face_detection = True
blur_kernel = "avg"
kernel_size = 5

images = Path('./images').glob("*.jpg")

experiments = ef.BlurExperiments(face_model_path, hd_model_path)

for image in images:
    full_df = experiments.blur_iter_experiment(str(image),
                                               str(image),
                                               iterations,
                                               hd_threshold,
                                               map_face_detection,
                                               blur_kernel,
                                               kernel_size, True)

    fig = plt.figure()

    if full_df is None:
        image_name = Path(image).resolve().stem
        pkl_data_path = "./results/{}_data.pkl".format(image_name)

        full_df = pd.read_pickle(pkl_data_path)

# power = [float(i*0.33) for i in full_df.iloc[:, 0]]

# ax = plt.axes(projection='3d')
#
# persons_detected = np.array(full_df.iloc[:, 3])
#
# s = ax.scatter3D(full_df.iloc[:, 1], full_df.iloc[:, 3], power, c=power, cmap='Greens')
#
# #ax.plot_trisurf(fm_scores, hd_scores, power, cmap='viridis', edgecolor='none', label='PPU Plane')
# ax.set_title('Power x Privacy x Utility')
#
# ax.set_xlabel('Privacy', fontsize=20)
# ax.set_ylabel('Utility', fontsize=20)
# ax.set_zlabel('Power', fontsize=20)
#
# plt.show()

