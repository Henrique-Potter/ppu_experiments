import pandas as pd
import os
import time
import experiment_functions as ef
from pathlib import Path
import numpy as np


face_model_path = "face_id_models/20170512-110547.pb"
hd_model_path = "object_detection_models/frozen_inference_graph.pb"
hd_threshold = 0.7
iterations = 10
map_face_detection = 'y'
blur_kernel = "avg"
kernel_size = 5
use_cache = 'n'

images = Path('D:\\Downloads\\train2017').glob("*.jpg")

experiments = ef.BlurExperiments(face_model_path, hd_model_path)

for image in images:

    if np.random.randint(0, 100) > 1:
        print('Skipped {}'.format(image))
        continue

    image_name = Path(image).resolve().stem
    cache_pickle_name = "./results/{}_{}_data.pkl".format(image_name, blur_kernel)

    if use_cache is not 'y':

        print("\n--------Processing img number:{}----------\n".format(image_name))
        start = time.time()

        full_df = experiments.blur_iter_experiment(str(image),
                                                   str(image),
                                                   iterations,
                                                   hd_threshold,
                                                   map_face_detection,
                                                   blur_kernel,
                                                   kernel_size, True)

        end = time.time()

        print("Time elapsed: {} seconds".format(end - start))
        print("Saving data to Pickle format")
        image_name = Path(image).resolve().stem
        full_df.to_pickle("./results/{}_{}_data.pkl".format(image_name, blur_kernel))

        print("\n--------Processing finished----------\n")

    elif os.path.exists(cache_pickle_name):
            image_name = Path(image).resolve().stem
            pkl_data_path = "./results/{}_data.pkl".format(image_name)
            full_df = pd.read_pickle(pkl_data_path)





#fig = plt.figure()

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

