import pandas as pd
import os
import time
import experiment_functions as ef
from mpl_toolkits.mplot3d import axes3d
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import pylab


ins_annFile = './coco_annotations_db/instances_val2017.json'
key_points_annFile = './coco_annotations_db/person_keypoints2017.json'

#ins_coco = COCO(ins_annFile)
keyp_coco = COCO(ins_annFile)

#catIds = ins_coco.getCatIds(catNms=['person'])
#imgIds = ins_coco.getImgIds(catIds=catIds)

df_pickles_paths = Path('./results').glob("*.pkl")


df_counter = 0
total_data_df = np.zeros([25, 7])

start = time.time()

for image_df_path in df_pickles_paths:

    df_counter = df_counter + 1
    full_df = pd.read_pickle(str(image_df_path))

    image_df_name = os.path.basename(Path(image_df_path).resolve().stem)
    name_components = image_df_name.split('_')
    clean_image_id = name_components[0].lstrip('0')

    annIds = keyp_coco.getAnnIds(imgIds=int(clean_image_id), catIds=1, iscrowd=None)
    number_of_persons = len(annIds)

    face_gt = 0
    face_counter = 0
    for index, row in full_df.iterrows():

        if index == 0:
            face_gt = row['f_score']

        if row['f_score'] != -1:
            #Euclidean Distance Totals
            total_data_df[index][0] = total_data_df[index][0] + row['f_score']
            face_counter = face_counter + 1
            #True positive faces
            if face_gt > 0 and row['f_detection'] > 0:
                total_data_df[index][1] = total_data_df[index][1] + 1

            #False negative faces
            if face_gt > 0 and row['f_detection'] == 0:
                total_data_df[index][2] = total_data_df[index][2] + 1

        #False positive faces
        if face_gt < 0 and row['f_detection'] == 1:
            total_data_df[index][3] = total_data_df[index][3] + 1

        if number_of_persons >= row['human_det:']:
            # True positive persons 1
            total_data_df[index][4] = total_data_df[index][4] + row['human_det:']
            # False negative persons
            total_data_df[index][5] = total_data_df[index][5] + number_of_persons - row['human_det:']
        else:
            # True positive persons 2
            total_data_df[index][4] = total_data_df[index][4] + number_of_persons

            # False positive persons
            total_data_df[index][6] = total_data_df[index][6] + row['human_det:'] - number_of_persons

        print(total_data_df)


end = time.time()
print("Time elapsed: {} seconds".format(end - start))


fig = plt.figure()

power = [float(i*0.33) for i in range(10)]

ax = plt.axes(projection='3d')

s = ax.scatter3D(total_data_df.iloc[:, 0], total_data_df.iloc[:, 2], power, c=power, cmap='Greens')

#ax.plot_trisurf(fm_scores, hd_scores, power, cmap='viridis', edgecolor='none', label='PPU Plane')
ax.set_title('Power x Privacy x Utility')

ax.set_xlabel('Privacy', fontsize=20)
ax.set_ylabel('Utility', fontsize=20)
ax.set_zlabel('Power', fontsize=20)

plt.show()