import pandas as pd
import os
import time
import math
from mpl_toolkits.mplot3d import axes3d
import matplotlib.lines as mlines
from pathlib import Path
from pycocotools.coco import COCO
import numpy as np
from numpy import genfromtxt
import matplotlib
import matplotlib.pyplot as plt
from ppu_plotter import *
from tempfile import TemporaryFile

matplotlib.use('TkAgg')


def unite_results_data(path_generator, total_df_cache, keyp_coco, use_cache=False):

    total_data_df = np.zeros([25, 18])
    face_distances_df = np.zeros([25, 5000])
    face_distances_index = 0
    outliers = []

    # Ensuring a high value in column 10 to find the min of embeddings distances
    total_data_df[:, 9] = 10

    face_counter = 0
    tn_person_counter = 0
    tn_face_counter = 0

    if os.path.exists(total_df_cache) and use_cache:
        total_data_df = pd.read_pickle(total_df_cache).values
        print(total_data_df)
        face_counter = total_data_df[0, 7]
        tn_person_counter = total_data_df[1, 7]
        tn_face_counter = total_data_df[2, 7]
    else:

        for image_df_path in path_generator:
            image_data_df = pd.read_pickle(str(image_df_path))

            image_df_name = os.path.basename(Path(image_df_path).resolve().stem)
            print("Processing pickle file {}".format(image_df_name))

            name_components = image_df_name.split('_')
            clean_image_id = name_components[0].lstrip('0')

            ann_ids = keyp_coco.getAnnIds(imgIds=int(clean_image_id), catIds=1, iscrowd=None)
            nr_plp = len(ann_ids)
            total_data_df[3, 7] = total_data_df[3, 7] + nr_plp

            #True negative persons
            if nr_plp == 0:
                tn_person_counter = tn_person_counter + 1

            face_gt = 0
            ghost_detected = 0

            for index, row in image_data_df.iterrows():

                if index == 0:
                    face_gt = row['f_score']
                    if face_gt > 0 and nr_plp:
                        face_counter = face_counter + 1

                    if face_gt < 0 and nr_plp == 0:
                        tn_face_counter = tn_face_counter + 1

                if row['f_score'] != -1 and nr_plp:
                    # Euclidean Distance Totals
                    total_data_df[index, 0] = total_data_df[index, 0] + row['f_score']
                    face_distances_df[index, face_distances_index] = row['f_score']
                    total_data_df[index, 9] = row['f_score'] if total_data_df[index, 9] > row['f_score'] else total_data_df[index, 9]

                    # True positive faces
                    if face_gt > 0 and row['f_detection'] > 0:
                        total_data_df[index, 1] = total_data_df[index, 1] + 1
                        # True positive match
                        if face_gt < 1.1:
                            total_data_df[index, 10] = total_data_df[index, 10] + 1

                    # False negative faces
                    if (face_gt > 0 and row['f_detection'] == 0.0) or face_gt > 1.1:
                        total_data_df[index, 2] = total_data_df[index, 2] + 1

                # False positive faces
                if (face_gt < 0 and row['f_detection'] == 1 and nr_plp == 0):
                    total_data_df[index, 3] = total_data_df[index, 3] + 1

                if nr_plp >= row['human_det:']:
                    # True positive persons 1
                    total_data_df[index, 4] = total_data_df[index, 4] + row['human_det:']
                    # False negative persons
                    total_data_df[index, 5] = total_data_df[index, 5] + nr_plp - row['human_det:']
                else:
                    # True positive persons 2
                    total_data_df[index, 4] = total_data_df[index, 4] + nr_plp
                    # False positive persons
                    total_data_df[index, 6] = total_data_df[index, 6] + row['human_det:'] - nr_plp
                    ghost_detected = 1

                # TP Occupancy
                if nr_plp and row['human_det:']:
                    total_data_df[index, 15] = total_data_df[index, 15] + 1

                # FP Occupancy
                if nr_plp == 0 and row['human_det:']:
                    total_data_df[index, 16] = total_data_df[index, 16] + 1

                # FN Occupancy
                if nr_plp and row['human_det:'] == 0:
                    total_data_df[index, 17] = total_data_df[index, 17] + 1

                # Attacker and User Succeeds
                if (0 < row['f_score'] < 1.1 and face_gt > 0 and row['f_detection'] > 0.0) and nr_plp == row['human_det:'] and row['human_det:'] > 0:
                    total_data_df[index, 11] = total_data_df[index, 11] + 1
                # Attacker Succeeds and User Fails
                if 0 < row['f_score'] < 1.1 and face_gt > 0 and row['f_detection'] > 0.0 and nr_plp != row['human_det:']:
                    total_data_df[index, 12] = total_data_df[index, 12] + 1
                # Attacker Fails and User Succeeds
                if (row['f_score'] > 1.1 or row['f_detection'] == 0.0) and face_gt > 0 and nr_plp == row['human_det:'] and row['human_det:'] > 0:
                    total_data_df[index, 13] = total_data_df[index, 13] + 1
                # Attacker Fails and User Fails
                if (row['f_score'] > 1.1 or row['f_detection'] == 0.0) and face_gt > 0 and nr_plp != row['human_det:']:
                    total_data_df[index, 14] = total_data_df[index, 14] + 1

            if 0 < image_data_df['f_score'][24] < 0.75 and nr_plp:
                outliers.append(clean_image_id)

            face_distances_index = face_distances_index + 1
            if ghost_detected:
                total_data_df[4, 7] = total_data_df[4, 7] + 1

    total_data_df[0, 7] = face_counter
    total_data_df[1, 7] = tn_person_counter
    total_data_df[2, 7] = tn_face_counter

    #Adding standard deviation. This will be empty when the data is loaded from the cache
    clean_data = np.delete(face_distances_df, np.where(~face_distances_df.any(axis=0))[0], axis=1)
    if clean_data.shape[1] != 0:
        total_data_df[:, 8] = np.std(clean_data, axis=1)

    c = ['Distance', 'F TP', 'F FN', 'F FP', 'H TP', 'H FN', 'H FP', 'Aux', 'Std', 'Min Distances', 'FM TP', 'US AS', 'UF AS', 'US AF', 'UF AF', 'TP Occupancy', 'FP Occupancy', 'FN Occupancy']
    temp_df = pd.DataFrame(data=total_data_df, columns=c)
    pd.DataFrame(clean_data).to_excel(total_df_cache+"_all_distances.xlsx")
    temp_df.to_pickle(total_df_cache+ ".pkl")
    temp_df.to_excel(total_df_cache+".xlsx")

    print(outliers)

    print(temp_df)

    return total_data_df


def generate_plots(df_data, energy_traces, x_label):

    fd_embeddings_distance = df_data[:, 0] / df_data[0, 7]
    fd_embeddings_std = df_data[:, 8]

    fd_precision = df_data[:, 1] / (df_data[:, 1] + df_data[:, 3])
    fd_recall = df_data[:, 1] / (df_data[:, 1] + df_data[:, 2])
    fd_truth_recall = df_data[:, 10] / (df_data[:, 10] + df_data[:, 1])
    fd_f1_measure = (2 * fd_precision * fd_recall) / (fd_precision + fd_recall)
    fd_fpr = df_data[:, 3] / (df_data[2, 7] + df_data[:, 3])

    hd_precision = df_data[:, 4] / (df_data[:, 4] + df_data[:, 6])
    hd_recall = df_data[:, 4] / (df_data[:, 4] + df_data[:, 5])
    hd_f1_measure = (2 * hd_precision * hd_recall) / (hd_precision + hd_recall)
    hd_fpr = df_data[:, 6] / (df_data[1, 7] + df_data[:, 6])

    af_us_from_user_s_rate = df_data[:, 13] / (df_data[:, 11] + df_data[:, 13])
    us_af_from_a_fails_rate = df_data[:, 13] / (df_data[:, 14] + df_data[:, 13])
    as_uf_rate = df_data[:, 12] / (df_data[:, 11] + df_data[:, 12])


    x_holder = [i for i in range(1, len(fd_embeddings_distance)+1)]
    y_cut = [1.1 for i in range(1, len(fd_embeddings_distance)+1)]

    plot_face_embeddings_data(df_data, energy_traces, fd_embeddings_distance, fd_embeddings_std, x_holder, x_label,
                              y_cut)

    general_plot(energy_traces, fd_f1_measure, fd_fpr, fd_precision, fd_recall, fd_truth_recall, hd_f1_measure, hd_fpr, hd_precision, hd_recall,
                 x_holder, x_label)

    roc_plot(energy_traces, fd_fpr, fd_recall, hd_fpr, hd_recall, x_holder, x_label)

    raw_attack_rate_metric(af_us_from_user_s_rate, as_uf_rate, energy_traces, us_af_from_a_fails_rate, x_holder,
                           x_label)


ins_annFile = './coco_annotations_db/instances_val2017.json'
key_points_annFile = './coco_annotations_db/person_keypoints2017.json'

#Loading coco labels
coco_true_labels = COCO(ins_annFile)

df_avg_pickles_paths = Path('F:/results').glob("*_avg_data.pkl")
df_gaussian_pickles_paths = Path('F:/results').glob("*_gaussian_data.pkl")
df_median_pickles_paths = Path('F:/results').glob("*_median_data.pkl")
df_bilateral_pickles_paths = Path('F:/results').glob("*_bilateralFiltering_data.pkl")

full_avg_cache = 'F:/results/full_avg_results'
full_gau_cache = 'F:/results/full_gaussian_results'
full_med_cache = 'F:/results/full_med_results'
full_bila_cache = 'F:/results/full_bila_results'

avg_time_trace = genfromtxt('./power_traces/blur_times_avg_box.csv', delimiter=',')
gaussian_time_trace = genfromtxt('./power_traces/blur_times_gaussian_box.csv', delimiter=',')
bilateral_time_trace = genfromtxt('./power_traces/blur_times_bilateral_box.csv', delimiter=',')
median_time_trace = genfromtxt('./power_traces/blur_times_median_box.csv', delimiter=',')

avg_energy_trace = (avg_time_trace/100) * 5.16 * 0.460
gaussian_energy_trace = (gaussian_time_trace/100) * 5.16 * 0.760
bilateral_energy_trace = (bilateral_time_trace/100) * 5.16 * 0.742
median_energy_trace = (median_time_trace/100) * 5.16 * 0.460

avg_energy = [avg_energy_trace[2] * i for i in range(1, 25 + 1)]
gaussian_energy = [gaussian_energy_trace[2] * i for i in range(1, 25 + 1)]
bilateral_energy = [bilateral_energy_trace[2] * i for i in range(1, 25 + 1)]
median_energy = [median_energy_trace[2] * i for i in range(1, 25 + 1)]

#Generating totals and charts
#total_avg_df = unite_results_data(df_avg_pickles_paths, full_avg_cache, coco_true_labels, False)
#generate_plots(total_avg_df, avg_energy, 'Iterations of AVG Blur')

#total_gaussian_df = unite_results_data(df_gaussian_pickles_paths, full_gau_cache, coco_true_labels, False)
#generate_plots(total_gaussian_df, gaussian_energy, 'Iterations of Gaussian Blur')

total_med_df = unite_results_data(df_median_pickles_paths, full_med_cache, coco_true_labels, False)
#generate_plots(total_med_df, bilateral_energy, 'Iterations of Median Blur')

total_bila_df = unite_results_data(df_bilateral_pickles_paths, full_bila_cache, coco_true_labels, False)
#generate_plots(total_bila_df, median_energy, 'Iterations of Bilateral Filter Blur')


#ax = plt.axes(projection='3d')
# s = ax.scatter3D(total_avg_data_df.iloc[:, 0], total_avg_data_df.iloc[:, 2], power, c=power, cmap='Greens')
#
# #ax.plot_trisurf(fm_scores, hd_scores, power, cmap='viridis', edgecolor='none', label='PPU Plane')
# ax.set_title('Power x Privacy x Utility')
#
# ax.set_xlabel('Privacy', fontsize=20)
# ax.set_ylabel('Utility', fontsize=20)
# ax.set_zlabel('Power', fontsize=20)

