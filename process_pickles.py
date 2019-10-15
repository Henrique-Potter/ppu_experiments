import pandas as pd
import os
from pathlib import Path
from pycocotools.coco import COCO
import numpy as np
from numpy import genfromtxt
import matplotlib
from matplotlib import pyplot
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from ppu_plotter import *

matplotlib.use('TkAgg')


def unite_results_data(path_generator, total_df_cache, keyp_coco, use_cache=False):

    if os.path.exists(total_df_cache+".pkl") and use_cache:
        total_data_df = pd.read_pickle(str(total_df_cache+".pkl"))
        return total_data_df

    total_data_np = np.zeros([15, 150])
    general_data_np = np.zeros([15, 1])
    face_distances_df = np.zeros([25, 5000])
    face_distances_index = 0
    outliers = []

    gt_face_counter = 0
    gtn_face_counter = 0

    total_df_header = generate_headers()

    # Ensuring a high value in column 10 to find the min of embeddings distances
    # total_data_df[:, 9] = 10

    for image_df_path in path_generator:

        # -- Load data for pickle file
        image_data_df = pd.read_pickle(str(image_df_path))
        image_df_name = os.path.basename(Path(image_df_path).resolve().stem)
        print("Processing pickle file {}".format(image_df_name))

        # -- Finding ground truth for person classifier
        name_components = image_df_name.split('_')
        clean_image_id = name_components[0].lstrip('0')

        ann_ids = keyp_coco.getAnnIds(imgIds=int(clean_image_id), catIds=1, iscrowd=None)
        gt_nr_plp = len(ann_ids)

        # -- Preparing data
        dfs = np.split(image_data_df, [15], axis=1)
        face_df = dfs[0]
        person_df = dfs[1]
        img_all_boxes_data_np = None

        for column in range(15):
            ghost_detected = 0

            face_parcial_np = np.zeros([15, 7])
            person_parcial_np = np.zeros([15, 3])

            for row in range(15):

                f_values = face_df.iloc[row, column].decode().split(" ")
                f_dist = float(f_values[0])
                f_faces_tp = 0
                f_faces_found = 0

                try:
                    f_faces_tp = int(f_values[1])
                    f_faces_found = int(f_values[2])

                except:
                    pass

                if row == 0 and f_faces_tp:
                    gt_face_counter = gt_face_counter + 1
                elif row == 0 and not f_faces_tp and not gt_nr_plp:
                    gtn_face_counter = gtn_face_counter + 1

                if f_faces_tp and gt_nr_plp:
                    face_parcial_np[row, 0] = f_dist

                    # True positive faces
                    if f_faces_found:
                        face_parcial_np[row, 1] = f_faces_found
                        # True positive match
                        if f_dist < 1.1:
                            face_parcial_np[row, 2] += 1

                    # False negative faces
                    if f_faces_found < f_faces_tp:
                        face_parcial_np[row, 3] += f_faces_tp - f_faces_found

                    # False negative match
                    if not f_faces_found or f_dist > 1.1:
                        face_parcial_np[row, 4] += 1
                elif f_faces_found:
                    # False positive faces
                    face_parcial_np[row, 5] += 1

                # False positive match
                if f_faces_tp == 0 and (f_faces_found or 0 < f_dist < 1.1) and gt_nr_plp == 0:
                    face_parcial_np[row, 6] += 1

                if gt_nr_plp >= person_df.iloc[row, column]:
                    # True positive persons 1
                    person_parcial_np[row, 0] += person_df.iloc[row, column]
                    # False negative persons
                    person_parcial_np[row, 1] += gt_nr_plp - person_df.iloc[row, column]
                else:
                    # True positive persons 2
                    person_parcial_np[row, 0] += gt_nr_plp
                    # False positive persons
                    person_parcial_np[row, 2] += person_df.iloc[row, column] - gt_nr_plp
                    ghost_detected += 1

            if img_all_boxes_data_np is None:
                img_all_boxes_data_np = np.concatenate((face_parcial_np, person_parcial_np),
                                                       axis=1)
            else:
                img_all_boxes_data_np = np.concatenate((img_all_boxes_data_np, face_parcial_np, person_parcial_np),
                                                       axis=1)

        total_data_np = total_data_np + img_all_boxes_data_np

    general_data_np[0] = gt_face_counter // 15
    general_data_np[1] = gtn_face_counter // 15

    total_data_np = np.concatenate((general_data_np,
                                    total_data_np),
                                   axis=1)

    # #Adding standard deviation. This will be empty when the data is loaded from the cache
    # clean_data = np.delete(face_distances_df, np.where(~face_distances_df.any(axis=0))[0], axis=1)
    # if clean_data.shape[1] != 0:
    #     total_data_df[:, 8] = np.std(clean_data, axis=1)

    total_df_header = ['General Info'] + total_df_header
    total_data_df = pd.DataFrame(data=total_data_np, columns=total_df_header)
    # pd.DataFrame(clean_data).to_excel(total_df_cache+"_all_distances.xlsx")
    total_data_df.to_pickle(total_df_cache+".pkl")
    total_data_df.to_excel(total_df_cache+".xlsx")

    print(total_data_df)

    return total_data_df


def generate_headers():

    faces_dt_h = ['{}x{} Dist',
                  '{}x{} Face TP',
                  '{}x{} Face TM',
                  '{}x{} Face FN',
                  '{}x{} Face FNM',
                  '{}x{} Face FP',
                  '{}x{} Face FPM']

    persons_dt_h = ['{}x{} Person TP', '{}x{} Person FN', '{}x{} Person FP']
    final_header = []

    for box_size in range(3, 32, 2):
        l1 = []
        l2 = []
        for fh in faces_dt_h:
            l1.append(fh.format(box_size, box_size))
        for ph in persons_dt_h:
            l2.append(ph.format(box_size, box_size))

        final_header += l1
        final_header += l2

    return final_header


def calculate_f1_matrix(total_df):
    fm_f1_data = np.zeros([15, 15])
    hd_f1_data = np.zeros([15, 15])
    dist_data = np.zeros([15, 15])
    total_faces = total_df.iloc[0, 0]
    row_count = 0
    for row in total_df.iterrows():
        temp = 3
        for i in range(15):
            # Skip 10 by 10 columns to get the correct TP/FP/FN
            index_fix = i * 10

            print(index_fix)
            fm_precision = row[1][2 + index_fix] / (row[1][2 + index_fix] + row[1][6 + index_fix])
            pd_precision = row[1][8 + index_fix] / (row[1][8 + index_fix] + row[1][10 + index_fix])
            fm_recall = row[1][2 + index_fix] / (row[1][2 + index_fix] + row[1][4 + index_fix])
            pd_recall = row[1][8 + index_fix] / (row[1][8 + index_fix] + row[1][9 + index_fix])

            fm_f1 = 2 * (fm_precision * fm_recall) / (fm_precision + fm_recall)
            pd_f1 = 2 * (pd_precision * pd_recall) / (pd_precision + pd_recall)

            # Skip 2 columns to add face match and person F1 in the same matrix
            fm_f1_data[row_count, i] = fm_f1
            hd_f1_data[row_count, i] = pd_f1
            # Distance
            dist_data[row_count, i] = row[1][1 + index_fix] / total_faces

        row_count += 1

    return np.nan_to_num(fm_f1_data), np.nan_to_num(hd_f1_data), dist_data


def normalize(x):
    x = np.asarray(x)
    return (x - x.min()) / (np.ptp(x))


def plot_best_region(fm_f1_data, hd_f1_data, avg_time_trace, dist_data):

    from matplotlib import cm

    fm_f1_data = fm_f1_data.transpose()
    hd_f1_data = hd_f1_data.transpose()
    dist_data = dist_data.transpose()
    dist_map = dist_data >= 1.1

    best_for_user_map = fm_f1_data < hd_f1_data

    allowed_user_map = np.multiply(dist_map, best_for_user_map)
    best_for_user_values = np.multiply(allowed_user_map, hd_f1_data)

    ax = plt.axes(projection='3d')

    iterations = np.linspace(1, 15, num=15)
    boxes = np.linspace(1, 15, num=15)
    X, Y = np.meshgrid(iterations, boxes)

    heat_map = normalize(Y * avg_time_trace[:15]).transpose()

    s = ax.plot_surface(X, Y, np.divide(best_for_user_values, heat_map), facecolors=cm.Reds(heat_map), alpha=0.9)
    #s = ax.plot_surface(X, Y, np.divide(best_for_user_values, heat_map), facecolors=cm.Reds(heat_map), alpha=0.9)

    # Cut in human det
    # range_hd_data = np.multiply(hd_f1_data > 0.3, hd_f1_data)
    # best_solution = np.multiply(range_hd_data != 0, heat_map)
    # s = ax.scatter(X, Y, np.multiply(range_hd_data, dist_map), alpha=0.9)

    #cset = ax.scatter(X, Y, np.multiply(best_for_user_values.transpose(), heat_map.transpose()),)
    #cset2 = ax.scatter(X, Y, fm_f1_data)
    # cset = ax.contour(X, Y, fm_f1_data, zdir='y', offset=0)

    #s2 = ax.plot_wireframe(X, Y, fm_f1_data.transpose(), alpha=0.8)

    ax.set_title('Iteration x Box Size x F1')
    yticks_text = ['{}x{}'.format(i, i) for i in range(3, 33, 2)]
    xticks_text = [i for i in range(1, 16)]

    ax.set_xlabel('Iteration', fontsize=15, labelpad=10)
    ax.set_xticks(iterations)
    ax.set_xticklabels(xticks_text)

    ax.set_ylabel('Box Size', fontsize=15, labelpad=25)
    ax.set_yticks(iterations)
    ax.set_yticklabels(yticks_text, rotation=45)

    ax.set_zlabel('F1', fontsize=15)

    plt.show()
    pass


def regression_3d(fm_f1_data, hd_f1_data, avg_time_trace, dist_data):

    from matplotlib import cm
    import statsmodels.api as sm
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    iterations = np.linspace(1, 15, num=15)
    boxes = np.linspace(1, 15, num=15)
    iterations_gmesh, boxes_gmesh = np.meshgrid(iterations, boxes)

    flat_data = np.zeros([225, 4])
    count = 0

    for i in range(15):
        for j in range(15):
            flat_data[count, 0] = iterations_gmesh[i, j]
            flat_data[count, 1] = boxes_gmesh[i, j]
            flat_data[count, 2] = fm_f1_data[i, j]
            flat_data[count, 3] = hd_f1_data[i, j]
            count += 1

    features = flat_data[:, 0:2]
    poly_feats = PolynomialFeatures(degree=11)
    features_poly = poly_feats.fit_transform(features)

    I = flat_data[:, 0]
    B = flat_data[:, 1]
    face_f1 = flat_data[:, 2]
    hd_f1 = flat_data[:, 3]

    # reg = LinearRegression().fit(S_opt, f1)
    # c, _, _, _ = np.linalg.lstsq(S_opt, f1)
    fc_regressor_OLS = sm.OLS(face_f1, features_poly).fit()
    hd_regressor_OLS = sm.OLS(hd_f1, features_poly).fit()

    print(fc_regressor_OLS.summary())
    print(hd_regressor_OLS.summary())

    # Coefficients times polynomial features (e.g., temp = np.dot(S_opt, regressor_OLS.params) )
    fc_zp = fc_regressor_OLS.predict(features_poly)
    hd_zp = hd_regressor_OLS.predict(features_poly)

    # Mapping input space
    x_values_range = np.linspace(min(flat_data[:, 0]), max(flat_data[:, 0]), 15)
    y_values_range = np.linspace(min(flat_data[:, 1]), max(flat_data[:, 1]), 15)
    #power_values_range = np.linspace(min(avg_time_trace[:15]), max(avg_time_trace[:15]))

    heat_map = np.linspace(min(flat_data[:, 1]), max(flat_data[:, 1]))

    # Creating input space grid for plotting (Combining all possible x and y inputs)
    x_axis_map, y_axis_map = np.meshgrid(x_values_range, y_values_range)

    # Mapping to grid values in Z
    fc_outputs = griddata(flat_data[:, 0:2], fc_zp, (x_axis_map, y_axis_map), method='linear')
    hd_outputs = griddata(flat_data[:, 0:2], hd_zp, (x_axis_map, y_axis_map), method='linear')

    fig = pyplot.figure()
    ax = Axes3D(fig)

    heat_map = normalize(y_axis_map*avg_time_trace[:15])
    surf = ax.plot_surface(x_axis_map, y_axis_map, fc_outputs, facecolors=cm.Reds(heat_map.transpose()), antialiased=True)
    surf = ax.plot_wireframe(x_axis_map, y_axis_map, hd_outputs, antialiased=True)
    surf = ax.plot_wireframe(x_axis_map, y_axis_map, dist_data, color='red', antialiased=True)

    #ax.contour(x_axis_map, y_axis_map, abs(fc_outputs - hd_outputs))

    #ax.scatter(I, B, face_f1)
    ax.set_zlim3d(-0.2, 1.2)
    colorscale = cm.ScalarMappable(cmap=cm.Reds)
    colorscale.set_array(face_f1)
    fig.colorbar(colorscale, shrink=0.7)

    ax.set_title('Iteration x Box Size x F1')
    yticks_text = ['{}x{}'.format(i, i) for i in range(3, 33, 2)]
    xticks_text = [i for i in range(1, 16)]

    ax.set_xlabel('Iteration', fontsize=15, labelpad=10)
    ax.set_xticks(iterations)
    ax.set_xticklabels(xticks_text)

    ax.set_ylabel('Box Size', fontsize=15, labelpad=25)
    ax.set_yticks(iterations)
    ax.set_yticklabels(yticks_text, rotation=45)

    ax.set_zlabel('F1(Precision/Recall)', fontsize=15)

    pyplot.show()


def plot_3d(fm_f1_data, hd_f1_data, avg_time_trace):

    from matplotlib import cm

    ax = plt.axes(projection='3d')

    iterations = np.linspace(1, 15, num=15)
    boxes = np.linspace(1, 15, num=15)
    X, Y = np.meshgrid(iterations, boxes)

    heat_map = normalize(Y*avg_time_trace[:15])

    s = ax.plot_surface(X, Y, fm_f1_data, facecolors=cm.Reds(heat_map.transpose()), alpha=0.9)
    cset = ax.contour(X, Y, fm_f1_data, zdir='z', offset=0)
    # cset = ax.contour(X, Y, fm_f1_data, zdir='x', offset=0)
    # cset = ax.contour(X, Y, fm_f1_data, zdir='y', offset=0)

    s2 = ax.plot_surface(X, Y, hd_f1_data, facecolors=cm.Oranges(heat_map.transpose()), alpha=0.8)

    ax.set_title('Iteration x Box Size x F1')
    yticks_text = ['{}x{}'.format(i, i) for i in range(3, 33, 2)]
    xticks_text = [i for i in range(1, 16)]

    ax.set_xlabel('Iteration', fontsize=15, labelpad=10)
    ax.set_xticks(iterations)
    ax.set_xticklabels(xticks_text)

    ax.set_ylabel('Box Size', fontsize=15, labelpad=25)
    ax.set_yticks(iterations)
    ax.set_yticklabels(yticks_text, rotation=45)

    ax.set_zlabel('F1(Precision/Recall)', fontsize=15)

    plt.show()
    pass


def process_pickles():

    ins_annFile = './coco_annotations_db/instances_val2017.json'
    key_points_annFile = './coco_annotations_db/person_keypoints2017.json'
    # Loading coco labels
    coco_true_labels = COCO(ins_annFile)
    df_avg_pickles_paths = Path('F:/results').glob("*[0-9]_avg_data.pkl")

    df_gaussian_pickles_paths = Path('F:/results').glob("*[0-9]_gaussian_data.pkl")
    df_median_pickles_paths = Path('F:/results').glob("*_median_data.pkl")
    df_bilateral_pickles_paths = Path('F:/results').glob("*_bilateralFiltering_data.pkl")

    full_avg_cache = 'F:/results/full_avg_results'
    f1_fc_avg_cache = 'F:/results/f1_fm_avg_results'
    hd_fc_avg_cache = 'F:/results/f1_hd_avg_results'
    full_gau_cache = 'F:/results/full_gaussian_results'
    full_med_cache = 'F:/results/full_med_results'
    full_bila_cache = 'F:/results/full_bila_results'

    avg_time_trace = genfromtxt('./power_traces/blur_times_avg_box.csv', delimiter=',')
    gaussian_time_trace = genfromtxt('./power_traces/blur_times_gaussian_box.csv', delimiter=',')
    bilateral_time_trace = genfromtxt('./power_traces/blur_times_bilateral_box.csv', delimiter=',')
    median_time_trace = genfromtxt('./power_traces/blur_times_median_box.csv', delimiter=',')

    # Generating totals and charts
    total_gaussian_df = unite_results_data(df_avg_pickles_paths, full_avg_cache, coco_true_labels, True)

    fm_f1_data, hd_f1_data, dist_data = calculate_f1_matrix(total_gaussian_df)

    pd.DataFrame(fm_f1_data).to_excel(f1_fc_avg_cache+".xlsx")
    pd.DataFrame(hd_f1_data).to_excel(hd_fc_avg_cache+".xlsx")
    pd.DataFrame(dist_data).to_excel(hd_fc_avg_cache + "dist.xlsx")

    plot_best_region(fm_f1_data, hd_f1_data, avg_time_trace, dist_data)
    #regression_3d(fm_f1_data, hd_f1_data, avg_time_trace, dist_data)


if __name__ == "__main__":
    process_pickles()
    pass
