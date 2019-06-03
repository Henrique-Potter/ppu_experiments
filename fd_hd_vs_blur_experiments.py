import matplotlib.pyplot as plt
import pandas as pd
import cv2 as cv

import experiment_functions as ef


def blur_avg_box_experiment(fm_api, hd_api, img_base, img_adversary, iter_max, blur_kernel="avg"):

    hd_scores = []
    fm_scores = []
    blur_iterations = []
    img_blurred = img_adversary.copy()

    img_sizes = img_base.shape
    boxes1 = fm_api.extract_face(img_base)
    boxes2 = fm_api.extract_face(img_adversary)

    if blur_kernel == "resizing":
        iter_max = min(img_base.shape[0:2])

    for iteration in range(1, iter_max):

        if blur_kernel == "avg":
            img_blurred = cv.blur(img_blurred, (5, 5))
        elif blur_kernel == "gaussian":
            img_blurred = cv.GaussianBlur(img_blurred, (5, 5), 0)
        elif blur_kernel == "median":
            img_blurred = cv.medianBlur(img_blurred, 5)
        elif blur_kernel == "bilateralFiltering":
            img_blurred = cv.bilateralFilter(img_blurred, 9, 75, 75)
        elif blur_kernel == "resizing":
            x_axis_size = int(img_sizes[1] - img_sizes[1] * iteration/100)
            y_axis_size = int(img_sizes[0] - img_sizes[0] * iteration/100)

            if x_axis_size <= 40 or y_axis_size <= 40:
                break

            img_temp = cv.resize(img_adversary, (x_axis_size, y_axis_size))
            img_blurred = cv.resize(img_temp, (img_sizes[1], img_sizes[0]))

        boxes, scores, classes, num = hd_api.process_frame(img_blurred)
        h_boxes, h_scores = hd_api.get_detected_persons(boxes, scores, classes, 0.6)

        distance = fm_api.compare_faces_cropped(boxes1, boxes2, img_base, img_blurred)

        fm_scores.append(distance)
        if h_scores:
            hd_scores.append(h_scores[0])
        else:
            hd_scores.append(0)

        blur_iterations.append(iteration)

        print(distance)
        print(h_scores)

        # if iteration > 45:
        #     hdu.show_detections(img_blurred, boxes, scores, classes, 0.5)
        #     key = cv.waitKey(1)
        #     if key & 0xFF == ord('q'):
        #         break

    return blur_iterations, fm_scores, hd_scores


face_model_path = "face_id_models/20170512-110547.pb"
hd_model_path = "object_detection_models/frozen_inference_graph.pb"
hd_threshold = 0.7
iterations = 50
img_base_p = "./images/obama_signs.jpg"
img_adversary_p = "./images/obama_alone_office.jpg"
map_face_detection = True
blur_kernel = "avg"
kernel_size = 5


blur_iterations, fm_scores, hd_scores = ef.blur_iter_experiment(face_model_path,
                                                             hd_model_path,
                                                             img_base_p,
                                                             img_adversary_p,
                                                             iterations,
                                                             hd_threshold,
                                                             map_face_detection,
                                                             blur_kernel,
                                                             kernel_size, True)

#blur_iterations, fm_scores, hd_scores = blur_avg_box_experiment(face_det, human_det, img_base, img_adversary, 100, "resizing")

power = [float(i*0.33) for i in blur_iterations]
# Data
df = pd.DataFrame({'blur': blur_iterations,
                   'fm': fm_scores[:, 0],
                   'pd': hd_scores[:, 0]})

print(df)
# plt.plot('blur', 'fm', data=df, marker='', color='green', linewidth=2, label='Face Match (Euclidean Distance)')
# plt.plot('blur', 'pd', data=df, marker='', color='blue', linewidth=2, linestyle='dashed', label='Person Detection (Accuracy %)')
# plt.xticks(range(min(blur_iterations), max(blur_iterations), 10))
# plt.legend()
#
# plt.xlabel('Resizing Image - divided by')
# plt.ylabel('Euclidean Distance/Accuracy')
#
# plt.show()

fig = plt.figure()
power = [float(i*0.33) for i in df.iloc[:,0:2]]

ax = plt.axes(projection='3d')
s = ax.scatter3D(fm_scores[:, 0], hd_scores[:, 0], power, c=power, cmap='Greens')
s.set_edgecolors = s.set_facecolors = lambda *args:None
#ax.plot_trisurf(fm_scores, hd_scores, power, cmap='viridis', edgecolor='none', label='PPU Plane')
ax.set_title('Power x Privacy x Utility')

ax.set_xlabel('Privacy', fontsize=20)
ax.set_ylabel('Utility', fontsize=20)
ax.set_zlabel('Power', fontsize=20)

plt.show()

