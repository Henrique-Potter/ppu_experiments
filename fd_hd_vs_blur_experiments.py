import human_detection as hd
import human_detection_utils as hdu
import face_match as fm
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import pandas as pd


def blur_avg_box_experiment(fm_api, hd_api, img_base, img_adversary, iter_max, blur_kernel="avg"):

    hd_scores = []
    fm_scores = []
    blur_iterations = []
    img_blurred = img_adversary.copy()

    boxes1 = fm_api.extract_face(img_base)
    boxes2 = fm_api.extract_face(img_adversary)

    for i in range(1, iter_max):

        if blur_kernel == "avg":
            img_blurred = cv.blur(img_blurred, (5, 5))
        elif blur_kernel == "gaussian":
            img_blurred = cv.GaussianBlur(img_blurred, (5, 5), 0)
        elif blur_kernel == "median":
            img_blurred = cv.medianBlur(img_blurred, 5)
        elif blur_kernel == "bilateralFiltering":
            img_blurred = cv.bilateralFilter(img_blurred, 9, 75, 75)

        boxes, scores, classes, num = hd_api.process_frame(img_blurred)
        h_boxes, h_scores = hd_api.get_detected_persons(boxes, scores, classes, 0.6)

        #distance = fm_api.compare_faces(img_base, img_blurred)

        distance = fm_api.compare_faces_cropped(boxes1, boxes2, img_base, img_blurred)

        print(distance)
        print(h_scores)

        fm_scores.append(distance)
        if h_scores:
            hd_scores.append(h_scores[0])
        else:
            hd_scores.append(0)

        blur_iterations.append(i)

        if i > 45:
            hdu.show_detections(img_blurred, boxes, scores, classes, 0.5)
            key = cv.waitKey(1)
            if key & 0xFF == ord('q'):
                break

    return blur_iterations, fm_scores, hd_scores


face_det = fm.FaceMatch()

model_path = 'object_detection_models/frozen_inference_graph.pb'
human_det = hd.DetectorAPI(path_to_ckpt=model_path)
threshold = 0.7

img_base = cv.imread("./images/Obama_signs.jpg")
img_adversary = cv.imread("./images/obama_alone_office.jpg")

img_base = cv.resize(img_base, (1280, 720))
img_adversary = cv.resize(img_adversary, (1280, 720))

blur_iterations, fm_scores, hd_scores = blur_avg_box_experiment(face_det, human_det, img_base, img_adversary, 100, "avg")

power = [float(i*0.33) for i in blur_iterations]
# Data
df = pd.DataFrame({'blur': blur_iterations,
                   'fm': fm_scores,
                   'pd': hd_scores})


# plt.plot('blur', 'fm', data=df, marker='', color='green', linewidth=2, label='Face Match (Euclidean Distance)')
# plt.plot('blur', 'pd', data=df, marker='', color='blue', linewidth=2, linestyle='dashed', label='Person Detection (Accuracy %)')
# plt.xticks(range(min(blur_iterations), max(blur_iterations), 10))
# plt.legend()
#
# plt.xlabel('Average 5x5 Blur rounds')
# plt.ylabel('Euclidean Distance/Accuracy')
#
# plt.show()

fig = plt.figure()
power = [float(i*0.33) for i in blur_iterations]

ax = plt.axes(projection='3d')
ax.scatter3D(fm_scores, hd_scores, power, c=power, cmap='Greens');
#ax.plot_trisurf(fm_scores, hd_scores, power, cmap='viridis', edgecolor='none', label='PPU Plane')
ax.set_title('Power x Privacy x Utility')

ax.set_xlabel('Privacy', fontsize=20)
ax.set_ylabel('Utility', fontsize=20)
ax.set_zlabel('Power', fontsize=20)

plt.show()

