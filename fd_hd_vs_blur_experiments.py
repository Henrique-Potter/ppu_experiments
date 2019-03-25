import human_detection as hd
import human_detection_utils as hdu
import face_match as fm
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def blur_avg_box_experiment(fm_api, hd_api, img_base, img_adversary, iter_max, blur_kernel="avg"):

    hd_scores = []
    fm_scores = []
    blur_iterations = []
    img_blurred = img_adversary.copy()
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

        distance = fm_api.compare_faces(img_base, img_blurred)

        print(distance)
        print(h_scores)

        fm_scores.append(distance)
        hd_scores.append(h_scores[0])

        blur_iterations.append(i)

        #hdu.show_detections(img_base, boxes, scores, classes, 0.5)

        #key = cv.waitKey(1)
        #if key & 0xFF == ord('q'):
        #    break

    return blur_iterations, fm_scores, hd_scores


face_det = fm.FaceMatch()

model_path = 'faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
human_det = hd.DetectorAPI(path_to_ckpt=model_path)
threshold = 0.7

img_base = cv.imread("./images/Obama_signs.jpg")
img_adversary = cv.imread("./images/obama_alone_office.jpg")

img_base = cv.resize(img_base, (1280, 720))
img_adversary = cv.resize(img_adversary, (1280, 720))

blur_iterations, fm_scores, hd_scores = blur_avg_box_experiment(face_det, human_det, img_base, img_adversary, 20, "gaussian")

# Data
df = pd.DataFrame({'blur': blur_iterations,
                   'fm': fm_scores,
                   'pd': hd_scores})

plt.plot('blur', 'fm', data=df, marker='', color='green', linewidth=2, label='Face Match (Euclidean Distance)')
plt.plot('blur', 'pd', data=df, marker='', color='blue', linewidth=2, linestyle='dashed', label='Person Detection (Accuracy %)')
plt.xticks(range(min(blur_iterations), max(blur_iterations), 1))
plt.legend()

plt.xlabel('Gaussian 5x5 Blur rounds')
plt.ylabel('Euclidean Distance/Accuracy')

plt.show()




