import human_detection as hd
import face_match as fm
import numpy as np
import pandas as pd
from pathlib import Path
import os
import cv2 as cv


class BlurExperiments:

    def __init__(self, fid_model_path, hd_model_path):
        self.face_det = fm.FaceMatch(fid_model_path)
        self.human_det = hd.DetectorAPI(path_to_ckpt=hd_model_path)

    @staticmethod
    def show_detections(img_dbg, h_boxes, f_boxes, scores, classes, threshold):

        img_cp = img_dbg.copy()

        for f_box in f_boxes:

            cv.rectangle(img_cp, (f_box[0], f_box[1]), (f_box[2], f_box[3]), (255, 0, 0), 2)
            cv.putText(img_cp, "Face", (f_box[2] + 10, f_box[3]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        for i in range(len(h_boxes)):
            # Class 1 represents human
            if classes[i] == 1 and scores[i] > threshold:
                box = h_boxes[i]
                label = "Person: " + str(scores[i])

                cv.rectangle(img_cp, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)
                cv.putText(img_cp, label, (box[1], box[0] - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv.imshow("Debugging", img_cp)

    @staticmethod
    def image_resize(image, width=None, height=None, inter=cv.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image

        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)

        else:
            r = width / float(w)
            dim = (width, int(h * r))

        resized = cv.resize(image, dim, interpolation=inter)

        return resized

    def blur_iter_experiment(self, img_base_path, img_adversary_path, iter_max, hd_threshold=0.7, map_face_detection=False, blur_kernel="avg", blur_box_size=5, preview=False):

        img_base = cv.imread(img_base_path)
        img_adversary = cv.imread(img_adversary_path)

        face_det = self.face_det
        human_det = self.human_det

        if max(img_base.shape) > 1280:
            img_base = self.image_resize(img_base, width=1280)
            img_adversary = self.image_resize(img_adversary, width=1280)

        hd_scores = []
        fm_scores = []
        blur_iterations = []
        img_blurred = img_adversary.copy()
        #cv.imwrite('t.jpg', img_base)
        img_sizes = img_base.shape
        img_base_faces_box = face_det.extract_face(img_base)
        img_adversary_faces_box = face_det.extract_face(img_adversary)

        blur_box = (blur_box_size, blur_box_size)

        if blur_kernel == "resizing":
            iter_max = min(img_base.shape[0:2])

        for i in range(1, iter_max):

            if blur_kernel == "avg":
                img_blurred = cv.blur(img_blurred, blur_box)
            elif blur_kernel == "gaussian":
                img_blurred = cv.GaussianBlur(img_blurred, blur_box, 0)
            elif blur_kernel == "median":
                img_blurred = cv.medianBlur(img_blurred, blur_box_size)
            elif blur_kernel == "bilateralFiltering":
                img_blurred = cv.bilateralFilter(img_blurred, blur_box_size, 75, 75)
            elif blur_kernel == "resizing":
                x_axis_size = int(img_sizes[1] - img_sizes[1] * i/100)
                y_axis_size = int(img_sizes[0] - img_sizes[0] * i/100)

                if x_axis_size <= 40 or y_axis_size <= 40:
                    break

                img_temp = cv.resize(img_adversary, (x_axis_size, y_axis_size))
                img_blurred = cv.resize(img_temp, (img_sizes[1], img_sizes[0]))

            boxes, scores, classes, num = human_det.process_frame(img_blurred)
            h_boxes, h_scores = human_det.get_detected_persons(boxes, scores, classes, hd_threshold)

            distance = face_det.compare_faces_cropped(img_base_faces_box, img_adversary_faces_box, img_base, img_blurred)

            if map_face_detection is 'y':
                detected_blurred_faces = face_det.extract_face(img_blurred)
                fm_scores.append([distance, len(detected_blurred_faces) is not 0])
                print([distance, len(detected_blurred_faces) is not 0])
            else:
                fm_scores.append(distance)

            if len(h_scores) > 0:
                hd_scores.append(h_scores)
                print(h_scores)
            else:
                hd_scores.append([0])
                print([0])

            blur_iterations.append(i)

            if preview is True:
                self.show_detections(img_blurred, h_boxes, img_adversary_faces_box, scores, classes, 0.5)
                key = cv.waitKey(1)
                if key & 0xFF == ord('q'):
                    break

        df = pd.DataFrame({'blur': np.array(blur_iterations)})
        df2 = pd.DataFrame({'f_score': np.array(fm_scores)[:, 0], 'f_detection': np.array(fm_scores)[:, 1]})
        df3 = pd.DataFrame(np.array(hd_scores))

        full_df = pd.concat([df, df2, df3], axis=1)

        return full_df


