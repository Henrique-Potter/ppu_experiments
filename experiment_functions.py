import numpy as np
import cv2 as cv


class BlurExperiments:

    def __init__(self, fid_model, hd_model):
        self.face_det = fid_model
        self.human_det = hd_model

    @staticmethod
    def show_detections(img_cp, h_boxes, f_boxes, scores, classes, threshold):

        #img_cp = img_dbg.copy()

        for f_box in f_boxes:

            cv.rectangle(img_cp, (f_box[0], f_box[1]), (f_box[2], f_box[3]), (255, 0, 0), 2)
            cv.putText(img_cp, "Face", (f_box[2] + 10, f_box[3]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        for i in range(len(h_boxes)):
            # Class 1 represents human
            if classes[i] == 1 and scores[i] > threshold:
                box = h_boxes[i]
                label = "Person: " + str(scores[i])

                cv.rectangle(img_cp, (int(box[1]), int(box[0])), (int(box[3]), int(box[2])), (255, 0, 0), 2)
                cv.putText(img_cp, label, (int(box[1]), int(box[0] - 5)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

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

        import time
        import pandas as pd

        img_base = cv.imread(img_base_path)
        #img_adversary = cv.imread(img_adversary_path)
        img_adversary = img_base

        face_det = self.face_det
        human_det = self.human_det

        if max(img_base.shape) > 1280:
            img_base = self.image_resize(img_base, width=1280)
            img_adversary = self.image_resize(img_adversary, width=1280)

        #hd_scores = []
        #fm_scores = []

        hd_scores_np = np.zeros(iter_max, dtype=np.int8)
        fm_scores_np = np.zeros([iter_max, 2])

        img_blurred = img_adversary.copy()
        #cv.imwrite('t.jpg', img_base)
        img_sizes = img_base.shape

        img_base_faces_box = face_det.extract_face(img_base)
        #img_adversary_faces_box = face_det.extract_face(img_adversary)

        blur_box = (blur_box_size, blur_box_size)

        if blur_kernel == "resizing":
            iter_max = min(img_base.shape[0:2])

        for i in range(0, iter_max):

            start = time.time()

            if blur_kernel == "avg":
                img_blurred = cv.blur(img_blurred, blur_box)
            elif blur_kernel == "gaussian":
                img_blurred = cv.GaussianBlur(img_blurred, blur_box, 0)
            elif blur_kernel == "median":
                img_blurred = cv.medianBlur(img_blurred, blur_box_size)
            elif blur_kernel == "bilateralFiltering":
                img_blurred = cv.bilateralFilter(img_blurred, blur_box_size, 75, 75)
            elif blur_kernel == "resizing":
                x_axis_size = int(img_sizes[1] - img_sizes[1] * (i+1)/100)
                y_axis_size = int(img_sizes[0] - img_sizes[0] * (i+1)/100)

                if x_axis_size <= 40 or y_axis_size <= 40:
                    break

                img_temp = cv.resize(img_adversary, (x_axis_size, y_axis_size))
                img_blurred = cv.resize(img_temp, (img_sizes[1], img_sizes[0]))

            end = time.time()
            print("1 Blur with kernel {} time: {}".format(blur_kernel, end - start))

            boxes, scores, classes, humans_detected_map, num = human_det.process_frame(img_blurred, hd_threshold, 1)
            #h_boxes, h_scores = human_det.get_detected_persons(boxes, scores, classes, hd_threshold)

            distance = face_det.compare_faces_cropped(img_base_faces_box, img_base_faces_box, img_base, img_blurred)

            if map_face_detection is 'y':
                detected_blurred_faces = face_det.extract_face(img_blurred)
                fm_scores_np[i, 0] = distance
                fm_scores_np[i, 1] = len(detected_blurred_faces) > 0
                #fm_scores.append([distance, len(detected_blurred_faces) is not 0])
                #print(fm_scores_np[i, :])
            else:
                fm_scores_np[i, 0] = distance
                fm_scores_np[i, 1] = 0
                #fm_scores.append([distance, distance])
                #print(fm_scores_np[i, :])

            hd_scores_np[i] = np.count_nonzero(humans_detected_map)
            #print(hd_scores_np[i])


            # if len(h_scores) > 0:
            #     hd_scores.append(h_scores)
            #     print(h_scores)
            # else:
            #     hd_scores.append([0])
            #     print([0])

            # blur_iterations.append(i)

            if preview is True:
                self.show_detections(img_blurred, boxes, img_base_faces_box, scores, classes, 0.5)
                key = cv.waitKey(1)
                if key & 0xFF == ord('q'):
                    break

        df = pd.DataFrame({'f_score': fm_scores_np[:, 0], 'f_detection': fm_scores_np[:, 1], 'human_det:': hd_scores_np})

        print(df)

        # df3 = pd.DataFrame(np.array(hd_scores))
        #
        # full_df = pd.concat([df, df2, df3], axis=1)

        return df


