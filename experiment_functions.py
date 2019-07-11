import numpy as np


class BlurExperiments:

    def __init__(self, fid_model, hd_model):
        self.face_det = fid_model
        self.human_det = hd_model

    @staticmethod
    def show_detections(img_cp, h_boxes, f_boxes, scores, obj_map, threshold, window_title='Debugging'):
        import cv2 as cv

        img_temp = img_cp.copy()

        cv.putText(img_temp, window_title, (5, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

        for f_box in f_boxes:
            cv.rectangle(img_temp, (f_box[0], f_box[1]), (f_box[2], f_box[3]), (255, 0, 0), 2)
            cv.putText(img_temp, "Face", (f_box[2] + 10, f_box[3]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        for i in range(len(h_boxes)):
            # Class 1 represents human
            if obj_map[i] and scores[i] > threshold:
                box = h_boxes[i]
                label = "Person: " + str(scores[i])

                cv.rectangle(img_temp, (int(box[1]), int(box[0])), (int(box[3]), int(box[2])), (255, 0, 0), 2)
                cv.putText(img_temp, label, (int(box[1]), int(box[0] - 5)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv.imshow("", img_temp)

    @staticmethod
    def image_resize(image, width=None, height=None):
        import cv2 as cv
        dim = None
        inter = cv.INTER_AREA

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

    def blur_iter_experiment(self, img_base_path, img_adversary_path, iter_max, hd_threshold=0.7, map_face_detection=False, blur_kernel="avg", init_blur_box_size=3, preview=False):
        import cv2 as cv
        import time
        import pandas as pd

        img_base = cv.imread(img_base_path)
        #img_adversary = cv.imread(img_adversary_path)
        # The image will be copied further ahead
        img_adversary = img_base

        face_det = self.face_det
        human_det = self.human_det

        if max(img_base.shape) > 1280:
            img_base = self.image_resize(img_base, width=1280)
            img_adversary = self.image_resize(img_adversary, width=1280)

        hd_scores_np = np.zeros([iter_max, 15], dtype=np.int8)
        fm_scores_np = np.zeros([iter_max, 15], dtype="S20")

        #cv.imwrite('t.jpg', img_base)
        img_sizes = img_base.shape

        img_base_faces_box = face_det.extract_face(img_base)
        #img_adversary_faces_box = face_det.extract_face(img_adversary)

        if blur_kernel == "resizing":
            iter_max = min(img_base.shape[0:2])

        blur_box_iteration = 0

        for box_size in range(init_blur_box_size, 32, 2):

            img_blurred = img_adversary.copy()
            blur_box = (box_size, box_size)

            start2 = time.time()
            for iteration in range(0, iter_max):

                start = time.time()

                if blur_kernel == "avg":
                    img_blurred = cv.blur(img_blurred, blur_box)
                elif blur_kernel == "gaussian":
                    img_blurred = cv.GaussianBlur(img_blurred, blur_box, 0)
                elif blur_kernel == "median":
                    img_blurred = cv.medianBlur(img_blurred, box_size)
                elif blur_kernel == "bilateralFiltering":
                    img_blurred = cv.bilateralFilter(img_blurred, box_size, 75, 75)
                elif blur_kernel == "resizing":
                    x_axis_size = int(img_sizes[1] - img_sizes[1] * (iteration+1)/100)
                    y_axis_size = int(img_sizes[0] - img_sizes[0] * (iteration+1)/100)

                    if x_axis_size <= 40 or y_axis_size <= 40:
                        break

                    img_temp = cv.resize(img_adversary, (x_axis_size, y_axis_size))
                    img_blurred = cv.resize(img_temp, (img_sizes[1], img_sizes[0]))

                end = time.time() - start
                print("Blur iteration {} with kernel {} time: {}".format(iteration, blur_kernel, end))

                # ---- Applying classifiers over the image ----
                h_boxes, h_scores, obj_map, num = human_det.process_frame(img_blurred, hd_threshold, 1)
                distance = face_det.compare_faces_cropped(img_base_faces_box, img_base_faces_box, img_base, img_blurred)

                # ---- Collecting Results ----
                col_id = blur_box_iteration

                detected_blurred_faces = face_det.extract_face(img_blurred)
                fm_scores_np[iteration, col_id] = "{} {} {}".format(str(distance), len(img_base_faces_box), len(detected_blurred_faces))
                hd_scores_np[iteration, col_id] = np.count_nonzero(obj_map)

                if preview is True:
                    title = "Debugging Kernel:{} Box Size:{} Iteration:{}".format(blur_kernel, box_size, iteration)
                    self.show_detections(img_blurred, h_boxes, img_base_faces_box, h_scores, obj_map, 0.5, title)
                    key = cv.waitKey(1)
                    if key & 0xFF == ord('q'):
                        break

            blur_box_iteration += 1
            print("{}x{} box iteration total time: {}".format(box_size, box_size, time.time() - start2))

        # ---- Combining Results ----
        obj_headers = ['obj b{}x{}'.format(box, box) for box in range(init_blur_box_size, 32, 2)]
        face_headers = ['face b{}x{}'.format(box, box) for box in range(init_blur_box_size, 32, 2)]
        obj_dets_df = pd.DataFrame(data=hd_scores_np, columns=obj_headers)
        face_dets_df = pd.DataFrame(data=fm_scores_np, columns=face_headers)

        total_df = pd.concat([face_dets_df, obj_dets_df], axis=1, sort=False)

        # print(face_dets_df)
        # print(obj_dets_df)
        # print(total_df)

        return total_df


