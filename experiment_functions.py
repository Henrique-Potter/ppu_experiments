import human_detection as hd
import face_match as fm
import cv2 as cv


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


def blur_iter_experiment(fid_model_path, hd_model_path, img_base_path, img_adversary_path, iter_max, hd_threshold=0.7, map_face_detection=False, blur_kernel="avg", blur_box_size=5, preview=False):

    img_base = cv.imread(img_base_path)
    img_adversary = cv.imread(img_adversary_path)

    face_det = fm.FaceMatch(fid_model_path)
    human_det = hd.DetectorAPI(path_to_ckpt=hd_model_path)

    img_base = cv.resize(img_base, (1280, 720))
    img_adversary = cv.resize(img_adversary, (1280, 720))

    hd_scores = []
    fm_scores = []
    blur_iterations = []
    img_blurred = img_adversary.copy()

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
        fm_scores.append(distance)

        if map_face_detection:
            detected_blurred_faces = face_det.extract_face(img_blurred)
            fm_scores.append([distance, detected_blurred_faces is None])
            print([distance, detected_blurred_faces is None])
        else:
            fm_scores.append(distance)

        if h_scores:
            hd_scores.append(h_scores)
        else:
            hd_scores.append(0)
        print(h_scores)

        blur_iterations.append(i)

        if preview is True:
            show_detections(img_blurred, boxes, img_adversary_faces_box, scores, classes, 0.5)
            key = cv.waitKey(1)
            if key & 0xFF == ord('q'):
                break

    return blur_iterations, fm_scores, hd_scores


