import human_detection as hd
import human_detection_utils as hdu
import face_match as fm
import cv2 as cv


def blur_avg_box_experiment(fid_model_path, hd_model_path, img_base_path, img_adversary_path, iter_max, hd_threshold=0.7, map_face_detection=False, blur_kernel="avg", blur_box_size=5):

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

    img_base_faces_box = face_det.extract_face(img_base)
    img_adversary_faces_box = face_det.extract_face(img_adversary)

    blur_box = (blur_box_size, blur_box_size)

    for i in range(1, iter_max):

        if blur_kernel == "avg":
            img_blurred = cv.blur(img_blurred, blur_box)
        elif blur_kernel == "gaussian":
            img_blurred = cv.GaussianBlur(img_blurred, blur_box, 0)
        elif blur_kernel == "median":
            img_blurred = cv.medianBlur(img_blurred, blur_box_size)
        elif blur_kernel == "bilateralFiltering":
            img_blurred = cv.bilateralFilter(img_blurred, blur_box_size, 75, 75)

        boxes, scores, classes, num = human_det.process_frame(img_blurred)
        h_boxes, h_scores = human_det.get_detected_persons(boxes, scores, classes, hd_threshold)

        distance = face_det.compare_faces_cropped(img_base_faces_box, img_adversary_faces_box, img_base, img_blurred)

        print(distance)
        print(h_scores)

        fm_scores.append(distance)

        if map_face_detection:
            detected_blurred_faces = face_det.extract_face(img_blurred)
            fm_scores.append([distance, detected_blurred_faces is None])
        else:
            fm_scores.append(distance)

        if h_scores:
            hd_scores.append(h_scores)
        else:
            hd_scores.append(0)

        blur_iterations.append(i)

        if i > 45:
            hdu.show_detections(img_blurred, boxes, scores, classes, 0.5)
            key = cv.waitKey(1)
            if key & 0xFF == ord('q'):
                break

    return blur_iterations, fm_scores, hd_scores










