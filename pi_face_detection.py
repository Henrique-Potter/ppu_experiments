import argparse

import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import face_match as fm
import time
import experiment_functions
import cv2 as cv
import os
import numpy as np

learned_faces = []


def check_face(face_det, frame, debug=False):

    face_boxes = face_det.extract_face(frame)

    found_distance = 0
    frame_face_emb = 0

    if np.any(face_boxes):
        frame_face_data = face_det.get_face_embeddings(face_boxes, frame, debug)
        frame_face_emb = frame_face_data[0]['embedding']

        for face_emb in learned_faces:
            distance = face_det.euclidean_distance(frame_face_emb, face_emb)

            if distance < 1.1:
                found_distance = distance

    return found_distance, frame_face_emb, face_boxes


def learn_new_face(face_emb):
    learned_faces.append(face_emb)


def main_method():

    parser = argparse.ArgumentParser()
    parser.add_argument("--fid_m", type=str, required=True)
    #parser.add_argument("--hd_m", type=str, required=True)
    #parser.add_argument("--hd_thres", type=float, required=True)
    parser.add_argument("--preview", type=bool, required=True)

    args = parser.parse_args()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(dir_path, args.fid_m)

    experiment_utils = experiment_functions.BlurExperiments
    face_det = fm.FaceMatch(model_path)
    #human_det = hd.DetectorAPI(path_to_ckpt = args.hd_m)

    # vs = VideoStream(usePiCamera=True).start()
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    fps = FPS().start()
    
    while True:

        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        (fH, fW) = frame.shape[:2]

        start = time.time()

        f_dist, f_emb, f_boxes = check_face(face_det, frame)
        if f_dist is not 0:
            print("Face found")
        else:
            learn_new_face(f_emb)

        if args.preview is True:
            experiment_utils.show_detections(frame, [], f_boxes, 0, 0, 0.5)
            key = cv.waitKey(1)
            if key & 0xFF == ord('q'):
                break

        end = time.time()
        print("Time to process frame time: {}".format(end - start))

        fps.update()

    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    cv.destroyAllWindows()
    vs.stop()


    #boxes, scores, classes, humans_detected_map, num = human_det.process_frame(frame, args.hd_thres, 1)
    #h_boxes, h_scores = human_det.get_detected_persons(boxes, scores, classes, hd_threshold)


if __name__ == "__main__":
    main_method()
