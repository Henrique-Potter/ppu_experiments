from imutils.video import VideoStream
from imutils.video import FPS
import experiment_functions
import face_match as fm
import numpy as np
import cv2 as cv
import time
import os

blue_color = (255, 0, 0)
green_color = (0, 255, 0)
red_color = (0, 0, 255)


class PiFaceDet:

    def __init__(self, face_det_model_path='face_id_models\\20170512-110547.pb', preview=True):

        dir_path = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(dir_path, face_det_model_path)

        self.face_det = fm.FaceMatch(model_path)
        self.preview = preview
        self.learned_faces = []

        #self.vs =  VideoStream(usePiCamera=True).start()
        #self.vs = VideoStream(src=0)

        self.last_seen_face = 0

    def check_face(self, frame):

        face_boxes = self.face_det.extract_face(frame)
        frame_face_emb = None
        face_dist = 0
        known_face = False

        if np.any(face_boxes):
            frame_face_data = self.face_det.get_face_embeddings(face_boxes, frame)
            frame_face_emb = frame_face_data[0]['embedding']

            for face_emb in self.learned_faces:
                face_dist = self.face_det.euclidean_distance(frame_face_emb, face_emb)

                if face_dist < 1.1:
                    known_face = True

        return face_boxes, frame_face_emb, face_dist, known_face

    def run_identification(self, sample_frames=1000):

        experiment_utils = experiment_functions.BlurExperiments

        vs = VideoStream(src=0).start()
        time.sleep(2.0)
        fps = FPS().start()
        frames_count = 0

        known_face_found = False
        color = red_color

        while frames_count < sample_frames:

            start = time.time()

            frame = vs.read()
            #frame = imutils.resize(frame, width=400)
            #(fH, fW) = frame.shape[:2]

            f_boxes, frame_face_emb, face_dist, known_face = self.check_face(frame)

            if known_face:
                known_face_found = True
                color = green_color

            if self.preview:
                experiment_utils.show_detections(frame, [], f_boxes, color, 0, 0, 0.5)
                key = cv.waitKey(1)
                if key & 0xFF == ord('q'):
                    break

            frames_count = frames_count + 1

            end = time.time()
            print("Time to process frame: {}".format(end - start))
            fps.update()

        fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        cv.destroyAllWindows()
        vs.stop()
        time.sleep(2.0)

        return known_face_found

    def run_learn_face(self, sample_frames=1000):

        experiment_utils = experiment_functions.BlurExperiments

        vs = VideoStream(src=0).start()
        time.sleep(2.0)
        fps = FPS().start()
        frames_count = 0

        learn_success = False
        color = blue_color

        while frames_count < sample_frames:

            start = time.time()

            frame = vs.read()
            # frame = imutils.resize(frame, width=400)
            # (fH, fW) = frame.shape[:2]

            f_boxes, frame_face_emb, face_dist, known_face = self.check_face(frame)

            if frame_face_emb is not None and not known_face:
                self.learned_faces.append(frame_face_emb)
                learn_success = True
                color = blue_color

            if self.preview:
                experiment_utils.show_detections(frame, [], f_boxes, color, 0, 0, 0.5)
                key = cv.waitKey(1)
                if key & 0xFF == ord('q'):
                    break

            frames_count = frames_count + 1

            end = time.time()
            print("Time to process frame and add new face: {}".format(end - start))
            fps.update()

        fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        cv.destroyAllWindows()
        vs.stop()
        time.sleep(2.0)

        return learn_success
