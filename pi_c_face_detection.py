from imutils.video import VideoStream
from imutils.video import FPS
import face_match as fm
from pathlib import Path
import json
import numpy as np
import cv2 as cv
import platform
import time
import os

blue_color = (255, 0, 0)
green_color = (0, 255, 0)
red_color = (0, 0, 255)

use_raspiberry = False

if platform.uname()[1] is 'raspberrypi':
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BOARD)

    beep_pin = 40
    g_led_pin = 36
    r_led_pin = 38

    GPIO.setup(beep_pin, GPIO.OUT, initial=0)
    GPIO.setup(g_led_pin, GPIO.OUT, initial=1)
    GPIO.setup(r_led_pin, GPIO.OUT, initial=1)

    use_raspiberry = True


class PiFaceDet:

    def __init__(self, face_det_model_path='face_id_models/20170512-110547.pb', preview=False):

        dir_path = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(dir_path, face_det_model_path)

        self.faces_db_file = os.path.join(dir_path, "faces_db.json")

        if Path(self.faces_db_file).is_file():
            with open(self.faces_db_file) as json_file:
                self.faces_db = json.load(json_file)
        else:
            self.faces_db = {}

        self.face_det = fm.FaceMatch(model_path)
        self.preview = preview
        self.last_seen_face = 0
        self.learn_face = False

    def run_identification(self, sample_frames=1000):

        vs = self.get_cam()
        time.sleep(2.0)
        fps = FPS().start()
        frames_count = 0

        known_face_found = False

        while frames_count < sample_frames:

            start = time.time()
            frame = vs.read()
            #f_boxes, frame_face_emb, most_similar_face = self.check_face(frame)
            frames_count = frames_count + 1

            end = time.time()
            print("Time to process frame: {}".format(end - start))
            fps.update()

        fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        vs.stop()
        time.sleep(2.0)

        return known_face_found

    def continuous_face_identification(self, learn_face_count):

        vs = self.get_cam()
        time.sleep(2.0)
        fps = FPS().start()
        frames_count = 0

        color = blue_color

        while True:

            start = time.time()
            frame = vs.read()
            face_found, faces_boxes = self.detect_face(frame)

            if face_found and learn_face_count.empty():
                frame_face_data = self.face_det.get_face_embeddings(faces_boxes, frame)
                frame_face_emb = frame_face_data[0]['embedding']
                most_similar_name, most_similar_emb = self.find_face(frame_face_emb)

                if most_similar_name:
                    print("Found {}".format(most_similar_name))
                    self.beep(2, 0.3)
                    self.green_blink(1, 2)
                else:
                    print("Alert user no authorized")
                    self.beep(1, 1)
                    self.red_blink(1, 2)

            if not learn_face_count.empty() and face_found:
                self.learn_new_face(faces_boxes, frame)
                learn_face_count.get()
                self.beep(4, 0.3)
                self.green_blink(4)

            if self.preview:
                self.show_detections(frame, faces_boxes, color)
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

    def check_face(self, frame):

        face_boxes = self.face_det.extract_face(frame)
        frame_face_emb = None
        most_similar_emb = None
        most_similar_name = None

        if np.any(face_boxes):
            frame_face_data = self.face_det.get_face_embeddings(face_boxes, frame)
            frame_face_emb = frame_face_data[0]['embedding']
            most_similar_name, most_similar_emb = self.find_face(frame_face_emb)

        return face_boxes, most_similar_emb

    def detect_face(self, frame):
        face_boxes = self.face_det.extract_face(frame)
        faces_found = np.any(face_boxes)

        return faces_found, face_boxes

    def find_face(self, frame_face_emb):

        most_similar_emb = None
        most_similar_name = None
        dist_place_holder = 1.1

        if self.faces_db:
            for name, face_data in self.faces_db.items():
                f_emb = np.asarray(face_data['embedding'])
                face_dist = self.face_det.euclidean_distance(frame_face_emb, f_emb)

                if face_dist <= dist_place_holder:
                    dist_place_holder = face_dist
                    most_similar_emb = face_data
                    most_similar_name = name

        return most_similar_name, most_similar_emb

    def learn_new_face(self, faces_boxes, frame):
        frame_face_data = self.face_det.get_face_embeddings(faces_boxes, frame)
        frame_face_emb = frame_face_data[0]['embedding']
        most_similar_face = self.find_face(frame_face_emb)
        new_face_emb = None
        if most_similar_face is None:
            new_face_emb = frame_face_emb
            face = {'embedding': frame_face_emb}
            self.last_seen_face = self.last_seen_face + 1
            self.faces_db['new_face {}'.format(self.last_seen_face)] = face
            with open(self.faces_db_file, 'w') as outfile:
                json.dump(self.faces_db, outfile, sort_keys=True, indent=4, cls=NumpyEncoder)

        return new_face_emb

    @staticmethod
    def beep(beep_times, duration=0.1):
        if platform.uname()[1] is 'raspberrypi':
            for i in range(beep_times):
                GPIO.output(beep_pin, GPIO.HIGH)
                time.sleep(duration)
                GPIO.output(beep_pin, GPIO.LOW)
                time.sleep(duration)

    @staticmethod
    def red_blink(blink_times, duration=0.3):
        if platform.uname()[1] is 'raspberrypi':
            for i in range(blink_times):
                GPIO.output(r_led_pin, GPIO.HIGH)
                time.sleep(duration)
                GPIO.output(r_led_pin, GPIO.LOW)
                time.sleep(duration)

    @staticmethod
    def green_blink(blink_times, duration=0.3):
        if platform.uname()[1] is 'raspberrypi':
            for i in range(blink_times):
                GPIO.output(g_led_pin, GPIO.HIGH)
                time.sleep(duration)
                GPIO.output(g_led_pin, GPIO.LOW)
                time.sleep(duration)

    @staticmethod
    def get_cam():
        if use_raspiberry:
            return VideoStream(usePiCamera=True).start()
        else:
            return VideoStream(src=0).start()

    @staticmethod
    def show_detections(img_cp, f_boxes, color):
        for f_box in f_boxes:
            cv.rectangle(img_cp, (f_box[0], f_box[1]), (f_box[2], f_box[3]), color, 2)
            cv.putText(img_cp, "Face", (f_box[2] + 10, f_box[3]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv.imshow("Debugging", img_cp)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)