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

beep_pin = 40
g_led_pin = 36
r_led_pin = 38

if platform.uname()[1] == 'raspberrypi':
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BOARD)

    GPIO.setup(beep_pin, GPIO.OUT, initial=0)
    GPIO.setup(g_led_pin, GPIO.OUT, initial=0)
    GPIO.setup(r_led_pin, GPIO.OUT, initial=0)

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
            self.faces_db = {'face_count': 0, 'faces': {}}

        self.face_det = fm.FaceMatch(model_path)
        self.preview = preview
        self.face_counter = 0

    def id_face_trigger(self, sample_frames=10):

        vs = self.get_cam()
        time.sleep(2.0)

        color = blue_color
        frame_count = 0

        most_similar_name = None

        while frame_count < sample_frames:

            frame = vs.read()

            start1 = time.time()
            face_found, faces_boxes = self.detect_face(frame)
            print("Time to detect face: {}".format(time.time() - start1))

            if face_found:
                self.beep_blink(1, g_led_pin, 0.1)

                start2 = time.time()
                frame_face_data = self.face_det.get_face_embeddings(faces_boxes, frame)
                print("Time to extract embeddings: {}".format(time.time() - start2))

                frame_face_emb = frame_face_data[0]['embedding']

                start3 = time.time()
                most_similar_name, most_similar_emb, match_map = self.find_face(frame_face_emb)
                print("Time to find face in DB: {}".format(time.time() - start3))

                if most_similar_name:
                    print("Authorization confirmed".format(most_similar_name))
                    self.beep_blink(2, g_led_pin, 0.3)
                else:
                    print("Alert! User not authorized detected")
                    self.beep_blink(1, r_led_pin, 1.5)

            if self.preview:
                self.show_detections(frame, faces_boxes, color)
                key = cv.waitKey(1)
                if key & 0xFF == ord('q'):
                    break

            frame_count += 1

        cv.destroyAllWindows()
        vs.stop()
        time.sleep(2.0)

        return most_similar_name

    def learn_face_trigger(self, sample_frames=5):

        vs = self.get_cam()
        time.sleep(2.0)
        frame_count = 0

        color = blue_color

        state_changed = False
        new_name = None

        while frame_count < sample_frames:

            frame = vs.read()

            start1 = time.time()
            face_found, faces_boxes = self.detect_face(frame)
            print("Time to detect face: {}".format(time.time() - start1))

            if face_found:

                self.beep_blink(8, g_led_pin, 0.1)
                start4 = time.time()
                new_face_name, new_embs_added = self.learn_new_face(faces_boxes, frame)
                state_changed = new_embs_added
                new_name = new_face_name

                print("Time to learn face: {}".format(time.time() - start4))
                print("New face: {} was learned.".format(new_face_name))

                self.beep_blink(4, g_led_pin, 0.3)

            if self.preview:
                self.show_detections(frame, faces_boxes, color)
                key = cv.waitKey(1)
                if key & 0xFF == ord('q'):
                    break

            frame_count += 1

        if state_changed:
            self.save_db_state()

        cv.destroyAllWindows()
        vs.stop()
        time.sleep(2.0)

        return new_name

    def continuous_face_identification(self, learn_face_count):

        vs = self.get_cam()
        time.sleep(2.0)
        fps = FPS().start()

        color = blue_color

        while True:

            frame = vs.read()

            start1 = time.time()
            face_found, faces_boxes = self.detect_face(frame)
            print("Time to detect face: {}".format(time.time() - start1))

            if face_found and learn_face_count.empty():
                self.beep_blink(1, g_led_pin, 0.1)

                start2 = time.time()
                frame_face_data = self.face_det.get_face_embeddings(faces_boxes, frame)
                print("Time to extract embeddings: {}".format(time.time() - start2))

                frame_face_emb = frame_face_data[0]['embedding']

                start3 = time.time()
                most_similar_name, most_similar_emb, match_map = self.find_face(frame_face_emb)
                print("Time to find face in DB: {}".format(time.time() - start3))

                if most_similar_name:
                    print("Authorization confirmed".format(most_similar_name))
                    self.beep_blink(2, g_led_pin, 0.3)
                else:
                    print("Alert! User not authorized detected")
                    self.beep_blink(1, r_led_pin, 1.5)

            if face_found and not learn_face_count.empty():

                self.beep_blink(8, g_led_pin, 0.1)
                start4 = time.time()
                most_similar_name, state_changed = self.learn_new_face(faces_boxes, frame)
                print("Time to learn face: {}".format(time.time() - start4))

                print("New face: {} was learned.".format(most_similar_name))

                learn_face_count.get()
                self.beep_blink(4, g_led_pin, 0.3)

                if state_changed:
                    self.save_db_state()

            if self.preview:
                self.show_detections(frame, faces_boxes, color)
                key = cv.waitKey(1)
                if key & 0xFF == ord('q'):
                    break

            fps.update()

        fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        cv.destroyAllWindows()
        vs.stop()
        time.sleep(2.0)

    # def check_face(self, frame):
    #
    #     face_boxes = self.face_det.extract_face(frame)
    #     frame_face_emb = None
    #     most_similar_emb = None
    #     most_similar_name = None
    #
    #     if np.any(face_boxes):
    #         frame_face_data = self.face_det.get_face_embeddings(face_boxes, frame)
    #         frame_face_emb = frame_face_data[0]['embedding']
    #         most_similar_name, most_similar_emb, match_map = self.find_face(frame_face_emb)
    #
    #     return face_boxes, most_similar_emb

    def detect_face(self, frame):
        face_boxes = self.face_det.extract_face(frame)
        faces_found = np.any(face_boxes)

        return faces_found, face_boxes

    def find_face(self, frame_face_emb):

        face_embs = None
        most_similar_name = None
        best_matches = 0
        match_map = None

        if self.faces_db:
            for name, face_data in self.faces_db['faces'].items():
                f_emb = np.asarray(face_data['embedding'])

                face_distances = self.face_det.euclidean_distance_vec(frame_face_emb, f_emb)
                match_map = face_distances <= 1.0
                matches = np.sum(match_map)
                if matches > best_matches:
                    best_matches = matches
                    face_embs = face_data
                    most_similar_name = name

        return most_similar_name, face_embs, match_map

    def learn_new_face(self, faces_boxes, frame):
        new_embs_added = False

        frame_face_data = self.face_det.get_face_embeddings(faces_boxes, frame)
        frame_face_emb = frame_face_data[0]['embedding']
        most_similar_name, face_embs, match_map = self.find_face(frame_face_emb)
        if most_similar_name is None:

            # Broadcast the sum of 5 to force unrecognizable faces at the start
            faces = np.zeros([5, 128], dtype=np.float) + 5
            faces[0] = frame_face_emb[0]
            face = {'embedding': faces}
            self.face_counter = self.face_counter + 1
            self.faces_db['face_count'] += 1
            most_similar_name = 'face {}'.format(self.faces_db['face_count'])
            self.faces_db['faces'][most_similar_name] = face
            new_embs_added = True

        elif most_similar_name and np.sum(match_map) < 5:

            free_face_slot_index = np.where(np.invert(match_map))
            index = free_face_slot_index[0][0]
            face_embs['embedding'][index] = frame_face_emb[0]
            self.faces_db['faces'][most_similar_name] = face_embs
            new_embs_added = True

        return most_similar_name, new_embs_added

    def save_db_state(self):
        start = time.time()
        with open(self.faces_db_file, 'w') as outfile:
            json.dump(self.faces_db, outfile, sort_keys=True, indent=4, cls=NumpyEncoder)
        print("Time to dump json file: {}".format(time.time() - start))

    @staticmethod
    def beep(beep_times, duration=0.1):
        if platform.uname()[1] == 'raspberrypi':
            for i in range(beep_times):
                GPIO.output(beep_pin, GPIO.HIGH)
                time.sleep(duration)
                GPIO.output(beep_pin, GPIO.LOW)
                time.sleep(duration)

    @staticmethod
    def red_blink(blink_times, duration=0.3):
        if platform.uname()[1] == 'raspberrypi':
            for i in range(blink_times):
                GPIO.output(r_led_pin, GPIO.HIGH)
                time.sleep(duration)
                GPIO.output(r_led_pin, GPIO.LOW)
                time.sleep(duration)

    @staticmethod
    def green_blink(blink_times, duration=0.3):
        if platform.uname()[1] == 'raspberrypi':
            for i in range(blink_times):
                GPIO.output(g_led_pin, GPIO.HIGH)
                time.sleep(duration)
                GPIO.output(g_led_pin, GPIO.LOW)
                time.sleep(duration)

    @staticmethod
    def beep_blink(blink_times, led_pin, duration=0.3):
        if platform.uname()[1] == 'raspberrypi':
            for i in range(blink_times):
                GPIO.output(beep_pin, GPIO.HIGH)
                GPIO.output(led_pin, GPIO.HIGH)
                time.sleep(duration)
                GPIO.output(beep_pin, GPIO.LOW)
                GPIO.output(led_pin, GPIO.LOW)
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
        return json.JSONEncoder.default(obj)
