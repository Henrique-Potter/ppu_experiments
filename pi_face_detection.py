from imutils.video import VideoStream
from imutils.video import FPS
from ai import face_match as fm
from pathlib import Path
import numpy as np
import cv2 as cv
import platform
import time
import json
import os
import pandas as pd

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

        if os.path.exists("trigger_metrics.csv"):
            self.trigger_metrics_list = pd.read_csv("trigger_metrics.csv", index_col=False).values.tolist()
        else:
            self.trigger_metrics_list = []

    def id_face_trigger(self, sample_frames=10):

        color = blue_color
        frame_count = 0

        most_similar_name = None
        self.beep_blink(1, g_led_pin, 0.2)

        face_drawn_frame = None

        while frame_count < sample_frames:

            frame = self.get_photo()
            frame = cv.flip(frame, 0)

            start1 = time.time()
            face_found, faces_boxes = self.detect_face(frame)
            print("Time to detect face: {}".format(time.time() - start1))

            face_drawn_frame = self.show_detections(frame, faces_boxes)

            if face_found:
                self.beep_blink(2, g_led_pin, 0.1)

                start2 = time.time()
                frame_face_data = self.face_det.get_face_embeddings(faces_boxes, frame)
                print("Time to extract embeddings: {}".format(time.time() - start2))

                frame_face_emb = frame_face_data[0]['embedding']

                start3 = time.time()
                most_similar_name, most_similar_emb, match_map = self.find_face(frame_face_emb)
                print("Time to find face in DB: {}".format(time.time() - start3))

                if most_similar_name:
                    print("Authorization confirmed".format(most_similar_name))
                    self.beep_blink(3, g_led_pin, 0.3)
                else:
                    print("Alert! User not authorized detected")
                    self.beep_blink(1, r_led_pin, 1.5)

            if self.preview:
                img = self.show_detections(frame, faces_boxes)
                cv.imwrite("test_img.jpg", img)

            frame_count += 1

        frame_as_json = None
        data_dict = {}

        frame_as_json = self.compress_to_jpeg(data_dict, face_drawn_frame, frame_as_json, most_similar_name)

        return most_similar_name, frame_as_json

    @staticmethod
    def compress_to_jpeg(data_dict, face_drawn_frame, frame_as_json, face_name):
        if np.any(face_drawn_frame):
            import json
            encode_param = [int(cv.IMWRITE_JPEG_QUALITY), 90]
            result, encimg = cv.imencode('.jpg', face_drawn_frame, encode_param)

            data_dict['detection_frame'] = encimg.tolist()
            data_dict['person_name'] = face_name
            frame_as_json = json.dumps(data_dict)
        return frame_as_json

    def learn_face_trigger(self, sample_frames=5):

        self.beep_blink(1, g_led_pin, 0.2)
        vs = self.get_cam()
        time.sleep(2.0)
        frame_count = 0

        state_changed = False
        new_name = None

        while frame_count < sample_frames:

            frame = vs.read()
            frame = cv.flip(frame, 0)

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
                self.show_detections(frame, faces_boxes)
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

    # Used for measuring the accuracy of range sensor triggers
    def continuous_face_detection(self, process_queue):

        vs = self.get_cam()
        time.sleep(2.0)

        tm_counter = 10

        while True:

            if not process_queue.empty():
                trigger_data = process_queue.get()

                if trigger_data[0] == 1:
                    self.trigger_metrics_list.append(trigger_data)
                    tm_counter = tm_counter - 1
                    print('[INFO - Micro Lidar] Get received at:{} Save deadline:{}'.format(trigger_data, tm_counter))

                elif trigger_data[0] == 2:
                    self.trigger_metrics_list.append(trigger_data)
                    tm_counter = tm_counter - 1
                    print('[INFO - Sonar] Get received at:{} Save deadline:{}'.format(trigger_data, tm_counter))

            frame = vs.read()
            frame = cv.flip(frame, 0)

            start1 = time.time()
            face_found, faces_boxes = self.detect_face(frame)

            if face_found:
                self.beep_blink(1, g_led_pin, 0.5)
                time_stamp = time.time()
                self.trigger_metrics_list.append([0, time_stamp])
                tm_counter = tm_counter - 1
                print('[INFO - CAM] Get received at:{} Save deadline:{}'.format(time_stamp, tm_counter))
                print('[INFO - CAM] Time to detect face: {}'.format(time.time() - start1))
                time.sleep(5)

            if tm_counter < 1:
                total_data_df = pd.DataFrame(self.trigger_metrics_list)
                try:
                    total_data_df.to_csv("trigger_metrics.csv", index=False)
                except Exception as e:
                    print(e)
                tm_counter = 10
                print('[INFO] Saving trigger metrics.')
                print(total_data_df)

    def continuous_face_identification(self, process_queue):
        last_detection_time = 0

        vs = self.get_cam()
        time.sleep(2.0)
        fps = FPS().start()

        color = blue_color

        while True:

            if not process_queue.empty():
                msg_code = process_queue.get_nowait()
            else:
                msg_code = None

            frame = vs.read()

            # if platform.uname()[1] == 'raspberrypi':
            #     frame = cv.flip(frame, 0)

            start1 = time.time()
            face_found, faces_boxes = self.detect_face(frame)

            if face_found and msg_code is None:
                self.beep_blink(1, g_led_pin, 0.1)
                print("Time to detect face: {}".format(time.time() - start1))

                start2 = time.time()
                frame_face_data = self.face_det.get_face_embeddings(faces_boxes, frame)
                print("Time to extract embeddings: {}".format(time.time() - start2))

                frame_face_emb = frame_face_data[0]['embedding']

                start3 = time.time()
                most_similar_name, most_similar_emb, match_map = self.find_face(frame_face_emb)
                print("Time to find face in DB: {}".format(time.time() - start3))

                if most_similar_name:
                    print("Authorization for {} confirmed".format(most_similar_name))
                    self.beep_blink(2, g_led_pin, 0.3)
                else:
                    print("Alert! User not authorized detected")
                    self.beep_blink(1, r_led_pin, 1.5)

            if face_found and msg_code == 2:

                self.beep_blink(8, g_led_pin, 0.1)
                start4 = time.time()
                most_similar_name, state_changed = self.learn_new_face(faces_boxes, frame)
                print("Time to learn face: {}".format(time.time() - start4))

                print("New face: {} was learned.".format(most_similar_name))

                process_queue.get()
                self.beep_blink(4, g_led_pin, 0.3)

                if state_changed:
                    self.save_db_state()

            if self.preview:
                img = self.show_detections(frame, faces_boxes)
                key = cv.waitKey(1)
                cv.imshow('', img)
                if key & 0xFF == ord('q'):
                    break

            fps.update()

        fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        cv.destroyAllWindows()
        vs.stop()
        time.sleep(2.0)

    def detect_face(self, frame):
        face_boxes = self.face_det.extract_face(frame)
        faces_found = np.any(face_boxes)

        return faces_found, face_boxes

    def find_face(self, frame_face_emb):

        face_embs = None
        most_similar_name = None
        best_matches = 0
        best_match_map = None

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
                    best_match_map = match_map

        return most_similar_name, face_embs, best_match_map

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
    def get_photo(resolution=(320, 240)):
        if use_raspiberry:
            import picamera
            import picamera.array
            camera = picamera.PiCamera()
            try:
                camera.resolution = resolution
                rawCapture = picamera.array.PiRGBArray(camera, size=resolution)
                camera.capture(rawCapture, format="bgr")

                return rawCapture.array
            finally:
                camera.close()
        else:
            stream = cv.VideoCapture(0)
            time.sleep(1)
            try:
                (grabbed, frame) = stream.read()
                return frame
            finally:
                stream.release()

    @staticmethod
    def get_cam():
        if use_raspiberry:
            return VideoStream(usePiCamera=True).start()
        else:
            return VideoStream(src=0).start()

    @staticmethod
    def show_detections(img, f_boxes):
        img_cp = img.copy()
        for f_box in f_boxes:
            cv.rectangle(img_cp, (f_box[0], f_box[1]), (f_box[2], f_box[3]), (0,0,255), 2)
            cv.putText(img_cp, "Face", (f_box[2] + 10, f_box[3]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        return img_cp


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(obj)

