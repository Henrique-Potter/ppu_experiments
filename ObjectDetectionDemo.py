import platform
from imutils.video import VideoStream
from imutils.video import FPS
from ai import face_match as fm
from pathlib import Path
import numpy as np
import cv2 as cv
import platform
import time
import os
from ai import object_detection as hd


blue_color = (255, 0, 0)
green_color = (0, 255, 0)
red_color = (0, 0, 255)

beep_pin = 40
g_led_pin = 36
r_led_pin = 38

face_det_model_path = 'face_id_models/20170512-110547.pb'
dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(dir_path, face_det_model_path)

path = "object_detection_models/frozen_inference_graph.pb"

face_det = fm.FaceMatch(model_path)
human_det = hd.CocoDetectorAPI(path_to_ckpt=path)

use_raspiberry = False
if platform.uname()[1] == 'raspberrypi':
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BOARD)

    GPIO.setup(beep_pin, GPIO.OUT, initial=0)
    GPIO.setup(g_led_pin, GPIO.OUT, initial=0)
    GPIO.setup(r_led_pin, GPIO.OUT, initial=0)

    use_raspiberry = True


def detect_face(frame):

    face_boxes = face_det.extract_face(frame)
    faces_found = np.any(face_boxes)

    return faces_found, face_boxes


def get_cam():
    if use_raspiberry:
        return VideoStream(usePiCamera=True).start()
    else:
        return VideoStream(src=0).start()


# def show_detections(img, f_boxes):
#     img_cp = img.copy()
#     for f_box in f_boxes:
#         cv.rectangle(img_cp, (f_box[0], f_box[1]), (f_box[2], f_box[3]), (0, 0, 255), 2)
#         cv.putText(img_cp, "Face", (f_box[2] + 10, f_box[3]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
#
#     return img_cp


def show_detections(img_cp, h_boxes, f_boxes, scores, obj_map, threshold, window_title='Debugging'):
    import cv2 as cv

    img_temp = img_cp.copy()

    cv.putText(img_temp, window_title, (5, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

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


if __name__ == "__main__":
    vs = get_cam()
    time.sleep(2.0)
    while True:

        frame = vs.read()
        frame = cv.flip(frame, 0)

        start1 = time.time()
        
        #face_found, faces_boxes = detect_face(frame)
        h_boxes, h_scores, obj_map, num = human_det.process_frame(frame, 0.7, 1)
        frame = show_detections(frame, h_boxes, [], h_scores, obj_map,0.7)

        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break


