import argparse

from imutils.video import VideoStream
from imutils.video import FPS
from multiprocessing import Process
from multiprocessing import Queue
import imutils
import experiment_functions
import face_match as fm
import numpy as np
import cv2 as cv
import time
import os


class Peyes:

    def check_face(self, model_path, input_queue, output_queue, learned_faces, debug=False):

        face_det = fm.FaceMatch(model_path)

        while True:

            if not input_queue.empty():

                frame = input_queue.get()

                face_boxes = face_det.extract_face(frame)

                if np.any(face_boxes):
                    frame_face_data = face_det.get_face_embeddings(face_boxes, frame, debug)
                    frame_face_data[0]['face_boxes'] = face_boxes
                    frame_face_emb = frame_face_data[0]['embedding']

                    for face_emb in learned_faces:
                        distance = face_det.euclidean_distance(frame_face_emb, face_emb)

                        if distance < 1.1:
                            frame_face_data[0]['distance'] = distance

                    # write the detections to the output queue
                    output_queue.put(frame_face_data)

    def learn_new_face(self, face_emb, learned_faces):
        print("Learning new face")
        learned_faces.append(face_emb)

    def run(self):

        parser = argparse.ArgumentParser()
        parser.add_argument("--fid_m", type=str, required=True)
        #parser.add_argument("--hd_m", type=str, required=True)
        #parser.add_argument("--hd_thres", type=float, required=True)
        parser.add_argument("--preview", type=bool, required=True)

        args = parser.parse_args()

        dir_path = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(dir_path, args.fid_m)

        experiment_utils = experiment_functions.BlurExperiments
        #face_det = fm.FaceMatch(model_path)
        #human_det = hd.DetectorAPI(path_to_ckpt = args.hd_m)

        learned_faces = []
        input_queue = Queue(maxsize=1)
        output_queue = Queue(maxsize=1)
        face_detections = None

        print("[INFO] starting process...")
        p = Process(target=self.check_face, args=(model_path, input_queue, output_queue, learned_faces))
        p.daemon = True
        p.start()

        # vs = VideoStream(usePiCamera=True).start()
        vs = VideoStream(src=0).start()
        time.sleep(2.0)
        fps = FPS().start()
        f_boxes = []

        while True:

            frame = vs.read()
            frame = imutils.resize(frame, width=400)
            (fH, fW) = frame.shape[:2]

            if input_queue.empty():
                input_queue.put(frame)

            if not output_queue.empty():
                face_detections = output_queue.get()

            start = time.time()

            if face_detections is not None:

                f_emb = face_detections[0]['embedding']
                f_boxes = face_detections[0]['face_boxes']
                #f_dist = face_detections[0]['distance']

                if 'distance' in face_detections[0]:
                    print("Face found")
                else:
                    self.learn_new_face(f_emb, learned_faces)

            if args.preview is True:
                experiment_utils.show_detections(frame, [], f_boxes, 0, 0, 0.5)
                key = cv.waitKey(1)
                if key & 0xFF == ord('q'):
                    break

            end = time.time()
            #print("Time to process frame time: {}".format(end - start))

            fps.update()

        fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        cv.destroyAllWindows()
        vs.stop()


        #boxes, scores, classes, humans_detected_map, num = human_det.process_frame(frame, args.hd_thres, 1)
        #h_boxes, h_scores = human_det.get_detected_persons(boxes, scores, classes, hd_threshold)


if __name__ == "__main__":
    peyes = Peyes()
    peyes.run()
