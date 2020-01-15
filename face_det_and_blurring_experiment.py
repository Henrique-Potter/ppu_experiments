from ai import face_match as fm
import argparse
import numpy as np
from pathlib import Path
import cv2 as cv
import time
import pandas as pd


def extract_faces_imgs(img, faces_boxes):
    faces_img = []

    if not len(faces_boxes) == 0:
        for box in faces_boxes:
            bb = np.empty(4, dtype=np.int16)
            bb[0] = box[0]
            bb[1] = box[1]
            bb[2] = box[2]
            bb[3] = box[3]

            face_img = img[bb[1]:bb[3], bb[0]:bb[2], :]

            faces_img.append(face_img)

    return faces_img

parser = argparse.ArgumentParser()

parser.add_argument("--fid_m", type=str, required=True)
parser.add_argument("--images", type=str, required=True)

args = parser.parse_args()
images = Path(args.images).glob("*.jpg")

face_det = fm.FaceMatch(args.fid_m)
once = 0

for image in images:

    img_base = cv.imread(str(Path(image).resolve()))

    input("Press Enter to continue...")

    box_iteration_results = []

    for iteration in range(1, 16):

        box_test_results = []

        for i in range(3, 32, 2):

            print("Iterations: {}".format(iteration))
            print("Box size: {}x{}".format(i, i))
            sample_times = []

            for a in range(20):

                img_blur = img_base.copy()
                start = time.time()
                img_base_faces_box = face_det.extract_face(img_blur)
                faces_img = extract_faces_imgs(img_blur, img_base_faces_box)




                sample_times.append(time.time()-start)
                #print("Number of faces: {}".format(len(img_base_faces_box)))

            box_test_results.append(np.average(sample_times))
            print(box_test_results)

        box_iteration_results.append(box_test_results)

    dump_df = pd.DataFrame(data=box_iteration_results)
    dump_df.to_csv("face_detection_blur_results.csv")

