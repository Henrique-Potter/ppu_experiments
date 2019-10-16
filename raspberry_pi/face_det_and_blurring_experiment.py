from ai import face_match as fm
import argparse
import numpy as np
from pathlib import Path


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

for image in images:

    import cv2 as cv

    img_base = cv.imread(str(Path(image).resolve()))
    cv.imshow("", img_base)
    input("Press Enter to continue...")

    for i in range(3, 10000, 2):

        img_base_faces_box = face_det.extract_face(img_base)
        faces_img = extract_faces_imgs(img_base, img_base_faces_box)

        for j in range(len(img_base_faces_box)):
            face_img = faces_img[j]
            face_box = img_base_faces_box[j]
            face = cv.blur(face_img, (3, 3))
            img_base[face_box[1]:face_box[3], face_box[0]:face_box[2], :] = face
            faces_img[j] = face

            cv.imshow("", img_base)

        print(i)


