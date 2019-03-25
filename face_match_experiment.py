import argparse
import cv2
import matplotlib.pyplot as plt
from face_match import FaceMatch
import time

parser = argparse.ArgumentParser()
parser.add_argument("--img1", type=str, required=True)
parser.add_argument("--img2", type=str, required=True)
args = parser.parse_args()


def sample_demo(img_base, img_target, face_match):

    distances_found = []
    img_target_blur = 0
    for i in range(1, 5):
        img_target_blur = cv2.blur(img_target, (5 + i * 2, 5 + i * 2))

        my_distance = face_match.compare_faces(img_base, img_target_blur)
        distances_found.append(my_distance)

    return img_target_blur


def blur_avg_rep_experiment(img_base, img_target, face_match, max_repetitions=40):

    distances_found = []
    reps_list = []
    blurred_img = img_target.copy()
    for reps in range(1, max_repetitions):

        distance = 0

        blurred_img = cv2.blur(blurred_img, (5, 5))

        distance = face_match.compare_faces(img_base, blurred_img)

        print("Blur AVG Rep experiment - Iteration {} distance {}".format(reps, distance))
        if distance == -1:
            print("Face detection/align failed")
            break
        reps_list.append(reps)
        distances_found.append(distance)

    print(distances_found)

    return reps_list, distances_found


def blur_avg_box_experiment(img_base, img_target, face_match, box_max_size=40):

    distances_found = []
    box_sizes = []

    for i in range(1, box_max_size):
        img_target_blur = cv2.blur(img_target, (5 + i, 5 + i))
        distance = face_match.compare_faces(img_base, img_target_blur)
        print("Box size AVG experiment - Iteration {} box size {} distance {}".format(i, i+5, distance))
        if distance == -1:
            print("Face detection/align failed")
            break
        distances_found.append(distance)
        box_sizes.append(5 + i)
    print(distances_found)

    return box_sizes, distances_found


fm = FaceMatch()

#img1 = cv2.imread(args.img1)
#img2 = cv2.imread(args.img2)

#img1 = cv2.imread("./images/daniel-radcliffe.jpg")
#img1 = cv2.imread("./images/Barack_Obama.jpg")
img1 = cv2.imread("./images/Obama_signs.jpg")
img2 = cv2.imread("./images/obama_alone_office.jpg")

start_time = time.time()
face_match_distance = fm.compare_faces(img1, img2, use_white=True)
print("--- %s seconds ---" % (time.time() - start_time))

face_match_threshold = 1.10

print("distance = " + str(face_match_distance))
print("Result = " + ("same person" if face_match_distance <= face_match_threshold else "not same person"))

sample_demo(img1, img2, fm)

# Box size experiment
box_sizes, distances = blur_avg_box_experiment(img1, img2, fm, 10)
#plt.subplot(211)
plt.plot(box_sizes, distances)
#plt.title('Box size and Repetition experiments')
plt.xlabel('Blur Box Size(px)')
plt.ylabel('Euclidean Distance')

# Blur repetitions experiment
reps, rep_distances = blur_avg_rep_experiment(img1, img2, fm, 10)
#plt.subplot(212)
plt.plot(reps, rep_distances)
plt.xlabel('Number of Blurring rounds/Box Sizes')
plt.ylabel('Euclidean Distance')
#plt.subplots_adjust(hspace=.63)
plt.legend()
plt.show()

# fm_new = FaceMatch("20180402-114759/20180402-114759.pb")
# face_match_distance = fm_new.compare_faces(img1, img2)
#
# sample_demo(img1, img2, fm_new)
# plt.show()
# box_avg_experiment(img1, img2, fm_new)
# plt.show()

