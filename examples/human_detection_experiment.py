from ai import object_detection as hd
import experiment_functions as hdu
import cv2 as cv


def blur_avg_box_experiment(img_base, iter_max):

    scores_list = []
    blur_iterations = []
    for i in range(1, iter_max):
        img_base = cv.blur(img_base, (5, 5))
        boxes, scores, classes, num = odapi.process_frame(img_base)
        print(scores)

        blur_iterations.append(i)

        hdu.show_detections(img_base, boxes, scores, classes, 0.5)

        key = cv.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    return scores_list


model_path = 'object_detection_models/frozen_inference_graph.pb'
odapi = hd.CocoDetectorAPI(path_to_ckpt=model_path)
threshold = 0.7

img1 = cv.imread("./images/obama_alone_office.jpg")

img = cv.resize(img1, (1280, 720))

blur_avg_box_experiment(img, 490)


