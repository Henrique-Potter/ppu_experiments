import human_detection as hd
import cv2 as cv


if __name__ == "__main__":
    model_path = 'faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
    odapi = hd.DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.7

    cap = cv.VideoCapture('video/TownCentreXVID.avi')

    while True:
        r, img = cap.read()
        img = cv.resize(img, (1280, 720))

        boxes, scores, classes, num = odapi.process_frame(img)

        # Visualization of the results of a detection.
        for i in range(len(boxes)):
            # Class 1 represents human
            if classes[i] == 1 and scores[i] > threshold:
                box = boxes[i]
                label = "Person: " + str(scores[i])
                cv.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)
                cv.putText(img, label, (box[1], box[0]-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv.imshow("preview", img)
        key = cv.waitKey(1)
        if key & 0xFF == ord('q'):
            break

