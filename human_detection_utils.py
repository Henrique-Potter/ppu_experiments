import cv2 as cv


def show_detections(img_dbg, boxes, scores, classes, threshold):

    img_cp = img_dbg.copy()

    for i in range(len(boxes)):
        # Class 1 represents human
        if classes[i] == 1 and scores[i] > threshold:
            box = boxes[i]
            label = "Person: " + str(scores[i])

            cv.rectangle(img_cp, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)
            cv.putText(img_cp, label, (box[1], box[0] - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv.imshow("Debugging", img_cp)
