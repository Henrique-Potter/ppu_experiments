import numpy as np
import tensorflow as tf
import time
import os


class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = os.path.expanduser(path_to_ckpt)

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def process_frame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)

        (res_boxes, res_scores, res_classes, res_num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})

        im_height, im_width, _ = image.shape
        boxes_list = [None for i in range(res_boxes.shape[1])]
        for detection_index in range(res_boxes.shape[1]):
            boxes_list[detection_index] = (
                        int(res_boxes[0, detection_index, 0]*im_height),
                        int(res_boxes[0, detection_index, 1]*im_width),
                        int(res_boxes[0, detection_index, 2]*im_height),
                        int(res_boxes[0, detection_index, 3]*im_width))

        return boxes_list, res_scores[0].tolist(), [int(x) for x in res_classes[0].tolist()], int(res_num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()

    @staticmethod
    def get_detected_persons(boxes, scores, classes, threshold):

        h_boxes = []
        h_scores = []

        for i in range(len(boxes)):
            # Class 1 represents human
            if classes[i] == 1 and scores[i] > threshold:
                h_boxes.append(boxes[i])
                h_scores.append(scores[i])

        return h_boxes, h_scores


