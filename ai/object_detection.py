import numpy as np
import os


class CocoDetectorAPI:

    def __init__(self, path_to_ckpt=None):

        if path_to_ckpt is None:
            path_to_ckpt = 'object_detection_models/frozen_inference_graph.pb'

        self.path_to_ckpt = os.path.expanduser(path_to_ckpt)

        import tensorflow as tf

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Defining input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent the level of confidence for each of the objects
        # Score is shown on the result image, together with the class label
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def process_frame(self, image, threshold, obj_class):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)

        (res_boxes, res_scores, res_classes, res_num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})

        im_height, im_width, _ = image.shape
        np_mask = np.array([im_height, im_width, im_height, im_width], dtype=np.int16)
        res_boxes[0, :, :] = res_boxes[0, :, :] * np_mask
        res_boxes = res_boxes.astype(np.int16)

        objects_detected_map = np.logical_and(res_scores[0] > threshold, res_classes[0] == obj_class)

        np_scores = np.take(res_scores[0], np.where(objects_detected_map))
        np_classes = np.take(res_classes[0], np.where(objects_detected_map))

        return res_boxes[0], np_scores, np_classes[0], res_num

    def close(self):
        self.sess.close()
        self.default_graph.close()

    @staticmethod
    def get_detected_objs(boxes, scores, classes, threshold, target=1):

        obj_boxes = []
        obj_scores = []

        for i in range(len(boxes)):
            if scores[i] == 0:
                break
            # Class 1 represents detected humans
            if classes[i] == target and scores[i] > threshold:
                obj_boxes.append(boxes[i])
                obj_scores.append(scores[i])

        return obj_boxes, obj_scores


