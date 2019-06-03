import numpy as np
import facenet
from face_detection import detect_face
import cv2
import matplotlib.pyplot as plt


class FaceMatch:

    def __init__(self, model_path="face_id_models/20170512-110547.pb"):
        # some constants kept as default from facenet
        self.minsize = 20
        self.threshold = [0.6, 0.7, 0.7]
        self.factor = 0.709
        self.margin = 44
        self.input_image_size = 160

        import tensorflow as tf

        self.sess = tf.Session()

        # read pnet, rnet, onet models from face_detection directory and files are det1.npy, det2.npy, det3.npy
        self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.sess, 'face_detection')

        # read 20170512-110547 model file downloaded from https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk
        facenet.load_model(model_path)

        if model_path == "face_id_models/20170512-110547.pb":
            self.distance_model = 'euclidean'
        elif model_path == "face_id_models/20180402-114759.pb":
            self.distance_model = 'cosine'

        # Get input and output tensors
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        self.embedding_size = self.embeddings.get_shape()[1]

    def extract_face(self, img):

        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = detect_face.detect_face(img, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
        faces_boxes = np.zeros([len(bounding_boxes), 4], dtype=np.int16)
        boxes_nr = len(bounding_boxes)
        if not boxes_nr == 0:
            for idx in range(boxes_nr):
                if bounding_boxes[idx, 4] > 0.50:
                    det = np.squeeze(bounding_boxes[idx, 0:4])
                    faces_boxes[idx, 0] = np.maximum(det[0] - self.margin / 2, 0)
                    faces_boxes[idx, 1] = np.maximum(det[1] - self.margin / 2, 0)
                    faces_boxes[idx, 2] = np.minimum(det[2] + self.margin / 2, img_size[1])
                    faces_boxes[idx, 3] = np.minimum(det[3] + self.margin / 2, img_size[0])

        return faces_boxes

    def get_face_embeddings(self, faces_boxes, img, debug_faces=False):

        embeddings = []
        if not len(faces_boxes) == 0:
            for box in faces_boxes:

                bb = np.empty(4, dtype=np.int16)
                bb[0] = box[0]
                bb[1] = box[1]
                bb[2] = box[2]
                bb[3] = box[3]

                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                resized = cv2.resize(cropped, (self.input_image_size, self.input_image_size), interpolation=cv2.INTER_CUBIC)
                prewhiten = facenet.prewhiten(resized)

                # if debug_faces:
                #     plt.subplot(132), plt.imshow(cropped), plt.title('Cropped')
                #     plt.subplot(133), plt.imshow(prewhiten), plt.title('Whited and Resized to 160 x 160')
                #     plt.xticks([]), plt.yticks([])
                #     plt.show()

                    #faces.append({'face': resized, 'rect': [bb[0], bb[1], bb[2], bb[3]], 'embedding': self.get_embedding(prewhiten)})

                embeddings.append({'face': resized, 'embedding': self.get_embedding(prewhiten)})

        return embeddings

    def get_embedding(self, resized):
        reshaped = resized.reshape(-1, self.input_image_size, self.input_image_size, 3)
        feed_dict = {self.images_placeholder: reshaped, self.phase_train_placeholder: False}
        embedding = self.sess.run(self.embeddings, feed_dict=feed_dict)
        return embedding

    def compare_faces(self, img1, img2, debug_faces=False):
        if self.distance_model == 'euclidean':
            return self.compare_faces_ed(img1, img2, debug_faces)
        elif self.distance_model == 'cosine':
            return self.compare_faces_cd(img1, img2, debug_faces)

    # Calculates distance based in Euclidian Distance
    def compare_faces_ed(self, img1, img2, debug_faces):
        boxes1 = self.extract_face(img1)
        face1 = self.get_face_embeddings(boxes1, img1, debug_faces)

        boxes2 = self.extract_face(img2)
        face2 = self.get_face_embeddings(boxes2, img2, debug_faces)
        if face1 and face2:
            dist = np.sqrt(np.sum(np.square(np.subtract(face1[0]['embedding'], face2[0]['embedding']))))
            return dist
        else:
            return -1

    def compare_faces_cropped(self, boxes1, boxes2, img1, img2, debug_faces=False):

        face1 = self.get_face_embeddings(boxes1, img1, debug_faces)
        face2 = self.get_face_embeddings(boxes2, img2, debug_faces)

        if face1 and face2:
            dist = np.sqrt(np.sum(np.square(np.subtract(face1[0]['embedding'], face2[0]['embedding']))))
            return dist
        else:
            return -1

    # Calculates distance based in Cosine Difference. Meant to be used with 20180402-114759 model
    def compare_faces_cd(self, img1, img2, debug_faces=False):
        boxes1 = self.extract_face(img1)
        face1 = self.get_face_embeddings(boxes1, img1, debug_faces)

        boxes2 = self.extract_face(img2)
        face2 = self.get_face_embeddings(boxes2, img2, debug_faces)
        if face1 and face2:
            dist = self.cosine_similarity(face1[0]['embedding'], face2[0]['embedding'])
            return dist
        else:
            return -1

    @staticmethod
    def euclidean_distance(face1, face2):
        dist = np.sqrt(np.sum(np.square(np.subtract(face1, face2))))
        return dist

    @staticmethod
    def cosine_similarity(x, y):
        return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))
