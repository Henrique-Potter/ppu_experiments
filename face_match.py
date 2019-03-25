import tensorflow as tf
import numpy as np
import facenet
from align import detect_face
import cv2
import matplotlib.pyplot as plt


class FaceMatch:

    def __init__(self, model_path="20170512-110547/20170512-110547.pb"):
        # some constants kept as default from facenet
        self.minsize = 20
        self.threshold = [0.6, 0.7, 0.7]
        self.factor = 0.709
        self.margin = 44
        self.input_image_size = 160

        self.sess = tf.Session()

        # read pnet, rnet, onet models from align directory and files are det1.npy, det2.npy, det3.npy
        self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.sess, 'align')

        # read 20170512-110547 model file downloaded from https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk
        facenet.load_model(model_path)

        if model_path == "20170512-110547/20170512-110547.pb":
            self.distance_model = 'euclidean'
        elif model_path == "20180402-114759/20180402-114759.pb":
            self.distance_model = 'cosine'

        # Get input and output tensors
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        self.embedding_size = self.embeddings.get_shape()[1]

    def get_face(self, img, use_white, debug_faces):
        faces = []
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = detect_face.detect_face(img, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
        if not len(bounding_boxes) == 0:
            for face in bounding_boxes:
                if face[4] > 0.50:
                    det = np.squeeze(face[0:4])
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0] - self.margin / 2, 0)
                    bb[1] = np.maximum(det[1] - self.margin / 2, 0)
                    bb[2] = np.minimum(det[2] + self.margin / 2, img_size[1])
                    bb[3] = np.minimum(det[3] + self.margin / 2, img_size[0])

                    cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                    resized = cv2.resize(cropped, (self.input_image_size, self.input_image_size), interpolation=cv2.INTER_CUBIC)

                    if use_white:
                        resized = facenet.prewhiten(resized)

                    if debug_faces:
                        plt.subplot(131), plt.imshow(img), plt.title('Original')
                        plt.subplot(132), plt.imshow(cropped), plt.title('Cropped')
                        if use_white:
                            plt.subplot(133), plt.imshow(resized), plt.title('Whited')
                        plt.xticks([]), plt.yticks([])
                        plt.show()

                    faces.append({'face': resized, 'rect': [bb[0], bb[1], bb[2], bb[3]], 'embedding': self.get_embedding(resized)})
        return faces

    def get_embedding(self, resized):
        reshaped = resized.reshape(-1, self.input_image_size, self.input_image_size, 3)
        feed_dict = {self.images_placeholder: reshaped, self.phase_train_placeholder: False}
        embedding = self.sess.run(self.embeddings, feed_dict=feed_dict)
        return embedding

    def compare_faces(self, img1, img2, use_white=True, debug_faces=False):
        if self.distance_model == 'euclidean':
            return self.compare_faces_ed(img1, img2, use_white, debug_faces)
        elif self.distance_model == 'cosine':
            return self.compare_faces_cd(img1, img2, use_white, debug_faces)

    # Calculates distance based in Euclidian Distance
    def compare_faces_ed(self, img1, img2, use_white, debug_faces):
        face1 = self.get_face(img1, use_white, debug_faces)
        face2 = self.get_face(img2, use_white, debug_faces)
        if face1 and face2:
            dist = np.sqrt(np.sum(np.square(np.subtract(face1[0]['embedding'], face2[0]['embedding']))))
            return dist
        else:
            return -1

    # Calculates distance based in Cosine Difference. Meant to be used with 20180402-114759 model
    def compare_faces_cd(self, img1, img2, use_white=True, debug_faces=False):
        face1 = self.get_face(img1, use_white, debug_faces)
        face2 = self.get_face(img2, use_white, debug_faces)
        if face1 and face2:
            dist = self.cosine_similarity(face1[0]['embedding'], face2[0]['embedding'])
            return dist
        else:
            return -1

    @staticmethod
    def cosine_similarity(x, y):
        return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))
