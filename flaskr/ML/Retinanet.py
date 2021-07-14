# from .ImageProcessing import ImageProcessing
import cv2
from threading import Thread

from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu
import numpy as np
import os

# model_path = os.path.join('./model', 'firemodel_conveted.h5')
# model = models.load_model(model_path, backbone_name='resnet101')
# labels_to_names = {0: 'fail'}

class Retinanet:

    def __init__(self, model_name, backbone, labels, source):
        model_path = os.path.join('flaskr', 'ML', 'model', model_name)
        self.model = models.load_model(model_path, backbone_name=backbone)
        self.labels_to_names = labels
        self.processing_frame = None
        self.processed_frame = None
        self.source = source
        self.frame_to_show = None
        self.image_processing = ImageProcessing(source).start()

    def start(self):
        Thread(target=self.analyze, args=()).start()
        return self

    def analyze(self):
        frame = self.image_processing.frame
        ret, buffer = cv2.imencode('.jpg', frame)
        self.frame_to_show = buffer.tobytes()
        while True:
            frame = self.image_processing.frame
            print(frame)
            if(frame is not None):
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                image = preprocess_image(bgr)
                image, scale = resize_image(image, int(300), int(300))

                # process image
                boxes, scores, labels = self.model.predict(np.expand_dims(image, axis=0))
                # scale = 1

                # correct for image scale
                boxes /= scale

                # visualize detections
                for box, score, label in zip(boxes[0], scores[0], labels[0]):
                    # scores are sorted so we can break
                    if score < 0.05:
                        break
                    color = label_color(label)
                    
                    b = box.astype(int)
                    draw_box(frame, b, color=color)
                    
                    caption = "{} {:.3f}".format(self.labels_to_names[label], score)
                    draw_caption(frame, b, caption)
                
                ret, buffer = cv2.imencode('.jpg', frame)
                self.frame_to_show = buffer.tobytes()
            else:
                pass

class ImageProcessing:
    def __init__(self, source):
        self.cv2 = cv2
        self.source = cv2.VideoCapture(source)
        (self.ret, self.frame) = self.source.read()
        self.stopped = False
        self.frame_modify = None
        self.frame_to_show = None

    def start(self):
        self.frame_modify = self.frame
        self.frame_to_show = self.frame
        Thread(target=self.read_source, args=()).start()
        return self

    def read_source(self):
        while not self.stopped:
            if(not self.ret):
                self.stop()
            else:
                (self.ret, self.frame) = self.source.read()
                # self.retinanet.setProcessing(self.frame)
                # ret, buffer = cv2.imencode('.jpg', self.retinanet.processed_frame)
                # self.frame_to_show = buffer.tobytes()
    
    def stop(self):
        self.stopped = True