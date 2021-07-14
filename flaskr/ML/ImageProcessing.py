from threading import Thread
from .Retinanet import Retinanet
import cv2

class ImageProcessing:
    def __init__(self, source=0):
        self.cv2 = cv2
        self.source = cv2.VideoCapture(source)
        (self.ret, self.frame) = self.source.read()
        self.stopped = False

    def start(self):
        Thread(target=self.read_source, args=()).start()
        return self

    def read_source(self):
        while not self.stopped:
            if(not self.ret):
                self.stop()
            else:
                (self.ret, self.frame) = self.source.read()
    
    def stop(self):
        self.stopped = True