import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

class DatasetRetriever:
    def __init__(self, path_dir):
        self.dir = path_dir
        self.image_list = os.listdir(self.dir)
        self.image=None
        self.full_path=None
        self.im_name=None

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx: int):
        im_name=self.image_list[idx]
        full_path = os.path.join(self.dir, im_name)
        image=self.transform(full_path)
        return image,im_name

    def transform(self, full_path):
        image = cv2.imread(full_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def getimage(self,image_name):
        self.im_name = image_name
        self.full_path = os.path.join(self.dir, self.im_name)
        print(self.full_path)
        self.image = self.transform(self.full_path)
        return self.image