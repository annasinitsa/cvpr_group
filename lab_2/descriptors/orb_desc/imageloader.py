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

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx: int):
        if self.image_list[idx].split('.')[-1] in ['jpg','png']:
            self.im_name=self.image_list[idx]
            self.full_path = os.path.join(self.dir, self.im_name)
            self.image=self.transform(self.full_path)
            return self.image,self.im_name

    def transform(self, full_path):
        image = cv2.imread(full_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def visualize(image_loader):
        samples, im_name = next(iter(image_loader))
        fig = plt.figure(figsize=(40, 40))
        fig.suptitle("Some examples of images of the dataset", fontsize=16)
        iterator = iter(image_loader)
        samples, im_name = next(iterator)
        for i in range(len(samples)):
            plt.subplot(len(samples)//10, len(samples)//10, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(samples[i], cmap=plt.cm.binary)
        plt.show()


