from imageloader import DatasetRetriever
import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Metrics:
    def __init__(self, descriptor, train_dir, test_dir):
        self.train_dataset = DatasetRetriever(train_dir)
        self.test_dataset = DatasetRetriever(test_dir)
        self.orb = descriptor
        self.metrics = {}
        self.test_image_name=None
        self.test_image=None
        self.test_image_descriptor=None
        self.test_image_keypoints=None
        self.train_image_name = None
        self.train_image = None
        self.train_image_descriptor = None
        self.train_image_keypoints = None
        self.ratio=0.85

    def __trainDescriptor__(self):
        (self.train_image, self.train_name) = next(iter(self.train_dataset))
        self.train_keypoints,   self.train_descriptor = self.orb.detectAndCompute(self.train_image, None)
        self.train_descriptor = np.float32(self.train_descriptor)
        return self.train_keypoints, self.train_descriptor, self.train_image, self.train_name

    def __testDescriptor__(self):
        self.train_keypoints, self.train_descriptor, self.train_image, self.train_name = self.__trainDescriptor__()
        metrics = {}
        correct_features=0
        for (test_image, test_name) in iter(self.test_dataset):
            print(test_name)
            start_time = datetime.datetime.now()
            try:
                test_keypoints, test_descriptor = self.orb.detectAndCompute(test_image, None)
                test_descriptor = np.float32(test_descriptor)
                end_time = datetime.datetime.now()
                time_diff = (end_time - start_time)
                time = time_diff.total_seconds()
                bf = cv2.BFMatcher()
                try:
                    matches = bf.knnMatch(self.train_descriptor, test_descriptor, 2)
                    good_matches = []
                    for m, n in matches:
                        if m.distance < 0.75 * n.distance:
                            good_matches.append([m])
                    correct_features = len(good_matches) / len(test_keypoints)
                    try:
                        local_mistake=0
                        for i in range(len(good_matches)):
                            local_mistake += good_matches[i][0].distance
                        local_mistake /= len(good_matches)
                    except:
                        local_mistake = -1

                except TypeError:
                    print("BAD DESCRIPTOR")
                    local_mistake = -1
                    self.correct_features = 0

                image_size = test_image.shape[0] * test_image.shape[1]
                metrics.update(
                    {test_name: [correct_features, local_mistake, time, image_size]})
            except:
                print("CANNOT make the descriptor")
                local_mistake = -1
                correct_features = 0
                time=0
                image_size = test_image.shape[0] * test_image.shape[1]
                metrics.update({test_name: [correct_features, local_mistake, time, image_size]})

        return metrics

    def image_metrics(self,image_name):
        self.train_keypoints, self.train_descriptor, self.train_image, self.train_name = self.__trainDescriptor__()
        self.test_image=self.test_dataset.getimage(image_name)
        self.test_keypoints, self.test_descriptor = self.orb.detectAndCompute(self.test_image, None)
        test_descriptor = np.float32(self.test_descriptor)
        bf = cv2.BFMatcher()
        try:
            matches = bf.knnMatch(self.train_descriptor, test_descriptor, 2)
            try:
                good_matches = []
                for m, n in matches:
                    if m.distance < self.ratio * n.distance:
                        good_matches.append([m])
                result = cv2.drawMatchesKnn(self.train_image,self.train_keypoints,self.test_image,self.test_keypoints,good_matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                plt.imshow(result)
                plt.show()
            except: print("no good matches")
        except:
            print("CANNOT make the descriptor")

    def execute(self):
        self.metrics=self.__testDescriptor__()
        return self.metrics



