from imageloader import DatasetRetriever
import datetime
import matplotlib.pyplot as plt
import cv2


class Metrics:
    def __init__(self, descriptor, train_dir, test_dir):
        self.correct_percent = None
        self.all_features = None
        self.correct_features = None
        self.time = None
        self.local_mistake = 0
        self.train_dataset = DatasetRetriever(train_dir)
        self.test_dataset = DatasetRetriever(test_dir)
        self.orb = descriptor
        self.test_image = None
        self.train_image = None
        self.train_name = None
        self.test_name = None
        self.metrics = {}
        self.image_size = None

    def __trainDescriptor__(self):
        (self.train_image, self.train_name) = next(iter(self.train_dataset))
        self.train_keypoints, self.train_descriptor = self.orb.detectAndCompute(self.train_image, None)
        return self.train_keypoints, self.train_descriptor, self.train_image, self.train_name

    def __testDescriptor__(self):
        self.train_keypoints, self.train_descriptor, self.train_image, self.train_name = self.__trainDescriptor__()
        for (self.test_image, self.test_name) in iter(self.test_dataset):
            print(self.test_name)
            start_time = datetime.datetime.now()
            self.test_keypoints, self.test_descriptor = self.orb.detectAndCompute(self.test_image, None)
            end_time = datetime.datetime.now()
            time_diff = (end_time - start_time)
            self.time = time_diff.total_seconds()
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(self.train_descriptor, self.test_descriptor)
            matches = sorted(matches, key=lambda x: x.distance)
            result = cv2.drawMatches(self.train_image, self.train_keypoints, self.test_image, self.test_keypoints,
                                     matches, self.test_image, flags=2)
            self.correct_features = len(matches) / len(self.train_keypoints)
            for i in matches:
                self.local_mistake += i.distance
            self.local_mistake /= len(matches)
            self.image_size = self.test_image.shape[0] * self.test_image.shape[1]
            self.metrics.update(
                {self.test_name: [self.correct_features, self.local_mistake, self.time, self.image_size]})
        return self.metrics
