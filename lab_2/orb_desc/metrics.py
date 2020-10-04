from imageloader import DatasetRetriever
import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt


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
        self.train_descriptor = np.float32(self.train_descriptor)
        return self.train_keypoints, self.train_descriptor, self.train_image, self.train_name

    def __testDescriptor__(self):
        self.train_keypoints, self.train_descriptor, self.train_image, self.train_name = self.__trainDescriptor__()
        metrics = {}
        for (self.test_image, self.test_name) in iter(self.test_dataset):
            print(self.test_name)
            start_time = datetime.datetime.now()
            try:
                self.test_keypoints, self.test_descriptor = self.orb.detectAndCompute(self.test_image, None)
                self.test_descriptor = np.float32(self.test_descriptor)
                end_time = datetime.datetime.now()
                time_diff = (end_time - start_time)
                self.time = time_diff.total_seconds()
                FLANN_INDEX_KDTREE = 0
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict()  # or pass empty dictionary
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                try:
                    matches = flann.knnMatch(self.train_descriptor, self.test_descriptor, 2)
                    ratio = 0.55
                    good_matches = []
                    matchesMask = [[0, 0] for i in range(len(matches))]

                    # ratio test as per Lowe's paper
                    for i, (m, n) in enumerate(matches):
                        if m.distance < ratio * n.distance:
                            good_matches.append(m)

                    draw_params = dict(matchColor=(0, 255, 0),
                                       singlePointColor=(255, 0, 0),
                                       matchesMask=matchesMask,
                                       flags=0)
                    print(len(good_matches))
                    print(len(self.test_keypoints))
                    plt.show()
                    self.correct_features = len(good_matches) / len(self.test_keypoints)
                    try:
                        for i in good_matches:
                            self.local_mistake += i.distance
                        self.local_mistake /= len(good_matches)
                    except:
                        self.local_mistake = -1

                except TypeError:
                    print("BAD DESCRIPTOR")
                    self.local_mistake = -1
                    self.correct_features = 0

                self.image_size = self.test_image.shape[0] * self.test_image.shape[1]
                metrics.update(
                    {self.test_name: [self.correct_features, self.local_mistake, self.time, self.image_size]})
            except:
                print("CANNOT make the descriptor")
                self.local_mistake = -1
                self.correct_features = 0

                self.image_size = self.test_image.shape[0] * self.test_image.shape[1]
                metrics.update({self.test_name: [self.correct_features, self.local_mistake, self.time, self.image_size]})

        return metrics

    def execute(self):
        self.metrics=self.__testDescriptor__()
        return self.metrics



