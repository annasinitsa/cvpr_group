import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from random import shuffle
from glob import glob
class DatasetRetriever:
    def __init__(self, path_dir) -> object:
        self.dir = path_dir
        self.image_list = os.listdir(self.dir)
        self.image=None
        self.full_path=None

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx: int):
        self.im_name=self.image_list[idx]
        self.full_path = os.path.join(self.dir, self.im_name)
        self.image=self.transform(self.full_path)
        return self.image,self.im_name

    def transform(self, full_path):
        image = cv2.imread(full_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
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





def descriptor(trainset,testset):
    (train_image, name) = next(iter(trainset))
    orb = cv2.ORB_create()
    train_keypoints, train_descriptor = orb.detectAndCompute(train_image, None)
    for (test_image, name) in iter(testset[50:]):
        test_keypoints, test_descriptor = orb.detectAndCompute(test_image, None)
        train_descriptor = np.float32(train_descriptor)
        test_descriptor = np.float32(test_descriptor)
        MIN_MATCH_COUNT = 10
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=10)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(train_descriptor, test_descriptor, 2)
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good.append(m)
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([train_descriptor[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([test_keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            h, w, d = train_image.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            img2 = cv2.polylines(test_image, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        else:
            print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
            matchesMask = None

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)
        img3 = cv2.drawMatches(train_image, train_keypoints, test_image, test_keypoints, good, None, **draw_params)
        plt.imshow(img3, 'gray')
        plt.show()
        '''

        # Need to draw only good matches, so create a mask
        matchesMask = [[0, 0] for i in xrange(len(matches))]
        #bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(train_descriptor, test_descriptor,k=2)
        #matches = sorted(matches, key=lambda x: x.distance)
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append([m])
        result = cv2.drawMatchesKnn(train_image, train_keypoints, test_image, test_keypoints, good_matches,test_image,
                                    flags=2)
        plt.rcParams['figure.figsize'] = [150, 150]
        plt.title('Best Matching Points')
        plt.imshow(result)
        plt.show()
        print('Train KEY pIONTS', len(train_keypoints))
        print('Test KEY pIONTS', len(test_keypoints))
        print("Number of Matching Keypoints Between The Training and Query Images: ", len(matches))
        '''
#descriptor(train_set,test_set)

