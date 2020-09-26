from ImageLoad import *
import datetime

class Metrics:
    def __init__(self, descriptor, train_dir, test_dir):
        self.correct_percent=None
        self.all_features=None
        self.correct_features=None
        self.time=None
        self.local_mistake=None
        self.train_keypoints=None
        self.train_descriptor=None
        self.test_keypoints = None
        self.test_descriptor = None
        self.train_dataset=DatasetRetriever(train_dir)
        self.test_dataset=DatasetRetriever(test_dir)
        self.orb=descriptor
        self.test_image=None
        self.train_image=None
        self.train_name=None
        self.test_name=None
        self.metrics=[]

    def __trainDescriptor__(self):
        (self.train_image, self.train_name) = next(iter(self.train_dataset))
        self.train_keypoints, self.train_descriptor = self.orb.detectAndCompute(self.train_image, None)
        return self.train_keypoints, self.train_descriptor, self.train_image, self.train_name

    def __testDescriptor__(self):
        self.train_keypoints, self.train_descriptor, self.train_image, self.train_name= self.__trainDescriptor__()
        for (self.test_image, self.test_name) in iter(self.test_dataset):
            print(self.test_name)
            start_time = datetime.datetime.now()
            self.test_keypoints, self.test_descriptor = self.orb.detectAndCompute(self.test_image, None)
            print('Test KEY pIONTS', len(self.test_keypoints))
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(self.train_descriptor, self.test_descriptor)
            end_time = datetime.datetime.now()
            time_diff = (end_time - start_time)
            self.time = time_diff.total_seconds() * 1000

            #matches = sorted(matches, key=lambda x: x.distance)
            #result = cv2.drawMatches(self.train_image, self.train_keypoints, self.test_image, self.test_keypoints, matches, self.test_image,
                                     #flags=2)
            #print("\nNumber of Matching Keypoints Between The Training and Query Images: ", len(matches))
            self.correct_features= len(matches) / len(self.test_keypoints)
            self.metrics.append({self.test_name:[self.correct_features,self.time]})
        return self.metrics

orb=cv2.ORB_create()
metrics=Metrics(orb,'../images/RB_dataset/train','../images/RB_dataset/test')
list_metrics=metrics.__testDescriptor__()
print(list_metrics)


