from metrics import Metrics
import cv2
import pandas as pd
import os

if __name__ == "__main__":
    orb = cv2.ORB_create()
    metrics = Metrics(orb, '../../images/RB_dataset/train', '../../images/RB_dataset/test')
    list_metrics = metrics.__testDescriptor__()
    metrics_df = pd.DataFrame.from_dict(list_metrics, orient='index',
                                        columns=['correct_features', 'local_mistake', 'time',
                                                 'image_size']).reset_index()
    metrics_df.to_csv("RB_dataset_metrics.csv",index=False)
