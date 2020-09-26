from .ImageLoad import *

class ORB_descriptor:
    def __init__(self):
        self.orb = cv2.ORB_create()
