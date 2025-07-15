import numpy as np

class SegmentationHelper:
    def __init__(self, min_depth=300, max_depth=1200):  # in millimeters
        self.min_depth = min_depth
        self.max_depth = max_depth

    def segment(self, depth_image):
        return ((depth_image > self.min_depth) & (depth_image < self.max_depth)).astype(np.uint8)
