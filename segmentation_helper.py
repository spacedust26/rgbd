import numpy as np

class SegmentationHelper:
    def __init__(self, min_depth=800, max_depth=3000):
        self.min_depth = min_depth
        self.max_depth = max_depth

    def segment(self, depth_map):
        return ((depth_map > self.min_depth) & (depth_map < self.max_depth)).astype(np.uint8)


# original dimensions min_depth=300, max_depth=1200