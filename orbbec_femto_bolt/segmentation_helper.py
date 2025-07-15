#import numpy as np

#class SegmentationHelper:
#    def __init__(self, min_depth=300, max_depth=1200):
#        self.min_depth = min_depth
#        self.max_depth = max_depth

#    def segment(self, depth_map):
#        return ((depth_map > self.min_depth) & (depth_map < self.max_depth)).astype(np.uint8)

import numpy as np
import cv2

class SegmentationHelper:
    def __init__(self, min_depth=300, max_depth=1200, roi_ratio=0.5, min_area=1000, max_area=50000):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.roi_ratio = roi_ratio  
        self.min_area = min_area
        self.max_area = max_area

    def segment(self, depth_map):
        # Step 1: Create depth-based binary mask
        depth_mask = (depth_map > self.min_depth) & (depth_map < self.max_depth)

        # Step 2: Focus on central ROI
        h, w = depth_map.shape
        roi_h = int(h * self.roi_ratio)
        roi_w = int(w * self.roi_ratio)
        start_h = (h - roi_h) // 2
        start_w = (w - roi_w) // 2

        roi_mask = np.zeros_like(depth_mask, dtype=bool)
        roi_mask[start_h:start_h + roi_h, start_w:start_w + roi_w] = True

        # Step 3: Combine ROI and depth range
        combined_mask = np.logical_and(depth_mask, roi_mask).astype(np.uint8)

        # Step 4: Keep only largest valid contour (likely object)
        contours, _ = cv2.findContours((combined_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_filtered = np.zeros_like(combined_mask)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area < area < self.max_area:
                cv2.drawContours(mask_filtered, [cnt], -1, 1, thickness=cv2.FILLED)

        return mask_filtered
