import numpy as np

def frame_to_bgr_image(color_frame):
    return np.asanyarray(color_frame.get_data())
