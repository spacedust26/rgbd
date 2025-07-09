import numpy as np

class AnnotationWriter:
    def __init__(self, label_class=0, normalized=True):
        self.label_class = label_class
        self.normalized = normalized

    def write(self, filepath, mask, img_shape):
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return False

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        w = x_max - x_min
        h = y_max - y_min

        if self.normalized:
            height, width = img_shape
            cx /= width
            cy /= height
            w /= width
            h /= height

        with open(filepath, "w") as f:
            f.write(f"{self.label_class} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        return True
