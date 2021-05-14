import numpy as np
from VideoToolkit.tools import get_cv_resize_function, bbox_from_mask, rescal_to_image


def threshold(frame, t):
    res = np.zeros(frame.shape, np.uint8)
    mask = np.where(frame > t)
    res[mask] = 1

    return res


class FrameWindow:
    def __init__(self, shape, window_size=15):
        self.window_size = window_size
        self.shape = shape
        self.window = [None for _ in range(self.window_size)]
        self.index = 0
        self.num = 0

    def inc_num(self):
        if self.num < self.window_size:
            self.num += 1

    def set_new_frame(self, frame):
        if frame.shape != self.shape:
            raise ValueError(
                f"expected image with shape {self.shape} bot got image with shape {frame.shape}"
            )

        self.window[self.index] = frame.astype("float32")
        self.index += 1
        self.index = self.fix_index(self.index)
        self.inc_num()

    def fix_index(self, index):
        return index % self.window_size


# TODO: convert to functional programming instead of class
class mean_var_window(FrameWindow):
    def __init__(self, shape, window_size=15):
        super().__init__(shape, window_size)

    def get_var(self):
        return np.var(self.window[: self.num], axis=0)

    def get_mean(self):
        return np.mean(self.window[: self.num], axis=0)

    def get_var_with_mean(self, mean):
        # TODO: use num
        if mean.shape != self.shape:
            raise ValueError(
                f"expected image with shape {self.shape} bot got image with shape {mean.shape}"
            )

        s = np.zeros(mean.shape, np.float32)
        m = self.get_mean()
        for f in self.window:
            sub = m - f
            s = s + np.multiply(sub, sub)

        var = s / self.window_size
        return var


class StatBS:
    def __init__(self, shape):
        self.longmvobj = mean_var_window(shape, window_size=100)
        self.long_mean = None
        self.long_var = None
        self.frame = None

    def apply(self, frame):
        self.longmvobj.set_new_frame(frame)
        self.long_mean = self.longmvobj.get_mean()
        self.long_var = self.longmvobj.get_var()

    def get_bmask(self, frame):
        return rescal_to_image(threshold(np.abs(self.long_mean - frame), 10))

    def get_uncertainty_maps(self, frame):
        mtx = np.matrix(self.long_var)
        var_threshold = mtx.mean()

        certain_fore = np.zeros(self.long_mean.shape, np.uint8)
        certain_fore[
            np.where(
                np.abs(frame.astype(np.float32) - self.long_mean)
                > (2.5 * np.sqrt(self.long_var))
            )
        ] = 1
        certain_fore[np.where(self.long_var < var_threshold)] = 0

        certain_back = np.zeros(self.long_var.shape, np.uint8)
        certain_back[
            np.where(
                np.abs(frame.astype(np.float32) - self.long_mean)
                < (1 * np.sqrt(self.long_var))
            )
        ] = 1
        certain_back[np.where(self.long_var < var_threshold)] = 1

        uncertain = np.abs(frame.astype(np.float32) - self.long_mean)
        # uncertain[np.where(certain_back == 1)] == 0
        # uncertain[np.where(certain_fore == 1)] == 0

        return certain_fore, certain_back, uncertain
