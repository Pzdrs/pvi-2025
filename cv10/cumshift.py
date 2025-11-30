import cv2
import numpy as np


class CumShift:
    def __init__(self, object_template_path: str):
        template = cv2.imread(object_template_path)
        height, width = np.floor_divide(template.shape[:2], 2)
        template_hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
        h, _, _ = cv2.split(template_hsv)

        hist = cv2.calcHist([h], [0], None, [180], [0, 180])

        self.template_hist = (hist / hist.sum()).ravel()
        self.template_shape = (height, width)
        self.first = True
        self.prev_coords = (0, 0, 0, 0)

    def detect(self, frame):
        if self.first:
            self.first = False
            return self._first_frame(frame)
        else:
            return self._subsequent_frame(frame)

    def _first_frame(self, frame):
        h, _, _ = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
        backproj = self.template_hist[h]

        #cv2.imshow("Backproj", backproj)

        centroid = self._zeroth_moment(backproj)

        return self._bbox(centroid)

    def _subsequent_frame(self, frame):
        h, _, _ = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
        backproj = self.template_hist[h]

        #cv2.imshow("Backproj", backproj)

        x1, y1, x2, y2 = self.prev_coords
        backproj = backproj[y1:y2, x1:x2]

        tmp_x, tmp_y = self._zeroth_moment(backproj)
        x_t = int(x1 + tmp_x)
        y_t = int(y1 + tmp_y)

        return self._bbox((x_t, y_t)) 

    def _zeroth_moment(self, back_projection):
        y, x = np.indices(back_projection.shape)
        total = back_projection.sum()
        x_t = (x * back_projection).sum() / total
        y_t = (y * back_projection).sum() / total
        return x_t, y_t

    def _bbox(self, centroid):
        x_t, y_t = centroid
        h, w = self.template_shape

        x1 = abs(int(x_t - w))
        x2 = abs(int(x_t + w))
        y1 = abs(int(y_t - h))
        y2 = abs(int(y_t + h))

        self.prev_coords = (x1, y1, x2, y2)
        return (x1, y1, x2, y2)
