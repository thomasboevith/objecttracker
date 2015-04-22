import numpy as np
import sys

class Trackpoint:
    def __init__(self, x, y, size=None, color=None, shape=None):
        self.x = x
        self.y = y
        self.size = size
        self.color = color
        self.shape = shape

    def length_to(self, tp):
        """
        Calculate the lengt between two track points.
        """
        assert(isinstance(tp, Trackpoint))
        return np.sqrt((self.x - tp.x) ** 2 + (self.y - tp.y) ** 2)

    def direction_to(self, tp, deg=False):
        """
        Calculate the direction to the input point.
        If deg is set to True, the output is in degrees.
        """
        assert(isinstance(tp, Trackpoint))
        # archtan(x/y)
        direction = np.arctan2((tp.x - self.x), (tp.y-self.y))
        if deg:
            direction = np.rad2deg(direction)
        return direction

    def __str__(self):
        return "({x}, {y}) {size} {color} {shape}".format(x=self.x,
                                                              y=self.y,
                                                              size=self.size,
                                                              color=self.color,
                                                              shape=self.shape)
    def sort_by_closest(self, tracks):
        tracks.sort(key=lambda t: t.length_to(self))
"""

        min_length = sys.maxsize
        min_index = None

        # Find closest existing track.
        for i, t in enumerate(tracks):
            # Length to last point in current track.
            length = t.length_to(self)
            # cv2.circle(frame, (int(tp.x), int(tp.y)), 3, (0,0,255), 1)
            # cv2.putText(frame, "L: %.2f S_tp: %.2f S_t: %.2f"%(length, tp.size, t.avg_size()), (int(tp.x), int(tp.y)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 100, 100), thickness=1)

            # Find smallest length and index.
            if length < min_length:
                # import pdb; pdb.set_trace()
                min_length = length
                min_index = i

"""
