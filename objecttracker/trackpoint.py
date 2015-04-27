import numpy as np
import sys

class Trackpoint:
    def __init__(self, x, y, frame, size=None, color=None, shape=None):
        self.x = x
        self.y = y
        self.size = size
        self.color = color
        self.shape = shape
        self.frame = frame

    def copy(self):
        return Trackpoint(self.x, self.y, self.frame, self.size, self.color, self.shape)

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
        return tracks
