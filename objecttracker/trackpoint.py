import numpy as np


class Trackpoint:
    def __init__(self, row, col, size=None, color=None, shape=None):
        self.row = row
        self.col = col
        self.size = size
        self.color = color
        self.shape = shape

    def length_to(self, tp):
        """
        Calculate the lengt between two track points.
        """
        assert(isinstance(tp, Trackpoint))
        return np.sqrt((self.row - tp.row) ** 2 + (self.col - tp.col) ** 2)

    def direction_to(self, tp, deg=False):
        """
        Calculate the direction to the input point.
        If deg is set to True, the output is in degrees.
        """
        assert(isinstance(tp, Trackpoint))
        # archtan(x/y)
        direction = np.arctan2((tp.row - self.row), (tp.col-self.col))
        if deg:
            direction = np.rad2deg(direction)
        return direction

    def __str__(self):
        return "({row}, {col}) {size} {color} {shape}".format(row=self.row,
                                                              col=self.col,
                                                              size=self.size,
                                                              color=self.color,
                                                              shape=self.shape)
