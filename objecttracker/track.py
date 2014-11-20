import numpy as np
import itertools


def pairwise(iterable):
    "Pairwise iteration: s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)


class Track:
    def __init__(self):
        self.parent = None
        self.trackpoints = []
        self.id = None
        self.parent_id = None
        self.children_ids = []

    def add_trackpoint(self, trackpoint):
        self.trackpoints.append(trackpoint)

    def set_parent(self, parent):
        self.parent = parent

    def length(self):
        sum = 0
        if self.number_of_trackpoints() > 0:
            for tp0, tp1 in pairwise(self.trackpoints):
                sum += np.sqrt((tp1.row - tp0.row)**2 + (tp1.col - tp0.col)**2)
        return sum

    def number_of_trackpoints(self):
        return len(self.trackpoints)

    def save(self, filename):
        with open(filename, 'w') as fp:
            for trackpoint in self.trackpoints:
                fp.write(trackpoint)
