import numpy as np
from trackpoint import Trackpoint


class Track:
    def __init__(self):
        self.parent = None
        self.trackpoints = []
        self.name = None

    def add_trackpoint(self, trackpoint):
        """
        Adding a trackpoint to the track.
        """
        assert(isinstance(trackpoint, Trackpoint))
        self.trackpoints.append(trackpoint)

    def set_parent(self, parent):
        """
        If a track splits up into two new tracks, the parent track
        should be connected to the new tracks.
        """
        assert(isinstance(parent, Track))
        self.parent = parent

    def split(self):
        """
        Splits the track into two new tracks.
        """
        return self._cut(), self._cut()

    def _cut(self):
        """
        Creates a new track..
        """
        t = Track()
        t.set_parent(self)
        return t

    def length(self, include_parents=False):
        """
        The length of a track, including its parent track.

        Recursively calls it self on the parents, if any.

        Example:
        If whe have a track with 4 trackpoints:
        track = [tp1, tp2, tp3, tp4]
        It is the distance between tp1 and tp2, plus the distance between
        t2 and t3, plus the distance between t3 and t4.
        """
        # The length of all the parents.
        parent_length = 0
        if include_parents and self.parent is not None \
           and len(self.parent.trackpoints) > 0:
            parent_length += self.parent.length()

            # The length between the last parent tracpoint and the this
            # first trackpoint.
            last_parent_tp = self.parent.trackpoints[-1]
            parent_length += last_parent_tp.length_to(self.trackpoints[0])

        return parent_length + sum(map(
            lambda (tp0, tp1): tp0.length_to(tp1),
            zip(self.trackpoints, self.trackpoints[1:])
        ))

    def number_of_trackpoints(self, include_parents=False):
        """
        Returns the number of trackpoints for the track.
        """
        number_of_trackpoints = 0
        if include_parents and self.parent is not None:
            number_of_trackpoints = self.parent.number_of_trackpoints(
                include_parents=True
                )
        return number_of_trackpoints + len(self.trackpoints)

    def save(self, filename, include_parents=False):
        """
        Saves the trackpoints to a file, including the parent track.
        """
        if include_parents and self.parent is not None:
            self.parent.save(filename, include_parents=True)

        with open(filename, 'a') as fp:
            for trackpoint in self.trackpoints:
                fp.write("{tp}\n".format(tp=trackpoint))
