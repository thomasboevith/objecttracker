import numpy as np
from trackpoint import Trackpoint


class Track:
    def __init__(self):
        self.parent = None
        self.trackpoints = []
        self.name = None
        self.age = 0

    def __str__(self):
        return "Length: '%i'. Age: '%i'. Avg size: '%f'. Avg. length between trackpoints: '%f'."%(self.length(), self.age, self.size_avg(), self.length_avg())

    def size_avg(self):
        return np.average([tp.size for tp in self.trackpoints])

    def incr_age(self):
        self.age += 1
        
    def append(self, trackpoint):
        self.add_trackpoint(trackpoint)

    def add_trackpoint(self, trackpoint):
        """
        Adding a trackpoint to the track.
        """
        assert(isinstance(trackpoint, Trackpoint))
        self.trackpoints.append(trackpoint)
        self.age = 0

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

    def length_avg(self, include_parents=False):
        number_of_trackpoints = self.number_of_trackpoints(include_parents)
        if number_of_trackpoints < 2:
            return 0
        return self.length(include_parents) / number_of_trackpoints - 1 

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


    def expected_next_point(self):
        # TODO: Handle inherited trackpoints.
        number_of_trackpoints = self.number_of_trackpoints()
        if number_of_trackpoints < 2:
            return None

        second_last_tp, last_tp = self.trackpoints[-2:]
        expected_next_point = Trackpoint((2*last_tp.x - second_last_tp.x),
                                         (2*last_tp.y - second_last_tp.y))
        return expected_next_point

        
    def kalman(self, trackpoint):
        assert(isinstance(trackpoint, Trackpoint))

        expected_tp = self.expected_next_point()
        if expected_tp != None:
            x = int((expected_tp.x + trackpoint.x)/2.0)
            y = int((expected_tp.y + trackpoint.y)/2.0)

            new_trackpoint = Trackpoint(None, None)
            new_trackpoint.__dict__ = trackpoint.__dict__
            new_trackpoint.x = x
            new_trackpoint.y = y
            return new_trackpoint
        return trackpoint

    def save(self, filename, include_parents=False):
        """
        Saves the trackpoints to a file, including the parent track.
        """
        if include_parents and self.parent is not None:
            self.parent.save(filename, include_parents=True)

        with open(filename, 'a') as fp:
            for trackpoint in self.trackpoints:
                fp.write("{tp}\n".format(tp=trackpoint))

