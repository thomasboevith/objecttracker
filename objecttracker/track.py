import numpy as np
import cv2
from trackpoint import Trackpoint

TRACK_MATCH_RADIUS = 100
SIZE_MATCH_RATIO = 0.5

def diff_degrees(A, B):
    diff = A - B
    if diff > 180:
        diff -= 360
    elif diff < -180:
        diff += 360
    return abs(diff)


class Track:
    def __init__(self):
        self.parent = None
        self.trackpoints = []
        self.name = None
        self.age = 0

    def __str__(self):
        return "Length: '%i'. Age: '%i'. Avg size: '%f'. Avg. length between trackpoints: '%f'."%(self.total_length(), self.age, self.size_avg(), self.length_avg())

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

    def total_length(self, include_parents=False):
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
            parent_length += self.parent.total_length(include_parents)

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
        return self.total_length(include_parents) / number_of_trackpoints - 1 

    def sum_size(self, include_parents=False):
        size = 0
        if self.parent != None and include_parents:
            size += self.parent.sum_size(include_parents)
        size += sum([tp.size for tp in self.trackpoints])
        return size
        

    def avg_size(self, include_parents=False):
        return self.sum_size(include_parents) / self.number_of_trackpoints(include_parents)

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

    def length_to(self, trackpoint):
        return self.trackpoints[-1].length_to(trackpoint)

    def save(self, filename, include_parents=False):
        """
        Saves the trackpoints to a file, including the parent track.
        """
        if include_parents and self.parent is not None:
            self.parent.save(filename, include_parents=True)

        with open(filename, 'a') as fp:
            for trackpoint in self.trackpoints:
                fp.write("{tp}\n".format(tp=trackpoint))

    def direction(self, deg=False):
        if self.number_of_trackpoints() < 2:
            return None
        return self.trackpoints[-2].direction_to(self.trackpoints[-1], deg)

    def match(self, trackpoint, frame=None):
        if frame != None:
            # Add a small circle in the middle of the object
            # on the main frame (computer?)
            cv2.circle(frame, (int(trackpoint.x), int(trackpoint.y)), 0, (0, 255, 0), thickness=2)
            tp_kalman = self.kalman(trackpoint)
            cv2.circle(frame, (int(tp_kalman.x), int(tp_kalman.y)), 1, (255, 255, 255), thickness=2)

            for radius in range(10, TRACK_MATCH_RADIUS+1, 20):
                cv2.circle(frame, (int(trackpoint.x), int(trackpoint.y)), radius, (0, 255, 0), thickness=1)

        if self.length_to(trackpoint) < TRACK_MATCH_RADIUS:        
            if trackpoint.size * (1-SIZE_MATCH_RATIO) < self.avg_size() < trackpoint.size * (1+SIZE_MATCH_RATIO):
                direction = self.direction()
                if direction == None or diff_degrees(trackpoint.direction_to(self.kalman(trackpoint), deg=True), direction) < 30:
                    return True
                elif self.length_to(self.kalman(trackpoint)) < 10:
                    return True
                else:
                    # Did object slow down/have a really small speed before changing direction?
                    pass
        return False


    def draw(self, frame):
        # Track lines.
        lines = np.array([[tp.x, tp.y] for tp in self.trackpoints])
        thickness_factor = 1.0/(self.age)
        cv2.polylines(frame, np.int32([lines]), 0, (255,0,255), thickness=int(3*thickness_factor))

        # Points i each line.
        for tp in self.trackpoints:
            cv2.circle(frame, (int(tp.x), int(tp.y)), 5, (0,255,255), 1)


