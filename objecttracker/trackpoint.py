import numpy as np
import sys
import cv2
import logging

# Define the logger
LOG = logging.getLogger(__name__)


class Trackpoint:
    def __init__(self, timestamp, x, y, frame=None, size=None, color=None):
        """
        A trackpoint is the centroid of the object.
        Each trackpoint is assigned to a tracks.
        """
        self.timestamp = timestamp
        self.x = x
        self.y = y
        self.frame = frame
        self.size = size
        self.color = color

    def __str__(self):
        """
        String representation of the trackpoint.
        """
        return "({x}, {y}) {size} {color}".format(x=self.x,
                                                  y=self.y,
                                                  size=self.size,
                                                  color=self.color)

    def copy(self):
        """
        Creates a new trackpoint with the same values.
        """
        return Trackpoint(self.timestamp, self.x, self.y,
                          self.frame, self.size, self.color)

    def length_to(self, tp):
        """
        Calculate the lengt between two track points. Linear length.
        """
        assert(isinstance(tp, Trackpoint))
        return np.sqrt((self.x - tp.x) ** 2 + (self.y - tp.y) ** 2)

    def draw(self, frame, radius=5, color=(0, 255, 255), thickness=1):
        """
        Draw the trackpoint on the frame.
        """
        cv2.circle(frame, (int(self.x), int(self.y)), radius, color,
                   thickness=thickness)

    def direction_to(self, tp, deg=False):
        """
        Calculate the direction to the input trackpoint
        If deg is set to True, the output is in degrees.
        """
        assert(isinstance(tp, Trackpoint))
        # archtan(x/y)
        direction = np.arctan2((tp.x - self.x), (tp.y - self.y))
        if deg:
            direction = np.rad2deg(direction)
        return direction

    def sort_tracks_by_closest(self, tracks):
        """
        Sorts the tracks (input) by the closest.
        The closest track is in the first in the tracks list.
        """
        tracks.sort(key=lambda t: t.length_to(self))
        return tracks

    def get_best_match(self, tracks, track_match_radius):
        best_match_track = None
        best_match_score = 0

        for t in tracks:
            match_score = t.match_score(self, track_match_radius)
            if match_score > best_match_score:
                best_match_score = match_score
                best_match_track = t

        LOG.debug("Best match score: %f. Returning track: %s." % (
                best_match_score,
                best_match_track))
        if best_match_track is not None:
            LOG.debug("RETURNING track age: %s" % best_match_track.age)
        return best_match_track
