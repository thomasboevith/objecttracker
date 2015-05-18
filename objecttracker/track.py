# coding: utf-8
import datetime
import numpy as np
import cv2
import os
from trackpoint import Trackpoint
import database
import logging

# Define the logger
LOG = logging.getLogger(__name__)
TABLE_NAME = "tracks"


# How the tracs table should look.
def create_tracks_table():
    value_types = [
        "id            integer primary key",
        "date          text",
        "avg_size      real",
        "linear_length real",
        "total_length  real",
        "direction     real",
        "number_of_tp  integer",
        ]
    SQL = '''CREATE TABLE IF NOT EXISTS %s (%s)''' % \
        (TABLE_NAME, ", ".join(value_types))
    SQL = " ".join(SQL.split())
    with database.Db() as db:
        LOG.debug(SQL)
        db.execute(SQL)

# Create the table.
create_tracks_table()


def diff_degrees(A, B):
    """
    The difference between two angles in degrees.

    Example:
    A = 60 deg
    B = 70 deg
    Diff = -10 deg
    """
    diff = A - B
    if diff > 180:
        diff -= 360
    elif diff < -180:
        diff += 360
    return diff


def score_factor(x_min, x_max, x):
    x = min(max(x, x_min), x_max)
    return 1 - ((x - x_min) / (x_max - x_min))


class TrackException(Exception):
    pass


class Track:
    def __init__(self):
        self.parent = None
        self.trackpoints = []
        self.name = datetime.datetime.now().isoformat()
        self.age = 0

    @property
    def direction_deg(self):
        return self.direction(deg=True)

    def direction(self, deg=False):
        if self.number_of_trackpoints() < 2:
            return None
        return self.trackpoints[0].direction_to(self.trackpoints[-1], deg)

    def __str__(self):
        return "Length: '%i'. Age: '%i'. Avg size: '%f'. Avg. length \
between trackpoints: '%f'." % (self.total_length(),
                               self.age,
                               self.size_avg(),
                               self.length_avg())

    def size_avg(self):
        return np.average([tp.size for tp in self.trackpoints])

    def incr_age(self):
        """
        Iterates the track one "age". Reset the age counter every
        time a new trackpoint is added.
        """
        self.age += 1

    def append(self, trackpoint):
        self.add_trackpoint(trackpoint)

    def add_trackpoint(self, trackpoint):
        """
        Adding a trackpoint to the track.
        The age counter gets a reset.
        """
        assert(isinstance(trackpoint, Trackpoint))
        self.age = 1
        self.trackpoints.append(trackpoint)

    def connect_tracks(self, track):
        """
        Connects two trackpoints.

        TODO: Use the parent functionality.
        self is currently parent.
        """
        assert(isinstance(track, Track))
        for tp in track.trackpoints:
            self.append(tp)

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
        """
        Calculates the average length of the trackpoints in the track.
        """
        number_of_trackpoints = self.number_of_trackpoints(include_parents)
        if number_of_trackpoints < 2:
            return 0
        return self.total_length(include_parents) / number_of_trackpoints - 1

    def sum_size(self, include_parents=False):
        size = 0
        if self.parent is not None and include_parents:
            size += self.parent.sum_size(include_parents)
        size += sum([tp.size for tp in self.trackpoints])
        return size

    def avg_size(self, include_parents=False):
        return self.sum_size(include_parents) / \
            self.number_of_trackpoints(include_parents)

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
        expected_next_point = Trackpoint(
            last_tp.timestamp + (last_tp.timestamp - second_last_tp.timestamp),
            (2 * last_tp.x - second_last_tp.x),
            (2 * last_tp.y - second_last_tp.y),
            None,
            )
        return expected_next_point

    def kalman(self, trackpoint):
        """
        Very simple implementation of a kalman filter.
        """
        assert(isinstance(trackpoint, Trackpoint))

        expected_tp = self.expected_next_point()
        if expected_tp is not None:
            x = int((expected_tp.x + trackpoint.x) / 2.0)
            y = int((expected_tp.y + trackpoint.y) / 2.0)

            new_trackpoint = trackpoint.copy()
            new_trackpoint.x = x
            new_trackpoint.y = y
            return new_trackpoint
        return trackpoint

    def length_to(self, trackpoint):
        """
        Length (number of pixels) from the last trackpoint
        to the given trackpoint.
        """
        assert(isinstance(trackpoint, Trackpoint))
        return self.last_trackpoint.length_to(trackpoint)

    def match(self, trackpoint, track_match_radius, size_match_ratio=0.8):
        """
        Check if the trackpoint matches this track.
        """
        assert(isinstance(trackpoint, Trackpoint))
        if self.length_to(trackpoint) < track_match_radius:
            #min_size = trackpoint.size * (1-size_match_ratio)
            #max_size = trackpoint.size * (1+size_match_ratio)
            #if min_size < self.avg_size() < max_size:
            return True
        return False

    def match_score(self, trackpoint, track_match_radius):
        """
        Calculates a match score for a specific trackpoint.
        The higher the match score, the better match.
        """
        assert(isinstance(trackpoint, Trackpoint))
        score = 1.0  # Total match.

        LOG.debug("TP: %s" % trackpoint)
        distance = self.length_to(trackpoint)
        if distance > track_match_radius * 2.0:
            LOG.debug("Too far away %f, %f" % (distance,
                                               track_match_radius * 2.0))
            score = 0
            return score

        score *= score_factor(0, track_match_radius * 2.0, distance)
        LOG.debug("Score after dist %f" % score)

        expected_next_point = self.expected_next_point()
        if False and expected_next_point is not None:
            expected_direction_deg = self.trackpoints[-1].direction_to(
                expected_next_point,
                deg=True)
            direction_deg = self.trackpoints[-1].direction_to(trackpoint,
                                                              deg=True)
            delta_direction = diff_degrees(expected_direction_deg,
                                           direction_deg)
            score *= score_factor(0, 180, abs(delta_direction))
            LOG.debug("Score after exp. dir: %f" % score)

            distance_to_expected_next_point = trackpoint.length_to(
                expected_next_point)
            score *= score_factor(0,
                                  track_match_radius,
                                  distance_to_expected_next_point)
            LOG.debug("Score after exp. dist: %f" % score)
        else:
            score *= score
            LOG.debug("Score squared: %f" % score)

        return score

    def draw_lines(self, frame, color=(255, 0, 255), thickness=1):
        """
        Draw the track to the frame.
        """
        # Track lines.
        lines = np.array([[tp.x, tp.y] for tp in self.trackpoints])
        cv2.polylines(frame, np.int32([lines]), 0, color, thickness=thickness)

    def draw_points(self, frame, color=(0, 255, 255), thickness=1):
        # Points i each line.
        for tp in self.trackpoints:
            tp.draw(frame, color=color, thickness=thickness)

    def linear_length(self):
        """
        Gets the linear distance from the first point to the last in a track.
        """
        first_tp, last_tp = self.trackpoints[0], self.trackpoints[-1]
        return first_tp.length_to(last_tp)

    def classify(self):
        """
        Very (too) simple classification of the track.
        """
        if self.size_avg() < 10000:
            return "Bike"
        else:
            return "Car"

    @property
    def first_trackpoint(self):
        """
        Gets the starting point of the tracks.
        """
        if len(self.trackpoints) == 0:
            return None
        return self.trackpoints[0]

    @property
    def last_trackpoint(self):
        """
        Gets the starting point of the tracks.
        """
        if len(self.trackpoints) == 0:
            return None
        return self.trackpoints[-1]

    def save_to_disk(self, min_linear_length, track_match_radius,
             trackpoints_save_directory=None):
        """
        Saves the trackpoints to a file, including the parent track.
        """
        LOG.debug("Saving track.")
        if trackpoints_save_directory is None:
            raise TrackException("Trackpoint save directory is not set.")

        if not os.path.isdir(trackpoints_save_directory):
            raise TrackException("Trackpoint save directory '%s' does not \
exist." % trackpoints_save_directory)

        if self.linear_length() < min_linear_length:
            LOG.debug("Too short track.")
            name = "short"
        else:
            name = "OK"

        # Saving the tracks to disk.
        self.save_trackpoints_to_directory(trackpoints_save_directory,
                                           name, track_match_radius)

    def save_to_db(self):
        """
        Saves the track features to the db.
        """
        # Date set to middle time stamp.
        date_str = (self.trackpoints[0].timestamp +
                    (self.trackpoints[-1].timestamp -
                     self.trackpoints[0].timestamp) / 2).isoformat()
        key_values = {
            "date": date_str,
            "avg_size": "%.3f" % self.avg_size(),
            "linear_length": "%.3f" % self.linear_length(),
            "total_length": "%.3f" % self.total_length(),
            "direction": "%.3f" % self.direction_deg,
            "number_of_tp": "%i" % len(self.trackpoints),
            }

        keys = key_values.keys()
        sql = '''INSERT INTO %s (%s) VALUES (%s)''' % (
            TABLE_NAME,
            ", ".join(keys),
            ", ".join(["?"] * len(keys)))

        LOG.debug(sql)

        values = key_values.values()
        LOG.debug("Values: '%s'." % ("', '".join(key_values.values())))

        with database.Db() as db:
            LOG.debug("Saving track to db.")
            db.execute(sql, values)

        LOG.info("Track saved.")

    def save_trackpoints_to_directory(self,
                                      trackpoints_save_directory,
                                      status_name,
                                      tp_search_radius):
        track_dir = os.path.join(
            trackpoints_save_directory, "%s_%s_%s" % (
                status_name,
                self.name,
                datetime.datetime.now().isoformat()))
        os.makedirs(track_dir)
        for i, tp in enumerate(self.trackpoints):
            self.draw_lines(tp.frame, color=(0, 255, 255))
            self.draw_points(tp.frame, color=(0, 255, 255))
            tp.draw(tp.frame, color=(255, 0, 255), thickness=3)
            tp.draw(
                tp.frame,
                radius=tp_search_radius,
                color=(0, 255, 0),
                thickness=1)
            cv2.imwrite(os.path.join(track_dir, "%0.5i.png" % (i)), tp.frame)
        LOG.info("'%s' track saved to %s. Track: %s " % (status_name,
                                                         track_dir,
                                                         self))
