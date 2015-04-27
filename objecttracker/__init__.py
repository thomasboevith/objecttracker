# coding: utf-8
import sys
import time
from collections import deque
import cv2
import numpy as np
import noise
import connected_components
import color
import track
import trackpoint

import logging
# Define the logger
LOG = logging.getLogger(__name__)

MIN_OBJECT_AREA = 500

def labelled2bgr(labelled_fgmask):
    """
    Assigns a colour to each a labeled connected component in the
    labelled mask.

    A labelled mask is a grayscale mask where all the connected
    components have the same number (labelled):
    +---------------------+
    | 0 0 0 0 0 0 0 0 0 0 |
    | 0 0 1 1 0 0 0 0 0 0 |
    | 0 0 1 1 0 0 0 0 0 0 |
    | 0 0 0 1 0 0 0 0 0 0 |
    | 0 0 0 0 0 0 0 2 2 0 |
    | 0 0 0 0 0 0 2 2 2 0 |
    | 0 0 0 0 0 0 0 2 2 0 |
    | 0 0 0 0 0 0 0 0 0 0 |
    +---------------------+

    The mask is converted to a rgb mask.
    Every point with the number 1 in the example above get the same
    colour in the rgb mask.
    """
    # Create an empty bgr mask.
    bgr_mask = np.zeros((labelled_fgmask.shape[0], labelled_fgmask.shape[1], 3), dtype=np.uint8)

    # Determine the number of connected components / labels.
    # The labels are assigned with the next number (i+=1), starting with 1,
    # so the number of labels is the highest (max) label.
    number_of_connected_components = np.max(labelled_fgmask)

    # If there are no objects in the labelled mask. Just return the empty bgr mask.
    if number_of_connected_components > 0:
        # Get an array of colors.
        colors = color.get_colors(number_of_connected_components)

        # Assign each colour to a label.
        for object_id in np.arange(np.max(labelled_fgmask)):
            bgr_mask[np.where(labelled_fgmask == object_id + 1)] = colors[object_id]

    # Return the backgound mask with its new, beautiful colours.
    return bgr_mask


def get_centroid(contour):
    """
    Gets the coordinates of the center of a contour.
    """
    moments = cv2.moments(contour)
    if moments['m00'] != 0.0:
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
        centroid = (cx, cy)
        return centroid


def get_bounding_box(contour):
    """
    Gets a bounding box around a contour"
    """
    return cv2.boundingRect(contour)


def get_foreground_mask(frame, fgbg):
    """
    Subtracting the background from the background.
    """
    # Blur image frame by 7,7.
    kernal = (frame.shape[0]/65, )*2
    # blurred_frame = cv2.blur(frame, (7,7))

    LOG.info("Kernal: %s"%str(kernal))
    blurred_frame = cv2.blur(frame, kernal)

    # cv2.imshow('blurred_frame', blurred_frame)

    # Extract background.
    fgmask = fgbg.apply(blurred_frame)
    
    # cv2.imshow('foreground frame', fgmask)
    return fgmask

def get_trackpoints(frame, fgbg):
    """
    Gets the trackpoints from the foreground mask.
    """
    # Subtract the background from the foreground.
    fgmask = get_foreground_mask(frame, fgbg)

    # Remove noise from the frame.
    fgmask = noise.remove_noise(fgmask)
    # cv2.imshow('after noise frame', fgmask)

    # The area must have a certain size.
    min_object_area = min(frame.shape[0]/3, frame.shape[1]/3)
    LOG.debug("Min object area: %i"%(min_object_area))

    # Collect the trackpoints.
    trackpoints = []
    for cnt in connected_components.find_contours(fgmask):
        contour_area = cv2.contourArea(cnt)
        LOG.debug(contour_area)
        # Only use contours of a certain size.
        if contour_area > min_object_area:
            cx, cy = get_centroid(cnt)
            trackpoints.append(trackpoint.Trackpoint(cx, cy, size=contour_area))
    return trackpoints

def get_tracks(trackpoints, tracks, track_match_radius):
    tracks = match_tracks(tracks, trackpoints, track_match_radius)
    for t in tracks: t.incr_age() 
    tracks = prune_tracks(tracks)
    tracks_to_save = []
    for t in [track for track in tracks if track.age > 10]:
        tracks_to_save.append(t)
        tracks.remove(t)
    return tracks, tracks_to_save

def get_bgr_fgmask(fgmask):
    """
    Gets a coloured mask with bgr colours.
    """
    # Get a frame with labelled connected components.
    labelled_fgmask = connected_components.create_labelled_frame(fgmask)
    bgr_fgmask = labelled2bgr(labelled_fgmask)
    return bgr_fgmask

def match_tracks(tracks, trackpoints, track_match_radius, frame=None):
    """
    Matches trackpoints with the most suitable track.
    If not tracks to match, a new track is created.
    """
    while len(trackpoints) > 0:
        tp = trackpoints.pop()
        
        matched = False
        closest_tracks = tp.sort_by_closest(tracks)
        # Find all the matching tracks.
        for t in closest_tracks:
            if t.match(tp, track_match_radius, frame):
                t.append(t.kalman(tp))
                matched = True
                break

        if not matched:
            # Appended to a new track.
            t = track.Track()
            t.append(tp)
            tracks.append(t)
    return tracks

def prune_tracks(tracks):
    for t in tracks:
        if t.age > 4:
            if t.total_length() < 10:
                tracks.remove(t)
                continue
    return tracks

def draw_tracks(tracks, frame):
    """
    Draws the tracks to the frame.
    """
    for t in tracks:
        t.draw_lines(frame)
        t.draw_points(frame)
