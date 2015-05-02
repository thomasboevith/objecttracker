# coding: utf-8
import sys
import cv2
import numpy as np
import connected_components
import color
import track
import trackpoint

import logging
# Define the logger
LOG = logging.getLogger(__name__)

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


def get_trackpoints(fgmask, frame):
    """
    Gets the trackpoints from the foreground mask.
    """
    # The area must have a certain size.
    min_object_area = min(fgmask.shape[0], fgmask.shape[1])/4
    LOG.debug("Min object area: %i"%(min_object_area))

    # Collect the trackpoints.
    trackpoints = []
    for cnt in connected_components.find_contours(fgmask):
        contour_area = cv2.contourArea(cnt)
        LOG.debug(contour_area)
        # Only use contours of a certain size.
        if contour_area > min_object_area:
            cx, cy = get_centroid(cnt)
            trackpoints.append(trackpoint.Trackpoint(cx, cy, frame, size=contour_area))
    return trackpoints

def connect_tracks(tracks, tracks_to_save, track_match_radius):
    LOG.debug("Connecting tracks")
    new_tracks_to_save = []
    while len(tracks_to_save):
        track_to_save = tracks_to_save.pop()
        new_tracks_to_save.append(track_to_save)

        last_tp = track_to_save.trackpoints[-1]
        last_tp.sort_tracks_by_closest(tracks)
        
        for t in tracks:
            first_tp = t.trackpoints[0]
            if last_tp.length_to(first_tp) > track_match_radius*2:
                break
            A = t.direction(deg=True)
            B = track_to_save.direction(deg=True)
            if A != None and B != None and np.abs(track.diff_degrees(A, B)) < 10:
                new_tracks_to_save.remove(track_to_save)
                track_to_save.connect(t)
                tracks.append(track_to_save)
                break
    return tracks, new_tracks_to_save
    

def get_tracks_to_save(fgbg, frame, tracks):
    # Extract background.
    resolution = frame.shape[0:2]
    track_match_radius = max(resolution)/6.0
    # Blur the frame a little.
    blurred_frame = cv2.blur(frame, (int(max(resolution)/100.0), )*2)

    # Subtract the foreground from the background.
    fgmask = fgbg.apply(blurred_frame, learningRate=0.001)

    # Remove the smallest noise and holes in objects.
    fgmask = erode_and_dilate(fgmask)

    # Get all trackpoints.
    trackpoints = get_trackpoints(fgmask, frame)

    # Match the tracks with the trackpoints.
    tracks, tracks_to_save = separate_tracks(trackpoints, tracks, track_match_radius)
    return tracks, tracks_to_save

def separate_tracks(trackpoints, tracks, track_match_radius):
    LOG.debug("Separating tracks.")

    tracks = match_tracks(tracks, trackpoints, track_match_radius)
    for t in tracks: t.incr_age() 
    tracks = prune_tracks(tracks)
    tracks_to_save = []
    new_tracks = []
    for track in tracks:
        if track.age > 5:
            tracks_to_save.append(t)
        else:
            new_tracks.append(t)
    tracks, tracks_to_save = connect_tracks(new_tracks, tracks_to_save, track_match_radius)
    return tracks, tracks_to_save

def get_bgr_fgmask(fgmask):
    """
    Gets a coloured mask with bgr colours.
    """
    # Get a frame with labelled connected components.
    labelled_fgmask = connected_components.create_labelled_frame(fgmask)
    bgr_fgmask = labelled2bgr(labelled_fgmask)
    return bgr_fgmask

def match_tracks(tracks, trackpoints, track_match_radius):
    """
    Matches trackpoints with the most suitable track.
    If not tracks to match, a new track is created.
    """
    while len(trackpoints) > 0:
        tp = trackpoints.pop()
        
        matched = False
        closest_tracks = tp.sort_tracks_by_closest(tracks)
        # Find all the matching tracks.
        for t in closest_tracks:
            if t.match(tp, track_match_radius):
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


def erode_and_dilate(fgmask):
    """
    To remove noise.

    Erode (makes the object bigger) to "swallow holes".
    then dilate (reduces the object) again.
    """
    LOG.debug("Removing noise.")

    LOG.debug("Eroding (making it smaller).")
    erode_kernel_size = max(fgmask.shape[:2])/150
    LOG.debug("Erode kernel size: '%s'."%(erode_kernel_size))
    ERODE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_kernel_size,)*2)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, ERODE_KERNEL)
    # cv2.imshow('eroded frame', fgmask)

    LOG.debug("Dilating (making it bigger again).")
    dilate_kernel_size = erode_kernel_size*3
    DILATE_KERNEL = np.ones((dilate_kernel_size,)*2, np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, DILATE_KERNEL, iterations=5)
    # cv2.imshow('dilated frame', fgmask)

    return fgmask


def counter(frames):
    fgbg = cv2.BackgroundSubtractorMOG()

    tracks = []
    while True:
        LOG.debug("Framesize: %i"%frames.qsize())

        frame = frames.get(block=True)
        LOG.debug("Got a frame.")

        tracks, tracks_to_save = get_tracks_to_save(fgbg, frame, tracks)

        for t in tracks_to_save:
            t.save(min_linear_length=max(frame.shape)*.5, track_match_radius=30, trackpoints_save_directory="/tmp/tracks")
        
