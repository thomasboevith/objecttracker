# coding: utf-8
import sys
import cv2
import numpy as np
import connected_components
import color
import track
import trackpoint
import time

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
    bgr_mask = np.zeros(
        (labelled_fgmask.shape[0], labelled_fgmask.shape[1], 3),
        dtype=np.uint8)

    # Determine the number of connected components / labels.
    # The labels are assigned with the next number (i+=1), starting with 1,
    # so the number of labels is the highest (max) label.
    number_of_connected_components = np.max(labelled_fgmask)

    # If there are no objects in the labelled mask. Just return the empty
    # bgr mask.
    if number_of_connected_components > 0:
        # Get an array of colors.
        colors = color.get_colors(number_of_connected_components)

        # Assign each colour to a label.
        for object_id in np.arange(np.max(labelled_fgmask)):
            bgr_mask[np.where(labelled_fgmask == object_id + 1)] = \
                colors[object_id]

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


def get_trackpoints(fgmask, raw_frame, timestamp):
    """
    Gets all the trackpoints from the foreground mask,
    greater than a certain size.
    """
    # The area must have a certain size.
    min_object_area = min(fgmask.shape[0], fgmask.shape[1]) / 4
    LOG.debug("Min object area: %i" % (min_object_area))

    # Collect the trackpoints.
    trackpoints = []
    # Remember that find countours changes the mask.
    for cnt in connected_components.find_contours(fgmask.copy()):
        contour_area = cv2.contourArea(cnt)
        LOG.debug(contour_area)
        # Only use contours of a certain size.
        if contour_area > min_object_area:
            cx, cy = get_centroid(cnt)
            trackpoints.append(
                trackpoint.Trackpoint(timestamp, cx, cy,
                                      raw_frame, size=contour_area))
    return trackpoints


def get_foreground(foreground_background_subtractor,
                   raw_frame,
                   learning_rate=0.001):
    # Extract background.
    resolution = raw_frame.shape[0:2]

    # Blur the frame a little.
    blurred_frame = cv2.blur(raw_frame, (int(max(resolution) / 50.0), ) * 2)

    # Subtract the foreground from the background.
    fgmask = foreground_background_subtractor.apply(blurred_frame,
                                                    learningRate=learning_rate)

    # Remove the smallest noise and holes in objects.
    return fgmask


def get_tracks_to_save(fgmask, raw_frame, timestamp, tracks,
                       track_match_radius):
    # Get all trackpoints from the fgmask.
    trackpoints = get_trackpoints(fgmask, raw_frame, timestamp)

    # Matching trackpoints with tracks.
    tracks = match_trackpoints_with_tracks(trackpoints, tracks,
                                           track_match_radius)

    for t in tracks:
        t.incr_age()

    # Remove old tracks that are smaller than the diameter of the
    # match circle.
    tracks = prune_tracks(tracks, track_match_radius * 2)

    # Split the tracks into tracks to save
    tracks, tracks_to_save = split_tracks(tracks,
                                          track_match_radius)


    return tracks, tracks_to_save


def split_tracks(tracks, track_match_radius):
    """
    Dividing tracks into the tracks to be saved and the other
    tracks (the ones to keep adding to).
    """
    LOG.debug("Separating tracks.")

    # Split old and new tracks.
    tracks_to_save = []
    new_tracks = []
    for t in tracks:
        if t.age > 10:
            tracks_to_save.append(t)
        else:
            new_tracks.append(t)

    # Connect possible small tracks.
    tracks = new_tracks
    tracks, tracks_to_save = connect_tracks(tracks, tracks_to_save,
                                            track_match_radius)
    return tracks, tracks_to_save


def connect_tracks(tracks, tracks_to_save, track_match_radius):
    """
    If e.g. a car is hiding a bike, the bike disappears from the view
    and the track ends. The next time the bike appears, it starts a new track
    again. This function tries to connect the new and the old tracks.

    TODO: It can clearly be extended to do something more intelligent.
    """
    LOG.debug("Connecting tracks")
    new_tracks_to_save = []
    new_tracks = []
    for track_to_save in tracks_to_save:
        matched_track = None
        max_score = 0
        for match_track in tracks:
            score = track_to_save.match_score_track(
                match_track,
                track_match_radius * 3)
            if score > max_score:
                matched_track = match_track
                max_score = score

        if matched_track is not None and max_score > 0.2:
            tracks.remove(matched_track)
            tracks.append(track_to_save)
            track_to_save.connect_tracks(matched_track)
        else:
            new_tracks_to_save.append(track_to_save)

    return tracks, new_tracks_to_save


def get_bgr_fgmask(fgmask):
    """
    Gets a coloured mask with bgr colours.
    """
    # Get a frame with labelled connected components.
    labelled_fgmask = connected_components.create_labelled_frame(fgmask)
    bgr_fgmask = labelled2bgr(labelled_fgmask)
    return bgr_fgmask


def match_trackpoints_with_tracks(trackpoints, tracks, track_match_radius):
    """
    Matches trackpoints with the most suitable track.
    If not tracks to match, a new track is created.
    """
    for tp in trackpoints:
        best_matched_track = tp.get_best_match(tracks, track_match_radius)

        LOG.debug("Best matched track: %s" % best_matched_track)
        if best_matched_track is None:
            LOG.debug("Creating new track.")
            t = track.Track()
            LOG.debug("Adding trackpoint %s" % (tp))
            t.append(tp)
            LOG.debug("Adding track %s" % (t))
            tracks.append(t)
        else:
            LOG.debug("Best matched track age: %s" % best_matched_track.age)
            LOG.debug("Track was matched.")
            best_matched_track.append(best_matched_track.kalman(tp))
    return tracks


def prune_tracks(tracks, min_track_length):
    """
    Removes tracks that are obviously not useful.
    """
    tracks_to_keep = []
    while len(tracks) > 0:
        t = tracks.pop()
        if t.age < 6:
            # Keep all newly updated tracks, no
            # matter the length.
            tracks_to_keep.append(t)
            continue

        # Age is larger than above.
        # Checking for length.
        if t.total_length() > min_track_length:
            # Keep all old tracks if the length is
            # long enough. Remove very small tracks.
            tracks_to_keep.append(t)
            continue
    return tracks_to_keep


def draw_tracks(tracks, frame):
    """
    Draws the tracks to the frame.
    """
    for t in tracks:
        t.draw_lines(frame)
        t.draw_points(frame)

def close(frame):
    kernel = np.ones((15,15), dtype=np.uint8)
    return cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel, iterations=1)

def erode(frame):
    """
    To remove noise.

    Erode (makes black areas bigger) to "swallow white holes".
    That is, remove very small white areas.
    """
    # Setting the kernel.
    LOG.debug("Eroding (making the black bigger).")
    erode_kernel_size = max(frame.shape[:2]) / 50
    LOG.debug("Erode kernel size: '%s'." % (erode_kernel_size))
    ERODE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                             (erode_kernel_size,) * 2)

    # Do the erotion.
    eroded_frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, ERODE_KERNEL)
    return eroded_frame


def dilate(frame):
    """
    Reduces the black area to make the white areas bigger again.
    """
    # Setting the dilation kernel.
    LOG.debug("Dilating (making it smaller again).")
    dilate_kernel_size = min(frame.shape[:2]) / 50
    DILATE_KERNEL = np.ones((dilate_kernel_size,) * 2, np.uint8)

    # Do the dilation.
    dilated_frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, DILATE_KERNEL,
                                     iterations=3)
    return dilated_frame


def foreground_extractor(raw_frames, foreground_frames, save_raw_frame=False):
    """
    Extracts the foreground (fgmask) from the raw frame and
    puts the foreground into the buffer.
    """
    fgbg = cv2.BackgroundSubtractorMOG()
    while True:
        LOG.debug("Foreground extractor: Waiting for a raw frame.")
        raw_frame, timestamp = raw_frames.get(block=True)
        LOG.debug("Foreground extractor: Got a frame. Number in queue: %i." %
                  raw_frames.qsize())

        # Get the foreground.
        fgmask = get_foreground(fgbg, raw_frame)

        # Only save the raw frame if it is necassary.
        if not save_raw_frame:
            raw_frame = None

        # Insert the frame and the timestamp into the buffer.
        foreground_frames.put([fgmask, raw_frame, timestamp])


def closer(input_frames, output_frames):
    while True:
        LOG.debug("Closer: Waiting for a frame.")
        fgmask, raw_frame, timestamp = input_frames.get(block=True)
        LOG.debug("Closer: Got a input frame. Number in queue: %i." %
                  input_frames.qsize())
        closed_fgmask = close(fgmask)
        output_frames.put([closed_fgmask, raw_frame, timestamp])


def eroder(input_frames, output_frames):
    while True:
        LOG.debug("Eroder: Waiting for a frame.")
        fgmask, raw_frame, timestamp = input_frames.get(block=True)
        LOG.debug("Eroder: Got a input frame. Number in queue: %i." %
                  input_frames.qsize())
        eroded_fgmask = erode(fgmask)
        output_frames.put([eroded_fgmask, raw_frame, timestamp])


def dilater(input_frames, output_frames):
    """
    Dilates the frame from the input queue and inserts the
    new frame into the output queue.
    """
    while True:
        LOG.debug("Dilater: Waiting for a eroded frame.")
        fgmask, raw_frame, timestamp = input_frames.get(block=True)

        LOG.debug("Dilater: Got a frame. Number in queue: %i." %
                  input_frames.qsize())
        dilated_fgmask = dilate(fgmask)
        output_frames.put([dilated_fgmask, raw_frame, timestamp])


def tracker(input_frames, output_tracks, track_match_radius):
    tracks = []
    while True:
        LOG.debug("Tracker: Waiting for a frame.")
        fgmask, raw_frame, timestamp = input_frames.get(block=True)

        LOG.debug("Tracker: Got a fgmask. Number in queue: %i." %
                  input_frames.qsize())
        tracks, tracks_to_save = get_tracks_to_save(fgmask,
                                                    raw_frame,
                                                    timestamp,
                                                    tracks,
                                                    track_match_radius)

        for t in tracks_to_save:
            # Putting tracks to save in the save queue.
            output_tracks.put(t)


def track_saver(input_queue, min_linear_length, track_match_radius,
                trackpoints_save_directory, save_tracks_to_disk=False):
    """
    Process responsible for saving the track to the database and disk.
    """
    while True:
        LOG.debug("Tracksaver: Waiting for a track to save.")
        track_to_save = input_queue.get(block=True)
        LOG.debug("Tracksaver: Got a track to save. Number of tracks to \
save in queue: %i." % input_queue.qsize())
        track_to_save.save_to_db()
        LOG.info(track_to_save)
        if save_tracks_to_disk:
            track_to_save.save_to_disk(min_linear_length, track_match_radius,
                                       trackpoints_save_directory)
