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

def get_trackpoints(fgmask, raw_frame, timestamp):
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
            trackpoints.append(trackpoint.Trackpoint(timestamp, cx, cy, raw_frame, size=contour_area))
    return trackpoints

def get_foreground(foreground_background_subtractor,
                   raw_frame,
                   learning_rate=0.001):
    # Extract background.
    resolution = raw_frame.shape[0:2]

    # Blur the frame a little.
    blurred_frame = cv2.blur(raw_frame, (int(max(resolution)/50.0), )*2)

    # Subtract the foreground from the background.
    fgmask = foreground_background_subtractor.apply(blurred_frame,
                                                    learningRate=learning_rate)

    # Remove the smallest noise and holes in objects.
    return fgmask

def get_tracks_to_save(fgmask, raw_frame, timestamp, tracks, track_match_radius):
    # Get all trackpoints from the fgmask. 
    trackpoints = get_trackpoints(fgmask, raw_frame, timestamp)

    # Match the tracks with the trackpoints.
    tracks, tracks_to_save = separate_tracks(trackpoints, tracks, track_match_radius)
    return tracks, tracks_to_save

def separate_tracks(trackpoints, tracks, track_match_radius):
    """
    Dividing tracks into the tracks to be saved and the other
    tracks (the ones to keep adding to).
    """
    LOG.debug("Separating tracks.")

    tracks = match_tracks(tracks, trackpoints, track_match_radius)
    for t in tracks: t.incr_age() 
    tracks = prune_tracks(tracks)  # Removes old, very small tracks.

    tracks_to_save = []
    new_tracks = []
    for track in tracks:
        if track.age > 10:
            tracks_to_save.append(t)
        else:
            new_tracks.append(t)
    # tracks, tracks_to_save = connect_tracks(tracks, tracks_to_save, track_match_radius)
    return new_tracks, tracks_to_save

def connect_tracks(tracks, tracks_to_save, track_match_radius):
    """
    If e.g. a car is hiding a bike, the bike disappears from the view
    and the track ends. The next time the bike appears, it starts a new track
    again. This function tries to connect the new and the old tracks.

    TODO: It can clearly be expanded doing something more intelligent.
    """
    LOG.debug("Connecting tracks")

    # As the list need to be manipulated in the loops,
    # new ones are created, to avoid assignment problems.
    new_tracks_to_save = []

    while len(tracks_to_save):
        # The current track to save is popped from the list.
        track_to_save = tracks_to_save.pop()

        # TODO: If the track ends in the center box, somehow, somewhere.

        # Asuming that the track will still be saved unless
        # a new track has been found.
        new_tracks_to_save.append(track_to_save)

        # Get the next expected point for the current track.
        next_trackpoint = track_to_save.expected_next_point()
        if next_trackpoint is None:
            # This should never happen because it means that 
            # there is only one trackpoint in a track that is
            # supposed to be saved...
            next_trackpoint = track_to_save.trackpoints[-1]
            # else:
            # Use the kalman filter on it to reduce noise errors.
            #    next_trackpoint = track_to_save.kalman(next_trackpoint)

        # Sort the tracks by the closest track.
        # If there are two tracks in the tracks list
        # the one with the first trackpoint closest to the
        # last trackpoint from the track to save comes
        # first.
        next_trackpoint.sort_tracks_by_closest(tracks)

        # Now the closest track appears first in this loop.
        for t in tracks:
            first_tp = t.trackpoints[0]
            # Is the current track close enough?
            if next_trackpoint.length_to(first_tp) > track_match_radius*2:
                # Guess not... Then the next is not close enough either.
                # No track is close enough.
                break

            # Is the direction the same, or almost the same?
            #A = t.direction(deg=True)  # Current track.
            #B = track_to_save.direction(deg=True)  # The popped track.
            #if A is not None and B is not None and np.abs(track.diff_degrees(A, B)) > 30:
                # The direction is too differnt. Go on to
                # the next track in the list.
            #    continue

            # All conditions were met so far. We have a match.
            # Connect the current track to the new track.
            new_tracks_to_save.remove(track_to_save)  # The asumption was not correct.
            tracks.remove(t)  # This will be replaced by the new connected track below.
            track_to_save.connect(t)  # Connect the two tracks.
            tracks.append(track_to_save)  # The track will not be saved yet.
            
            # The match was found. We assume that this was the best match.
            # TODO: The next one may actually be better. Or the next one thereafter.
            break
    return tracks, new_tracks_to_save

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
    if len(trackpoints) > 0:
        frame = trackpoints[0].frame
        draw_tracks(tracks, frame)

    while len(trackpoints) > 0:
        tp = trackpoints.pop()

        tp.draw(frame)
        tp.draw(frame, radius=track_match_radius, color=(255, 255, 0))

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

    if len(trackpoints) > 0:
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            sys.exit()
        time.sleep(1/6.0)

    return tracks

def prune_tracks(tracks):
    """
    Removes tracks that are obviously not useful.
    """
    tracks_to_keep = []
    for t in tracks:
        if t.age < 6:
            # Keep all newly updated tracks.
            tracks_to_keep.append(t)
            continue
        if t.total_length() > 10:
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


def erode(frame):
    """
    To remove noise.

    Erode (makes the object bigger) to "swallow holes".
    """
    # Setting the kernel.
    LOG.debug("Eroding (making it bigger).")
    erode_kernel_size = max(frame.shape[:2])/100
    LOG.debug("Erode kernel size: '%s'."%(erode_kernel_size))
    ERODE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_kernel_size,)*2)

    # Do the erotion.
    eroded_frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, ERODE_KERNEL)
    return eroded_frame

def dilate(frame):
    """
    Then dilate (reduces the object) again.
    """
    # Setting the dilation kernel.
    LOG.debug("Dilating (making it smaller again).")
    dilate_kernel_size = min(frame.shape[:2])/50
    DILATE_KERNEL = np.ones((dilate_kernel_size,)*2, np.uint8)

    # Do the dilation.
    dilated_frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, DILATE_KERNEL, iterations=3)
    return dilated_frame

def foreground_extractor(raw_frames, foreground_frames):
    """
    Extracts the foreground (fgmask) from the raw frame and
    puts the foreground into the buffer.
    """
    fgbg = cv2.BackgroundSubtractorMOG()
    while True:
        LOG.debug("Foreground extractor: Waiting for a raw frame.")
        raw_frame, timestamp = raw_frames.get(block=True)
        LOG.debug("Foreground extractor: Got a frame. Number in queue: %i."%raw_frames.qsize())

        # Get the foreground.
        fgmask = get_foreground(fgbg, raw_frame)

        # Insert the frame and the timestamp into the buffer.
        foreground_frames.put([fgmask, raw_frame, timestamp])

def eroder(input_frames, output_frames):
    while True:
        LOG.debug("Eroder: Waiting for a frame.")
        fgmask, raw_frame, timestamp = input_frames.get(block=True)
        LOG.debug("Eroder: Got a input frame. Number in queue: %i."%input_frames.qsize())
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

        LOG.debug("Dilater: Got a frame. Number in queue: %i."%input_frames.qsize())
        dilated_fgmask = dilate(fgmask)
        output_frames.put([dilated_fgmask, raw_frame, timestamp])

def counter(input_frames, output_tracks, track_match_radius):
    tracks = []
    while True:
        LOG.debug("Track counter: Waiting for a frame.")
        fgmask, raw_frame, timestamp = input_frames.get(block=True)

        LOG.debug("Track counter: Got a fgmask. Number in queue: %i." % input_frames.qsize())
        tracks, tracks_to_save = get_tracks_to_save(fgmask, raw_frame, timestamp, tracks, track_match_radius)

        for t in tracks_to_save:
            # Putting tracks to save in the save queue.
            output_tracks.put(t)

def track_saver(input_queue, min_linear_length, track_match_radius, trackpoints_save_directory):
    while True:
        LOG.info("Tracksaver: Waiting for a track to save.")
        t = input_queue.get(block=True)
        LOG.debug("Tracksaver: Got a track to save. Number of tracks to save in queue: %i."%input_queue.qsize())
        t.save(min_linear_length, track_match_radius, trackpoints_save_directory)
        
