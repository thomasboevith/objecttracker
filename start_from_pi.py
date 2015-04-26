#!/usr/bin/env python
# coding: utf-8
import os
import sys
import time

__doc__ = """Usage:
    {filename} [options] [--verbose|--debug] [--video-speed=<vs>] [--hide-video]

Options:
    -h, --help                        This help message.
    -d, --debug                       Output a lot of info..
    -v, --verbose                     Output less less info.
    --video-speed=<vs>                Speed of the video [default: 1].
    --log-filename=logfilename        Name of the log file.
    --slow-down                       Slow down when a track is active.
    --hide-video                      Does not show the video frames.
""".format(filename=os.path.basename(__file__))

from collections import deque
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
from objecttracker import noise
from objecttracker import connected_components
from objecttracker import color
from objecttracker import track
from objecttracker import trackpoint

import logging
# Define the logger
LOG = logging.getLogger(__name__)

import docopt
args = docopt.docopt(__doc__, version="1.0")

if args["--debug"]:
    logging.basicConfig(filename=args["--log-filename"], level=logging.DEBUG)
elif args["--verbose"]:
    logging.basicConfig(filename=args["--log-filename"], level=logging.INFO)
else:
    logging.basicConfig(filename=args["--log-filename"], level=logging.WARNING)
LOG.info(args)



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
    blurred_frame = cv2.blur(frame, (7, 7))

    # Extract background.
    fgmask = fgbg.apply(blurred_frame)
    return fgmask

def get_trackpoints(fgmask):
    """
    Gets the trackpoints from the foreground mask.
    """
    # Remove noise from the frame.
    fgmask = noise.remove_noise(fgmask)

    # Collect the trackpoints.
    trackpoints = []
    for cnt in connected_components.find_contours(fgmask):
        contour_area = cv2.contourArea(cnt)
        LOG.debug(contour_area)
        # Only use contours of a certain size.
        if contour_area > MIN_OBJECT_AREA:
            cx, cy = get_centroid(cnt)
            trackpoints.append(trackpoint.Trackpoint(cx, cy, size=contour_area))
    return trackpoints

def get_bgr_fgmask(fgmask):
    """
    Gets a coloured mask with bgr colours.
    """
    # Get a frame with labelled connected components.
    labelled_fgmask = connected_components.create_labelled_frame(fgmask)
    bgr_fgmask = labelled2bgr(labelled_fgmask)
    return bgr_fgmask

def match_tracks(tracks, trackpoints, frame=None):
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
            if t.match(tp, frame):
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

def get_tracks(frame, fgbg, tracks, hide_video=False):
        # Subtract the background from the foreground.
        fgmask = get_foreground_mask(frame, fgbg)
        trackpoints = get_trackpoints(fgmask)
        
        # TODO: CLEAN THIS UP!!
        if hide_video:
            tracks = match_tracks(tracks, trackpoints)
        else:
            tracks = match_tracks(tracks, trackpoints, frame)

        for t in tracks: t.incr_age() 
        tracks = prune_tracks(tracks)

        if not hide_video:
            draw_tracks(tracks, frame)

        frame_width = frame.shape[0]
        for t in [track for track in tracks if track.age > 10]:
            tracks.remove(t)
            if t.linear_length() > 0.5*frame_width:
                LOG.debug("Saving track.")
                t.save()

                if not hide_video:
                    # Simple classification of the track.
                    track_class = t.classify()

                    LOG.debug("Draw the track to the frame.")
                    t.draw_lines(frame, (255, 0, 255), thickness=3)
                    t.draw_points(frame, (0, 255, 255))
                    cv2.putText(frame, track_class, (10,100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), thickness=2)

        return frame, tracks


def start_counting(video_speed=1, slow_down=False, hide_video=False):
    camera = PiCamera()
    camera.resolution = (640/2, 480/2)
    camera.framerate = 32
    rawCapture = PiRGBArray(camera, size=camera.resolution)
    # allow the camera to warmup
    time.sleep(0.1)

    fgbg = cv2.BackgroundSubtractorMOG()

    # For book keeping and later to calculate a track backwards in time...
    # frames = deque([None] * 5)
    # labelled_frames = deque([None] * 5)
    tracks = []

    try:
        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            # grab the raw NumPy array representing the image, then initialize the timestamp
            # and occupied/unoccupied text
            frame = frame.array
        
            # frames.append(frame)
            # frames.popleft() # Pops the oldest frame off the list.
            frame, tracks = get_tracks(frame, fgbg, tracks, hide_video)

            if len(tracks) > 0:
                LOG.debug(" ### ".join([str(t) for t in tracks]))
                if slow_down:
                    LOG.debug("Sleeping for 0.2s")
                    time.sleep(0.2)

            if not hide_video:
                cv2.imshow('frame', frame)


            # clear the stream in preparation for the next frame
            rawCapture.truncate(0)

            if cv2.waitKey(video_speed) & 0xFF == ord('q'):
                    break

        print "FIN"

        cap.release()
        cv2.destroyAllWindows()
    except Exception, e:
        LOG.error(e)
        print e
        

if __name__ == "__main__":
    start_counting(int(args["--video-speed"]), args["--slow-down"], args['--hide-video'])
