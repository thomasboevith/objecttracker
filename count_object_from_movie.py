#!/usr/bin/env python
# coding: utf-8
import os
import sys
import time

__doc__ = """Usage:
    {filename} [options] <video_filename> [--verbose|--debug] [--video-speed=<vs>]

Options:
    -h, --help                        This help message.
    -d, --debug                       Output a lot of info..
    -v, --verbose                     Output less less info.
    --video-speed=<vs>                Speed of the video [default: 1].
    --log-filename=logfilename        Name of the log file.
    --slow-down                       Slow down when a track is active.
""".format(filename=os.path.basename(__file__))

from collections import deque
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
draw_counter = {'bike': 100, 'car': 100}

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


def get_background(frame, fgbg):
    # Blur image frame by 7,7.
    blurred_frame = cv2.blur(frame, (7, 7))

    # Extract background.
    fgmask = fgbg.apply(blurred_frame)
    return fgmask


def view_video(video_filename, video_speed=1, slow_down=False):
    counter = {'b': 0, 'c': 0}

    cap = cv2.VideoCapture(video_filename)

    fgbg = cv2.BackgroundSubtractorMOG()

    # For book keeping and later to calculate a track backwards in time...
    frames = deque([None] * 5)
    labelled_frames = deque([None] * 5)
    tracks = []

    while(cap.isOpened()):
        ret, frame = cap.read()
        
        frames.append(frame)
        frames.popleft() # Returns and removes the oldest frame in the list.

        # Get the background from the frame.
        fgmask = get_background(frame, fgbg)

        # Remove noise from the frame.
        fgmask = noise.remove_noise(fgmask)

        # Get a frame with labelled connected components.
        labelled_fgmask = connected_components.create_labelled_frame(fgmask)
        labelled_frames.popleft()
        labelled_frames.append(labelled_fgmask)
        bgr_fgmask = labelled2bgr(labelled_fgmask)

        # Collect the trackpoints.
        trackpoints = []
        for cnt in connected_components.find_contours(fgmask):
            contour_area = cv2.contourArea(cnt)
            LOG.debug(contour_area)
            if contour_area > MIN_OBJECT_AREA:
                cx, cy = get_centroid(cnt)
                trackpoints.append(trackpoint.Trackpoint(cx, cy, size=contour_area))


        while len(trackpoints) > 0:
            tp = trackpoints.pop()

            # Sorts the list of tracks. The closest first.
            tp.sort_by_closest(tracks)
            
            for t in tracks:
                if t.match(tp, frame):
                    t.append(t.kalman(tp))
                    break
            
            t = track.Track()
            t.append(tp)
            tracks.append(t)

        for t in tracks:
            t.incr_age()
            t.draw(frame)

            if t.age > 3:
                if t.total_length() < 10:
                    tracks.remove(t)
                    continue

            if t.age > 10:
                first_tp, last_tp = t.trackpoints[0], t.trackpoints[-1]
                frame_width = frame.shape[0]
                length = first_tp.length_to(last_tp)
                if length > 0.5*frame_width:
                    if t.size_avg() < 10000:
                        draw_counter['bike'] = 0
                        counter['b'] += 1
                    else:
                        draw_counter['car'] = 0
                        counter['c'] += 1
                    print counter
                tracks.remove(t)

        if draw_counter['bike'] < 50:
            cv2.putText(frame, "BIKE", (10,100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), thickness=2)
            draw_counter['bike'] += 1

        if draw_counter['car'] < 50:
            cv2.putText(frame, "CAR", (10,100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), thickness=2)
            draw_counter['car'] += 1

        if len(tracks) > 0:
            LOG.debug(" ### ".join([str(t) for t in tracks]))

            if slow_down:
                LOG.debug("Sleeping for 0.2s")
                time.sleep(0.5)

        # View the frame.
        #cv2.imshow('fgmask', fgmask)
        #cv2.imshow('label', labelled_fgmask)
        #cv2.imshow('bgr_fgmask', bgr_fgmask)
        cv2.imshow('frame', frame)

        if cv2.waitKey(video_speed) & 0xFF == ord('q'):
            break

    print "FIN"
    print counter

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    view_video(args['<video_filename>'], int(args["--video-speed"]), args["--slow-down"])
