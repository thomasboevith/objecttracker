#!/usr/bin/env python
# coding: utf-8
import os
import sys

__doc__ = """Usage:
    {filename} [options] <video_filename> [--verbose|--debug] [--video-speed=<vs>]

Options:
    -h, --help                        This help message.
    -d, --debug                       Output a lot of info..
    -v, --verbose                     Output less less info.
    --video-speed=<vs>                Speed of the video [default: 1].
    --log-filename=logfilename        Name of the log file.
""".format(filename=os.path.basename(__file__))

import logging
# Define the logger
LOG = logging.getLogger(__name__)

import cv2
import numpy as np
from objecttracker import noise
from objecttracker import connected_components
from objecttracker import color

import docopt
args = docopt.docopt(__doc__, version="1.0")

if args["--debug"]:
    logging.basicConfig(filename=args["--log-filename"], level=logging.DEBUG)
elif args["--verbose"]:
    logging.basicConfig(filename=args["--log-filename"], level=logging.INFO)
else:
    logging.basicConfig(filename=args["--log-filename"], level=logging.WARNING)
LOG.info(args)


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


def view_video(video_filename, video_speed=1):
    cap = cv2.VideoCapture(video_filename)

    fgbg = cv2.BackgroundSubtractorMOG()

    frames = [None] * 5
    labelled_frames = [None] * 5

    tracks = []

    while(cap.isOpened()):
        ret, frame = cap.read()

        frames = frames[1:]
        frames.append(frame)

        # Blur image frame by 7,7.
        blurred_frame = cv2.blur(frame, (7, 7))

        # Extract background.
        fgmask = fgbg.apply(blurred_frame)

        # Remove noise from the frame.
        fgmask = noise.remove_noise(fgmask)

        kernel = np.ones((7, 7), np.uint8)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=5)

        # Get a frame with labelled connected components.
        labelled_fgmask = connected_components.find_labelled_frame(fgmask.copy())
        labelled_frames = labelled_frames[1:]
        labelled_frames.append(labelled_fgmask)

        for cnt in connected_components.find_contours(fgmask):
            LOG.debug(cv2.contourArea(cnt))
            if cv2.contourArea(cnt) > 500:
                cx, cy = get_centroid(cnt)
                cv2.circle(frame, (int(cx), int(cy)), 5, (0, 255, 255), 3)

        bgr_fgmask = labelled2bgr(labelled_fgmask)

        # Visualize tracks
        img = frame
#        for track in enumerate(tracks):
#            lines = []
#            for trackpoint in track.trackpoints:
#                p = [trackpoint.row, trackpoint.col]
#                lines.append(p)

#            lines = np.array(lines)
#            cv2.polylines(img, [lines], 0, (0,0,255))
#            for trackpoint in track.trackpoints:
#                cv2.circle(img, (trackpoint.row, trackpoint.col), 0, (0,255,255), -1)

        # View the frame.
        cv2.imshow('fgmask', fgmask)
        cv2.imshow('label', labelled_fgmask)
        cv2.imshow('bgr_fgmask', bgr_fgmask)
        cv2.imshow('frame', frame)
        if cv2.waitKey(video_speed) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    view_video(args['<video_filename>'], int(args["--video-speed"]))
