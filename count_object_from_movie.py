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
TRACK_MATCH_RADIUS = 100
SIZE_MATCH_RATIO = 0.5

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

                # Add a small circle in the middle of the object
                # on the main frame (computer?)
                cv2.circle(frame, (int(cx), int(cy)), 5, (0, 255, 255), 3)
                cv2.circle(frame, (int(cx), int(cy)), TRACK_MATCH_RADIUS, (0, 255, 0), 1)

        while len(trackpoints) > 0:
            tp = trackpoints.pop()
            min_length = 10**10
            min_index = None

            cv2.circle(frame, (int(tp.x), int(tp.y)), 3, (0,0,255), 1)

            # Find closest existing track.
            for i, t in enumerate(tracks):
                # Length to last point in current track.
                length = t.length_to(tp)
                cv2.putText(frame, "L: %.2f S_tp: %.2f S_t: %.2f"%(length, tp.size, t.avg_size()), (int(tp.x), int(tp.y)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 100, 100), thickness=1)

                # Find smallest length and index.
                if length < min_length:
                    # import pdb; pdb.set_trace()
                    min_length = length
                    min_index = i

            # Check that the point matches the minimum requirements.
            matching_track_found = False

            # Were there any minimum at all?
            if min_index != None:
                # Was the minimum within the an acceptable range?
                if tracks[min_index].length_to(tp) < TRACK_MATCH_RADIUS:
                    if tp.size * (1-SIZE_MATCH_RATIO) < tracks[min_index].avg_size() < tp.size * (1+SIZE_MATCH_RATIO):
                        matching_track_found = True

                # TODO:
                # Was the direction OK?
                # Did the object slow down, and change direction?
                # Did size match?

            if matching_track_found:
                tp = tracks[min_index].kalman(tp)
            else:
                # Create a new track and append it.
                min_index = len(tracks)
                tracks.append(track.Track())

            # Append the track poin to the relevant track.
            tracks[min_index].append(tp)


        for t in tracks:
            t.incr_age()

            lines = np.array([[tp.x, tp.y] for tp in t.trackpoints])
            thickness_factor = 1.0/(t.age)
            cv2.polylines(frame, np.int32([lines]), 0, (255,0,255), thickness=int(3*thickness_factor))

            for tp in t.trackpoints:
                cv2.circle(frame, (int(tp.x), int(tp.y)), 5, (0,255,255), 1)
            
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
                        counter['b'] += 1
                        print "Bike"
                    else:
                        print "Car"
                        counter['c'] += 1
                tracks.remove(t)

        if len(tracks) > 0:
            LOG.debug(" ### ".join([str(t) for t in tracks]))

            if slow_down:
                LOG.debug("Sleeping for 0.2s")
                time.sleep(0.2)

        # View the frame.
        #cv2.imshow('fgmask', fgmask)
        #cv2.imshow('label', labelled_fgmask)
        #cv2.imshow('bgr_fgmask', bgr_fgmask)
        cv2.imshow('frame', frame)
        print counter

        if cv2.waitKey(video_speed) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print counter

if __name__ == "__main__":
    view_video(args['<video_filename>'], int(args["--video-speed"]), args["--slow-down"])
