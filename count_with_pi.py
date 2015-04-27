#!/usr/bin/env python
# coding: utf-8
import os
__doc__ = """Usage:
    {filename} [options] [--verbose|--debug]

Options:
    -h, --help                        This help message.
    -d, --debug                       Output a lot of info..
    -v, --verbose                     Output less less info.
    --log-filename=logfilename        Name of the log file.
""".format(filename=os.path.basename(__file__))

import time
from collections import deque
import cv2
import numpy as np
import objecttracker
from picamera.array import PiRGBArray
from picamera import PiCamera

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
LOG.debug(args)


"""
def record():
    i = 0
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        frame = frame.array
        i += 1
        filename = "capture/%05i.png"%i
        cv2.imwrite(filename, frame)
        if i > 10000:
            import sys
            sys.exit()
        # print filename
        rawCapture.truncate(0)
        continue 
"""     


def start_counting():
    fgbg = cv2.BackgroundSubtractorMOG()

    camera = PiCamera()
    camera.resolution = (640/2, 480/2)
    camera.framerate = 32

    rawCapture = PiRGBArray(camera, size=camera.resolution)
    # allow the camera to warmup
    time.sleep(2)

    track_match_radius = max(camera.resolution)/4

    tracks = []
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        frame = frame.array
        
        trackpoints = objecttracker.get_trackpoints(frame, fgbg)
        tracks, tracks_to_save = objecttracker.get_tracks(trackpoints, tracks, track_match_radius)

        frame_width = frame.shape[0]
        min_linear_length = 0.5*frame_width
        for t in tracks_to_save: t.save(min_linear_length)

        rawCapture.truncate(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print "FIN"
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_counting()
