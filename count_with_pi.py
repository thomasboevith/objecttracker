#!/usr/bin/env python
# coding: utf-8
import os
__doc__ = """Usage:
    {filename} [options] [--verbose|--debug] [[-s][-r]|--record-frames-only]

Options:
    -h, --help                        This help message.
    -d, --debug                       Output a lot of info..
    -v, --verbose                     Output less less info.
    --log-filename=logfilename        Name of the log file.
    -s, --save-tracks                 Save the tracks.
    -r, --record-frames               Save all the frames.
    --record-frames-only          Only save the frames. Do count or save tracks.
""".format(filename=os.path.basename(__file__))

import time
from collections import deque
import cv2
import numpy as np
import objecttracker
from picamera.array import PiRGBArray
from picamera import PiCamera
import gc

import logging
# Define the logger
LOG = logging.getLogger(__name__)

import docopt
args = docopt.docopt(__doc__, version="1.0")

if args['--record-frames-only']:
    args['--record-frames'] = True
    args['--save-tracks'] = False
    

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


def start_counting(save_tracks, save_frames):
    LOG.info("Starting.")
    fgbg = cv2.BackgroundSubtractorMOG()

    camera = PiCamera()
    camera.resolution = (640/2, 480/2)
    camera.framerate = 30
    camera.iso = 800
    camera.zoom = (0.1, 0.2, 0.9, 0.9)

    # allow the camera to warmup
    LOG.info("Warming up.")
    time.sleep(2)

    #LOG.info("Setting shutter speed.")
    camera.shutter_speed = camera.exposure_speed
    camera.exposure_mode = 'off'
    camera.shutter_speed = 40000
    LOG.info(camera.shutter_speed)
 
    # LOG.info("Setting white ballance.")
    # camera.awb_mode = "cloudy"
    # g = camera.awb_gains
    # camera.awb_mode = 'off'
    # camera.awb_gains = g
    # LOG.info(g)

    LOG.info("Ready.")

    track_match_radius = max(camera.resolution)/6
    min_linear_length = 0.5*max(camera.resolution)
    tracks = []
    i = 0
    rawCapture = PiRGBArray(camera, size=camera.resolution)

    LOG.info("Counting...")
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        frame = frame.array

        if save_frames:
            i += 1
            directory = "/tmp/frames/"
            if not os.path.isdir(directory):
                os.makedirs(directory)
            filename = os.path.join(directory, "%05i.png"%(i))
            cv2.imwrite(filename, frame)

        # Extract background.
        blurred_frame = cv2.blur(frame, (max(camera.resolution)/100,)*2)

        fgmask = fgbg.apply(blurred_frame, learningRate=0.001)

        fgmask = objecttracker.erode_and_dilate(fgmask)

        trackpoints = objecttracker.get_trackpoints(fgmask, frame)
        tracks, tracks_to_save = objecttracker.get_tracks(trackpoints, tracks, track_match_radius)

        if save_tracks:
            for t in tracks_to_save:
                t.save(min_linear_length, track_match_radius, "/tmp/tracks")
                t = None

        tracks_to_save = None
        gc.collect()
        rawCapture.truncate(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print "FIN"
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_counting(args['--save-tracks'], args['--record-frames'], )
