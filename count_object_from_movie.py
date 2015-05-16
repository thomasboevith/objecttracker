#!/usr/bin/env python
# coding: utf-8
import os
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

import time
from collections import deque
import cv2
import numpy as np
import objecttracker

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

def start_counting(video_filename, video_speed=1, slow_down=False):
    cap = cv2.VideoCapture(video_filename)
    fgbg = cv2.BackgroundSubtractorMOG()

    tracks = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640/2, 480/2))
        LOG.debug("Frame shape: %s"%(str(frame.shape)))
        
        trackpoints = objecttracker.get_trackpoints(frame, fgbg)
        track_match_radius = min(frame.shape[:2])/5
        LOG.debug("Track match radius: %s"%(track_match_radius))
        tracks, tracks_to_save = objecttracker.get_tracks(trackpoints, tracks, track_match_radius)

        # Draw.
        objecttracker.draw_tracks(tracks, frame)

        frame_width = frame.shape[0]
        min_linear_length = 0.5*frame_width
        for t in tracks_to_save: t.save(min_linear_length)

        LOG.debug("Draw the track to the frame.")
        for t in tracks_to_save:
            t.draw_lines(frame, (255, 0, 255), thickness=3)
            t.draw_points(frame, (0, 255, 255))
            # cv2.putText(frame, track_class, (10,100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), thickness=2)

        if len(tracks) > 0:
            LOG.debug(" ### ".join([str(t) for t in tracks]))
            if slow_down:
                LOG.debug("Sleeping for 0.2s")
                time.sleep(0.2)

        cv2.imshow('frame', frame)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    print "FIN"

    cap.release()
    cv2.destroyAllWindows()
        

if __name__ == "__main__":
    start_counting(args['<video_filename>'], int(args["--video-speed"]), args["--slow-down"])
