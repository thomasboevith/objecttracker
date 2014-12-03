#!/usr/bin/env python
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
import docopt
import cv2
import numpy as np
from objecttracker import noise
from objecttracker import connected_components

# Define the logger
LOG = logging.getLogger(__name__)

args = docopt.docopt(__doc__, version="1.0")
if args["--debug"]:
    logging.basicConfig( filename=args["--log-filename"], level=logging.DEBUG )
elif args["--verbose"]:
    logging.basicConfig( filename=args["--log-filename"], level=logging.INFO )
else:
    logging.basicConfig( filename=args["--log-filename"], level=logging.WARNING )
LOG.info(args)

def view_video(video_filename, video_speed=1):
    cap = cv2.VideoCapture(video_filename)

    fgbg = cv2.BackgroundSubtractorMOG()

    while(cap.isOpened()):
        ret, frame = cap.read()

        # Blur image frame by 7,7.
        blurred_frame = cv2.blur(frame, (7,7))

        # Extract background.
        fgmask = fgbg.apply(blurred_frame)

        # Remove noise from the frame.
        fgmask = noise.remove_noise(fgmask)

        # Get a frame with labelled connected components.
        fgmask = connected_components.get_labelled_frame(fgmask)
        
        # Visualize tracks
        img = frame
        for track in enumerate(tracks):
            lines = []
            for trackpoint in track.trackpoints:
                p = [trackpoint.row, trackpoint.col]
                lines.append(p)

            lines = np.array(lines)
            cv2.polylines(img, [lines], 0, (0,0,255))
            for trackpoint in track.trackpoints:
                cv2.circle(img, (trackpoint.row, trackpoint.col), 0, (0,255,255), -1)

        # View the frame.
        cv2.imshow('frame', img)
        if cv2.waitKey(video_speed) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    view_video(args['<video_filename>'], int(args["--video-speed"]))
