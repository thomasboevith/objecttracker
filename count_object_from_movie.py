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

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    fgbg = cv2.BackgroundSubtractorMOG()

    while(cap.isOpened()):
        ret, frame = cap.read()

        # Extract background.
        fgmask = fgbg.apply(frame)

        # Blur image frame by 7,7.
        fgmask = cv2.blur(fgmask, (7,7))
        
        # Erode, then dilate. To remove noise.
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the contours.
        contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # Find Create a labelled frame.
        label = 0
        for contour in contours:
            label += 1
            cv2.drawContours(fgmask, [contour], 0, label, -1)  # -1 fills the image.

        LOG.debug("%i connected components."%(np.max(fgmask)))

        # View the frame.
        cv2.imshow('frame', fgmask*255)
        if cv2.waitKey(video_speed) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    view_video(args['<video_filename>'], int(args["--video-speed"]))
