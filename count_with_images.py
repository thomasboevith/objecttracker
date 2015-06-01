#!/usr/bin/env python
# coding: utf-8
import os
__doc__ = """Usage:
    {filename} [options] <image_directory> [--verbose|--debug]

Options:
    -h, --help                      This help message.
    -d, --debug                     Output a lot of info..
    -v, --verbose                   Output less less info.
    --log-filename=logfilename      Name of the log file.
    --frames-path=<path>            Where to find the frames,
                                    [default: /tmp/frames].
    --save-tracks                   Save the tracks to disk. Set the path
                                    by --tracks-save-path
    --tracks-save-path=<path>       Where to save the tracks,
                                    [default: /data/tracks].
    --track-match-radius=<radius>   Track match radius. When a trackpoint is
                                    found, the matching track must be within
                                    this radius from the last point in a track,
                                    to match the track.
                                                          ___
                                                         /   \\
                                      o---o----o---o----(--o  ) The new
                                                         \___/ trackpoint
                                                              must be within
                                                            the track radius.

                                    If not set, default value is calculated
                                    from frame size and framerate.
                                    [default: 25]
""".format(filename=os.path.basename(__file__))

import cv2
import multiprocessing
import objecttracker
import logging
import time
import datetime
import numpy as np

# Define the logger
LOG = logging.getLogger(__name__)


def get_frames(path, raw_frames):
    """
    Inserts frames into the raw frames queue.
    Each frame gets a timestamp attached.

    The raw frames must be a multiprocessing.Queue
    (multiprocessing.queues.Queue).
    """
    assert(isinstance(raw_frames, multiprocessing.queues.Queue))

    # Find all the png files in path and its subdirectories.
    for root, dirs, files in os.walk(path):
        LOG.debug("Current dir, '%s': %i files." % (root, len(files)))

        # Sort the files. The files are saved by time stamp, so this
        # ensures that the frames come in ascending order.
        files.sort()

        # Find all the png files and put them into the raw frames
        # queue / buffer.
        for filename in files:
            # Filter out only files ending with png.
            # Example filename: 2015-05-03T12:55:15.462884.png
            if filename.endswith(".png"):
                # Extract the timestamp from the filename.
                # Example filename above.
                timestamp = datetime.datetime.strptime(
                    filename,
                    "%Y-%m-%dT%H:%M:%S.%f.png")

                # Read the file into a frame , which is a numpy.ndarray.
                raw_frame = cv2.imread(os.path.join(root, filename))

                # Put the frame and timestamp into the buffer / queue.
                raw_frames.put([raw_frame, timestamp])

                if raw_frames.qsize() > 100:
                    time.sleep(1)

def do_it(raw_frames, track_match_radius, min_linear_length):
    fgbg = cv2.BackgroundSubtractorMOG()
    tracks = []
    tracks_to_save = []
    ff = True

    i = 0
    while True:
        i += 1
        if i % 1000 == 0:
            ff = False

        LOG.debug("Getting frame")
        raw_frame, timestamp = raw_frames.get(block=True)
        LOG.debug("Foreground extractor: Got a frame. Number in queue: %i." %
                  raw_frames.qsize())

        # Get the foreground.
        fgmask = objecttracker.get_foreground(fgbg, raw_frame)
    
        # print np.unique(fgmask)
        # fgmask = np.array(np.where(fgmask == 255, 255, 0), dtype=np.uint8)

        # eroded_fgmask = objecttracker.erode(fgmask)
        kernel = np.ones((15,15), dtype=np.uint8)
        # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=1)

        total_tracks = len(tracks) + len(tracks_to_save)
        # print total_tracks
        # print ""
        # dilated_fgmask = objecttracker.dilate(fgmask)
        if total_tracks > 0:
            fg = fgmask.copy()
            for t in tracks:
                t.draw_points(fg, (100,))
            for t in tracks_to_save:
                t.draw_points(fg, (150,))

            cv2.imshow("FG", fg)
            cv2.imshow("Frame", raw_frame)
            if not ff:
                time.sleep(1/16.0)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print "ff"
                ff = True

        tracks, tracks_to_save = objecttracker.get_tracks_to_save(fgmask,
                                                                  raw_frame,
                                                                  timestamp,
                                                                  tracks,
                                                                  track_match_radius)



if __name__ == "__main__":
    import docopt
    args = docopt.docopt(__doc__, version="1.0")

    if args["--debug"]:
        logging.basicConfig(filename=args["--log-filename"],
                            level=logging.DEBUG)
    elif args["--verbose"]:
        logging.basicConfig(filename=args["--log-filename"],
                            level=logging.INFO)
    else:
        logging.basicConfig(filename=args["--log-filename"],
                            level=logging.WARNING)
    LOG.info(args)

    resolution = (320, 240)
    min_linear_length = max(resolution) / 2
    track_match_radius = int(args["--track-match-radius"]) #min_linear_length / 10
    print "Track match radius", track_match_radius

    # The frames queue is a list queue with list items.
    # Each item is a list with a timestamp and an image:
    # E.g. [<timestamp>, <image>]
    raw_frames = multiprocessing.Queue()

    # The only purpose of the framereader is to read the frames
    # from the camera / directory and put them into a buffer (frames queue).
    # If the buffer is filled, e.g. if it increases all the time
    # maybe the resolution is too large. It might also be that
    # some of the subsequent processes are too slow and maybe should
    # be adjusted.
    frame_reader = multiprocessing.Process(
        target=get_frames,
        args=(args['<image_directory>'], raw_frames)
        )
    frame_reader.daemon = True
    frame_reader.start()
    LOG.info("Frames reader started.")


    # It also puts the data into the database.
    do_iter = multiprocessing.Process(
        target=do_it,
        args=(raw_frames, track_match_radius, min_linear_length,)
        )
    do_iter.daemon = True
    do_iter.start()
    LOG.info("Doing it.")

    # Wait for all processes to end, which should never happen.
    frame_reader.join()

    print "FIN"
