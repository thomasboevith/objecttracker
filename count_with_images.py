#!/usr/bin/env python
# coding: utf-8
import os
__doc__ = """Usage:
    {filename} [options] <image_directory> [--verbose|--debug]

Options:
    -h, --help                        This help message.
    -d, --debug                       Output a lot of info..
    -v, --verbose                     Output less less info.
    --log-filename=logfilename        Name of the log file.
""".format(filename=os.path.basename(__file__))

import cv2
import multiprocessing
import objecttracker
import logging

# Define the logger
LOG = logging.getLogger(__name__)


def get_frames(frames_queue, path):
    """
    Inserts dates and frames into the frames queue.
    The frames queue must be a multiprocessing.Queue.
    """
    for root, dirs, files in os.walk(path):
        LOG.debug("Dir, '%s': %i." % (root, len(files)))
        files.sort()
        for filename in files:
            if filename.endswith(".png"):
                frames_queue.put(
                    [datetime.datetime.now(),
                     cv2.imread(os.path.join(root, filename))
                     ]
                    )


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
    track_match_radius = min_linear_length / 10

    # track_match_radius = max(resolution) / 6.0
    trackpoints_save_directory = "/data/tracks"

    # The frames queue is a list queue with list items.
    # Each item is a list with a timestamp and an image:
    # E.g. [<timestamp>, <image>]
    frames_queue = multiprocessing.Queue()

    # The tracks to save queue is a queue of tracks to save.
    # Simple as that.
    tracks_to_save_queue = multiprocessing.Queue()

    # The frame reader puts the frames into the frames queue.
    frame_reader = multiprocessing.Process(
        target=get_frames,
        args=(frames_queue, args['<image_directory>']))
    frame_reader.daemon = True
    frame_reader.start()
    LOG.info("Frames reader started.")

    # The counter creates tracks from the frames.
    # When a full track is created, it is inserted into
    # the tracks_to_save_queue.
    counter_process = multiprocessing.Process(
        target=objecttracker.counter,
        args=(frames_queue, tracks_to_save_queue, track_match_radius)
        )
    counter_process.daemon = True
    counter_process.start()
    LOG.info("Counter process started.")

    # The track saver saves the tracks that needs to be saved.
    # It also puts the data into the database.
    track_saver = multiprocessing.Process(
        target=objecttracker.save,
        args=(
            tracks_to_save_queue,
            min_linear_length,
            track_match_radius,
            trackpoints_save_directory))
    track_saver.daemon = True
    track_saver.start()
    LOG.info("Track saver started.")

    # Wait for all processes to end, which should never happen.
    frame_reader.join()
    counter_process.join()
    track_saver.join()

    print "FIN"
