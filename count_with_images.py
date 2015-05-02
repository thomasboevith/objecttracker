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

def get_frames(frames, path):
    """
    Inserts frames into the frames queue, which must be a 
    multiprocessing.Queue.
    """
    for root, dirs, files in os.walk(path):
        LOG.debug("Dir, '%s': %i."%(root, len(files)))
        files.sort()
        for filename in files:
            if filename.endswith(".png"):
                frames.put(cv2.imread(os.path.join(root, filename)))


if __name__ == "__main__":
    import docopt
    args = docopt.docopt(__doc__, version="1.0")

    if args["--debug"]:
        logging.basicConfig(filename=args["--log-filename"], level=logging.DEBUG)
    elif args["--verbose"]:
        logging.basicConfig(filename=args["--log-filename"], level=logging.INFO)
    else:
        logging.basicConfig(filename=args["--log-filename"], level=logging.WARNING)
    LOG.info(args)

    frames = multiprocessing.Queue()

    frame_reader = multiprocessing.Process(target=get_frames, args=(frames, args['<image_directory>']))
    frame_reader.daemon = True
    frame_reader.start()
    
    counter_process = multiprocessing.Process(target=objecttracker.counter, args=(frames, ))
    counter_process.daemon = True
    counter_process.start()

    # This will wait forever as there will allways be more frames.
    frame_reader.join()

    print "FIN"
