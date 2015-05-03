#!/usr/bin/env python
# coding: utf-8
import os
__doc__ = """Usage:
    {filename} [options] [--verbose|--debug] [[-s][-r]|--record-frames-only]

Options:
    -h, --help                  This help message.
    -d, --debug                 Output a lot of info..
    -v, --verbose               Output less less info.
    --log-filename=logfilename  Name of the log file.
    --record-frames-only        Only save the frames.
    --tracks-save-path=path     Where to save the tracks,
                                [default: /data/tracks].
""".format(filename=os.path.basename(__file__))

import time
import datetime
import objecttracker
import multiprocessing
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2

import logging
# Define the logger
LOG = logging.getLogger(__name__)

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
LOG.debug(args)


def get_frames(frames_queue, resolution):
    camera = PiCamera()
    camera.resolution = resolution
    camera.framerate = 16 #30
    # camera.iso = 800
    # camera.zoom = (0.1, 0.2, 0.9, 0.9)

    # Allow the camera to warmup
    sleeptime_s = 2
    LOG.info("Warming up camera. Sleeping for %i seconds." % (sleeptime_s))
    time.sleep(sleeptime_s)

    LOG.debug("Setting shutter speed.")
    camera.shutter_speed = 2980  # camera.exposure_speed
    camera.exposure_mode = 'off'
    LOG.info(camera.shutter_speed)

    LOG.debug("Setting white ballance.")
    # g = camera.awb_gains
    camera.awb_mode = 'off'
    # (Fraction(379, 256), Fraction(311, 256))
    camera.awb_gains = (1.4, 1.2)
    LOG.info(camera.awb_gains)

    LOG.info("Camera ready.")
    rawCapture = PiRGBArray(camera, size=camera.resolution)

    for frame in camera.capture_continuous(rawCapture,
                                           format="bgr",
                                           use_video_port=True):
        frames_queue.put([datetime.datetime.now(), frame.array])
        rawCapture.truncate(0)

def save_frames(frames_queue):
    while True:
        stamp, frame = frames_queue.get(block=True)
        directory = "/data/frames/%s" % stamp.strftime("%Y%m%dT%H")
        if not os.path.isdir(directory):
            os.makedirs(directory)
        cv2.imwrite(os.path.join(directory, "%s.png" % stamp.isoformat()), frame)


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

    # args['--record-frames-only']
    # args['--save-tracks']
    # args['--record-frames']
    resolution = (640/2, 480/2)
    min_linear_length = max(resolution) / 2
    track_match_radius = min_linear_length / 10

    # Where the tracks are saved.
    trackpoints_save_directory = args["--tracks-save-path"]

    # The frames queue is a list queue with list items.
    # Each item is a list with a timestamp and an image:
    # E.g. [<timestamp>, <image>]
    frames_queue = multiprocessing.Queue()

    # The queue to hold the foreground.
    erode_queue = multiprocessing.Queue()
    dilate_queue = multiprocessing.Queue()

    clean_queue = multiprocessing.Queue()

    # The tracks to save queue is a queue of tracks to save.
    # Simple as that.
    tracks_to_save_queue = multiprocessing.Queue()

    # The frame reader puts the frames into the frames queue.
    frame_reader = multiprocessing.Process(
        target=get_frames,
        args=(frames_queue, resolution))
    frame_reader.daemon = True
    frame_reader.start()
    LOG.info("Frames reader started.")

    if args['--record-frames-only']:
        frame_saver = multiprocessing.Process(
            target = save_frames,
            args=(frames_queue,)
            )
        frame_saver.daemon = True
        frame_saver.start()
    else:
        # Separates the foreground from the background
        foreground_extractor = multiprocessing.Process(
            target=objecttracker.foreground_extractor,
            args=(frames_queue, erode_queue)
            )
        foreground_extractor.daemon = True
        foreground_extractor.start()

        # 
        eroder = multiprocessing.Process(
            target=objecttracker.eroder,
            args=(erode_queue, dilate_queue)
            )
        eroder.daemon = True
        eroder.start()

        #
        dilater = multiprocessing.Process(
            target=objecttracker.dilater,
            args=(dilate_queue, clean_queue)
            )
        dilater.daemon = True
        dilater.start()

        # The counter creates tracks from the frames.
        # When a full track is created, it is inserted into
        # the tracks_to_save_queue.
        counter_process = multiprocessing.Process(
            target=objecttracker.counter,
            args=(clean_queue, tracks_to_save_queue, track_match_radius)
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

    max_datetime = datetime.datetime(2015, 05, 03, 14)
    while (datetime.datetime.now() < max_datetime):
        print "Framesqueue: %i" % frames_queue.qsize()
        time.sleep(10)
        


    # Wait for all processes to end, which should never happen.
    frame_reader.join()
    # counter_process.join()
    # track_saver.join()

    print "FIN"
