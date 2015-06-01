#!/usr/bin/env python
# coding: utf-8
import os
__doc__ = """Usage:
    {filename} [options] [--verbose|--debug] [[-s][-r]|--record-frames-only]

Options:
    -h, --help                      This help message.
    -d, --debug                     Output a lot of info..
    -v, --verbose                   Output less less info.
    --log-filename=logfilename      Name of the log file.
    --record-frames-only            Only save the frames.
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
    --frame-rate=<framerate>        The camera frame rate. I.e. number of
                                    images / second. [default: 12].
    --record-frames-path=<path>     Where to save the frames.
                                    [default: /data/frames].
    --save-tracks                   Save the tracks to disk. Set the path
                                    by --tracks-save-path
    --tracks-save-path=<path>       Where to save the tracks,
                                    [default: /data/tracks].
    --automatic-white-ballance      Automatically set white ballance.
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


def get_frames(frames_queue, resolution, framerate,
               automatic_white_ballance=False):
    camera = PiCamera()
    camera.resolution = resolution
    camera.framerate = framerate  # 16 #30
    # camera.iso = 800
    camera.zoom = (0.1, 0.2, 0.9, 0.9)

    # Allow the camera to warmup
    sleeptime_s = 2
    LOG.info("Warming up camera. Sleeping for %i seconds." % (sleeptime_s))
    time.sleep(sleeptime_s)

    if not automatic_white_ballance:
        camera.shutter_speed = 2980  # = camera.exposure_speed
        # camera.shutter_speed  at midnight: 0
        camera.exposure_mode = 'off'
        LOG.info("Camera shutter speed: %i" % camera.shutter_speed)

        camera.awb_mode = 'off'
        LOG.info("Camera auto white ballance: %s" % (camera.awb_mode))

        camera.awb_gains = (1.4, 1.2)
        # Camera awb_gains at midnight: (77/64, 191/128)
        LOG.info("Auto white ballance gains: (%s, %s)" % (camera.awb_gains))

    LOG.info("Camera ready.")
    rawCapture = PiRGBArray(camera, size=camera.resolution)

    for frame in camera.capture_continuous(rawCapture,
                                           format="bgr",
                                           use_video_port=True):
        frames_queue.put([frame.array, datetime.datetime.now()])
        # TODO: Set camera attributes by time or camera darkness or something.
        # It should change very slowly.
        rawCapture.truncate(0)


def save_frames(frames_queue, save_path):
    while True:
        frame, stamp = frames_queue.get(block=True)
        directory = os.path.join(save_path, stamp.strftime("%Y%m%dT%H"))
        if not os.path.isdir(directory):
            os.makedirs(directory)
        cv2.imwrite(os.path.join(directory, "%s.png" % stamp.isoformat()),
                    frame)


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

    resolution = (640 / 2, 480 / 2)
    min_linear_length = max(resolution) / 2
    if args['--track-match-radius'] is not None:
        track_match_radius = int(args['--track-match-radius'])
    else:
        track_match_radius = 2 * min_linear_length / int(args['--frame-rate'])
    LOG.info("Track match radius: %i" % (track_match_radius))

    if not args["--save-tracks"]:
        LOG.info("Tracks will not be saved... \
Use --save-tracks to save tracks.")

    raw_frames = multiprocessing.Queue()
    foreground_frames = multiprocessing.Queue()
    eroded_frames = multiprocessing.Queue()
    dilated_frames = multiprocessing.Queue()
    tracks_to_save = multiprocessing.Queue()

    # The frame reader puts the frames into the frames queue.
    frame_reader = multiprocessing.Process(
        target=get_frames,
        args=(raw_frames, resolution, int(args['--frame-rate']),
              args['--automatic-white-ballance']))
    frame_reader.daemon = True
    frame_reader.start()
    LOG.info("Frames reader started.")

    if args['--record-frames-only']:
        frame_saver = multiprocessing.Process(target=save_frames,
            args=(raw_frames, args['--record-frames-path']))
        frame_saver.daemon = True
        frame_saver.start()
    else:
        # The main purpose of the foreground extractor is to
        # separate the foreground from the background.
        foreground_extractor = multiprocessing.Process(
            target=objecttracker.foreground_extractor,
            args=(raw_frames, foreground_frames, args["--save-tracks"])
            )
        foreground_extractor.daemon = True
        foreground_extractor.start()

        # The erode process takes a foreground frame and erodes the
        # white pixels.
        eroder = multiprocessing.Process(
            target=objecttracker.eroder,
            args=(foreground_frames, eroded_frames)
            )
        eroder.daemon = True
        eroder.start()
        LOG.info("Eroder started.")

        # The dilate process dilates the foreground frame. This is done
        # after the erode process so that the frame is eroded and dilated.
        dilater = multiprocessing.Process(
            target=objecttracker.dilater,
            args=(eroded_frames, dilated_frames)
            )
        dilater.daemon = True
        dilater.start()
        LOG.info("Dilater started.")

        # The tracker creates tracks from the frames.
        # When a full track is created, it is inserted into
        # the tracks_to_save_queue.
        tracker_process = multiprocessing.Process(
            target=objecttracker.tracker,
            # args=(dilated_frames, temp_queue, track_match_radius)
            args=(dilated_frames, tracks_to_save, track_match_radius)
            )
        tracker_process.daemon = True
        tracker_process.start()
        LOG.info("Tracker process started.")

        # The track saver saves the tracks that needs to be saved.
        # It also puts the data into the database.
        track_saver = multiprocessing.Process(
            target=objecttracker.track_saver,
            args=(
                tracks_to_save,
                min_linear_length,
                track_match_radius,
                args["--tracks-save-path"],
                args["--save-tracks"]))
        track_saver.daemon = True
        track_saver.start()
        LOG.info("Track saver started.")

        d = datetime.datetime.now()
        while True:
            if (datetime.datetime.now() - d).total_seconds() > 60 * 30:
                d = datetime.datetime.now()
                print """Raw frames: %i, foreground frames: %i, eroded \
frames: %i, dilated frames: %i, frames to save: %i.""" % (
                    raw_frames.qsize(),
                    foreground_frames.qsize(),
                    eroded_frames.qsize(),
                    dilated_frames.qsize(),
                    tracks_to_save.qsize())

    # Wait for all processes to end, which should never happen.
    frame_reader.join()
    print "FIN"
