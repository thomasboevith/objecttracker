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
    --frames-path=<path>              Where to find the frames,
                                      [default: /tmp/frames].
    --tracks-save-path=<path>         Where to save the tracks,
                                      [default: /data/tracks].
""".format(filename=os.path.basename(__file__))

import cv2
import multiprocessing
import objecttracker
import logging
import time
import datetime

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

    # Where the tracks are saved.
    trackpoints_save_directory = args["--tracks-save-path"]

    # The frames queue is a list queue with list items.
    # Each item is a list with a timestamp and an image:
    # E.g. [<timestamp>, <image>]
    raw_frames = multiprocessing.Queue()
    foreground_frames = multiprocessing.Queue()
    eroded_frames = multiprocessing.Queue()
    dilated_frames = multiprocessing.Queue()
    tracks_to_save = multiprocessing.Queue()
    temp_queue = multiprocessing.Queue()

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

    # The main purpose of the foreground extractor is to
    # separate the foreground from the background.
    foreground_extractor = multiprocessing.Process(
        target=objecttracker.foreground_extractor,
        args=(raw_frames, foreground_frames)
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

    # The dilate process dilates the foreground frame. This is done
    # after the erode process so that the frame is eroded and dilated.
    dilater = multiprocessing.Process(
        target=objecttracker.dilater,
        args=(eroded_frames, dilated_frames)
        )
    dilater.daemon = True
    dilater.start()

    # The counter creates tracks from the frames.
    # When a full track is created, it is inserted into
    # the tracks_to_save_queue.
    counter_process = multiprocessing.Process(
        target=objecttracker.counter,
        args=(dilated_frames, temp_queue, track_match_radius), #tracks_to_save, track_match_radius),
        )
    counter_process.daemon = True
    counter_process.start()
    LOG.info("Counter process started.")

    # The track saver saves the tracks that needs to be saved.
    # It also puts the data into the database.
    track_saver = multiprocessing.Process(
        target=objecttracker.track_saver,
        args=(
            tracks_to_save,
            min_linear_length,
            track_match_radius,
            trackpoints_save_directory))
    track_saver.daemon = True
    # track_saver.start()
    LOG.info("Track saver started.")

    d = datetime.datetime.now()
    while True:
        if (datetime.datetime.now() - d).total_seconds() > 10:
            d = datetime.datetime.now()
            print """Raw frames: %i, foreground frames: %i, eroded frames: %i,
dilated frames: %i, frames to save: %i.""" % (
                raw_frames.qsize(),
                foreground_frames.qsize(),
                eroded_frames.qsize(),
                dilated_frames.qsize(),
                tracks_to_save.qsize())

        out = temp_queue.get(block=True)
        t = out
        # fgmask, raw_frame, timestamp = out
        # raw_frame, timestamp = out
        
        for tp in t.trackpoints:
            frame = tp.frame
            t.draw_lines(frame)
            t.draw_points(frame)
            tp.draw(frame)
            cv2.imshow("Raw Frame", frame)
            time.sleep(1/16.0)
            # cv2.imshow("fgmask", fgmask)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Wait for all processes to end, which should never happen.
    frame_reader.join()
    # counter_process.join()
    # track_saver.join()

    print "FIN"
