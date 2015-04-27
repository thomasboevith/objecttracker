#!/usr/bin/env python
# coding: utf-8
import os
__doc__ = """Usage:
    {filename} <variable> [options] [--verbose|--debug]

Options:
    -h, --help                        This help message.
    -d, --debug                       Output a lot of info..
    -v, --verbose                     Output less less info.
    --log-filename=logfilename        Name of the log file.
    -k=number_of_classes              Number of classes. The k in k means. [default: 2].
""".format(filename=os.path.basename(__file__))
import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
import objecttracker.database
import docopt
import logging

# Define the logger
LOG = logging.getLogger(__name__)
args = docopt.docopt(__doc__, version="1.0")

if args["--debug"]:
    logging.basicConfig(filename=args["--log-filename"], level=logging.DEBUG)
elif args["--verbose"]:
    logging.basicConfig(filename=args["--log-filename"], level=logging.INFO)
else:
    logging.basicConfig(filename=args["--log-filename"], level=logging.WARNING)
LOG.debug(args)

values = []
with objecttracker.database.Db() as db:
    values = np.array([float(row[0]) for row in db.get_rows("SELECT %s FROM tracks;"%(args["<variable>"]))])

print kmeans(values, int(args["-k"]))
