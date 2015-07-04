#!/usr/bin/env python
# coding: utf-8
import os
__doc__ = """
Create a plot of the tracks.

Usage:
    {filename} [options] [--verbose|--debug] [--date=<date>|(--date-from=<date> --date-to=<date>)]

Options:
    -h, --help                      This help message.
    -d, --debug                     Output a lot of info..
    -v, --verbose                   Output less less info.
    --log-filename=logfilename      Name of the log file.
    --date=<date>                   Date. If not specified: Today. Format YYYY-MM-DD.
    --date-from=<date>              Date from. Format YYYY-MM-DD.
    --date-to=<date>                Date to. Not included. Format YYYY-MM-DD.
    --street=<streetname>           Name of the street. Appears in the title of
                                    the plot.
    --output-dir=<dir>              Output directory. Where the plots are saved.
                                    [default: /data/plots]
""".format(filename=os.path.basename(__file__))

import objecttracker
import logging
import time
import datetime
import objecttracker.database
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class PlotException(Exception):
    pass

# Define the logger
LOG = logging.getLogger(__name__)

def create_plot(date_from, date_to, output_directory):
    size_ranges = (0, 2000, 10000, 1000000)
    colors = ("c", "m", "y")
    labels = ["S", "M", "L"]
    hour_range = ("05", "21")
    x_min = int(hour_range[0])
    x_max = int(hour_range[1])
    ticks = ["%02i"%i for i in range(x_min, x_max+1)]
    plt.clf()
    plt.xlim(xmin=x_min, xmax=x_max)
    plt.ylim(ymin=0, ymax=300)
    plt.xticks(range(x_min, x_max+1), ticks, rotation=30)

    title = "%s"%(date_from.strftime("%Y-%m-%d"))
    if (date_from - date_to).days > 1:
        title += " - %s"%(date_to.strftime("%Y-%m-%d"))
    if args['--street'] is not None:
        title = "%s %s"%(args['--street'], title)
    plt.title(title)

    plt.xlabel("Time")
    plt.ylabel("Antal")

    sql = """SELECT
               strftime('%Y-%m-%dT%H', date),
               strftime('%H', date),
               COUNT()
             FROM
               tracks
             WHERE
               date BETWEEN ? AND ?
               AND avg_size BETWEEN ? AND ?
               AND strftime('%H', date) BETWEEN ? AND ?
             GROUP BY
               strftime('%Y-%m-%dT%H', date);
          """

    with objecttracker.database.Db() as db:
        LOG.debug("Getting data from db.")
        numbers = {}
        for i, size_range in enumerate(zip(size_ranges[0:], size_ranges[1:])):
            sql_values = (date_from, date_to, size_range[0], size_range[1],
                          hour_range[0], hour_range[1])
            numbers[i] = []
            total = 0
            for row in db.get_rows(sql, sql_values):
                date, hour, count = row
                numbers[i].extend([int(hour),] * count)
            labels[i] += " (%i)"%(len(numbers[i]))

    plt.hist(numbers.values(), bins=x_max-x_min, range=(x_min, x_max),
             color=colors, label=labels, align='mid')
    plt.legend()
    if output_directory is not None and os.path.isdir(output_directory):
        filename = os.path.join(args['--output-dir'],
                                '%s.png'%(date_from.strftime("%Y-%m-%d")))
        plt.savefig(filename, bbox_inches='tight')
        print "%s saved"%(filename)
    else:
        raise PlotException("Output directory, '%s', must exist!" % (output_directory))


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

    date_stop = None

    if args['--date'] is not None:
        date = datetime.datetime.strptime(args['--date'], "%Y-%m-%d")
    elif args['--date-from'] is not None:
        date = datetime.datetime.strptime(args['--date-from'], "%Y-%m-%d")
    else:
        date = datetime.datetime.now()
        if date.hour < 2:
            # Calcluate yesterday too, for two hours.
            date = date - datetime.timedelta(days=1)
        date = date.date()
        date_stop = date + datetime.timedelta(days=1)

    if date_stop is None:
        if args['--date-to'] is not None:
            date_stop = datetime.datetime.strptime(args['--date-to'], "%Y-%m-%d")
        else:
            date_stop = date + datetime.timedelta(days=1)

    if (date_stop - date).days < 1:
        raise ValueError("Date from '%s' must be at least one smaller than date to '%s'." % (date.isoformat(), date_stop.isoformat()))
    
    while date < date_stop:
        date_to = date + datetime.timedelta(days=1)
        create_plot(date, date_to, args["--output-dir"])
        date += datetime.timedelta(days=1)
    print "FIN"
