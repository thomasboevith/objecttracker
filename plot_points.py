#!/usr/bin/env python
# coding: utf-8
import os
__doc__ = """
Plot values from the database on a line.

Usage:
    {filename} <point_type> [options] [--verbose|--debug] [--date=<date>|(--date-from=<date> --date-to=<date>)]

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
""".format(filename=os.path.basename(__file__))

import logging
import time
import datetime
import objecttracker.database
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

class PlotException(Exception):
    pass

# Define the logger
LOG = logging.getLogger(__name__)

def create_plot(point_type, date_from, date_to, output_directory):
    # ticks = ["%02i"%i for i in range(x_min, x_max+1)]
    plt.clf()  # Clear figure.
    # plt.xlim(xmin=x_min, xmax=x_max)
    # plt.ylim(ymin=0, ymax=300)
    # plt.xticks(range(x_min, x_max+1), ticks, rotation=30)

    title = "%s"%(date_from.strftime("%Y-%m-%d"))
    if (date_from - date_to).days > 1:
        title += " - %s"%(date_to.strftime("%Y-%m-%d"))
    if args['--street'] is not None:
        title = "%s %s"%(args['--street'], title)
    plt.title(title)

    plt.xlabel(point_type)
    plt.ylabel(point_type)

    sql = """SELECT
               {point_type}
             FROM
               tracks
             WHERE
               date BETWEEN ? AND ?
               AND {point_type} < ?
          """.format(point_type=point_type)

    max_size = 20000
    with objecttracker.database.Db() as db:
        LOG.debug("Getting data from db.")
        sql_values = (date_from, date_to, max_size)
        values = [float(row[0]) for row in db.get_rows(sql, sql_values)]

    plt.hist(values, bins=100, range=(0, max_size))
    # plt.plot(values, values, 'ro', label=len(values))
    plt.legend()
    if output_directory is not None and os.path.isdir(output_directory):
        filename = os.path.join(output_directory,
                                '%s.png'%(date_from.strftime("%Y-%m-%d")))
        plt.savefig(filename, bbox_inches='tight')
        print "%s saved"%(filename)
    else:
        plt.show()
        # raise PlotException("Output directory, '%s', must exist!" % (output_directory))


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
            date = date.date() - datetime.timedelta(days=1)
        date = date.date()
        date_stop = date + datetime.timedelta(days=1)

    if date_stop is None:
        if args['--date-to'] is not None:
            date_stop = datetime.datetime.strptime(args['--date-to'], "%Y-%m-%d")
        else:
            date_stop = date + datetime.timedelta(days=1)

    if (date_stop - date).days < 1:
        raise ValueError("Date from '%s' must be at least one smaller than date to '%s'." % (date.isoformat(), date_stop.isoformat()))
    
    create_plot(args["<point_type>"], date, date_stop, args["--output-dir"])
    print "FIN"
