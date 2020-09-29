import logging

from datetime import timedelta

n = [20, 40, 60]

no_circle_tmg_time = [timedelta(0), timedelta(0), timedelta(0)]
no_circle_bilp_time = [timedelta(0), timedelta(0), timedelta(0)]
max_circle_tmg_time = [timedelta(0), timedelta(0), timedelta(0)]
max_circle_bilp_time = [timedelta(0), timedelta(0), timedelta(0)]
single_circle_tmg_time = [timedelta(0), timedelta(0), timedelta(0)]
single_circle_bilp_time = [timedelta(0), timedelta(0), timedelta(0)]

count = [21, 21, 21]


def log_results():
    logging.debug("count: " + str(count))

    logging.debug("no_circle_tmg_time")
    for entry, num, c in zip(no_circle_tmg_time, n, count):
        logging.debug(str(num) + ": " + str(entry))
        logging.debug(str(num) + " mean-value: " + str(entry/c))

    logging.debug("no_circle_bilp_time")
    for entry, num, c in zip(no_circle_bilp_time, n, count):
        logging.debug(str(num) + ": " + str(entry))
        logging.debug(str(num) + " mean-value: " + str(entry/c))

    logging.debug("max_circle_tmg_time")
    for entry, num, c in zip(max_circle_tmg_time, n, count):
        logging.debug(str(num) + ": " + str(entry))
        logging.debug(str(num) + " mean-value: " + str(entry/c))

    logging.debug("max_circle_bilp_time")
    for entry, num, c in zip(max_circle_bilp_time, n, count):
        logging.debug(str(num) + ": " + str(entry))
        logging.debug(str(num) + " mean-value: " + str(entry/c))

    logging.debug("single_circle_tmg_time")
    for entry, num, c in zip(single_circle_tmg_time, n, count):
        logging.debug(str(num) + ": " + str(entry))
        logging.debug(str(num) + " mean-value: " + str(entry/c))

    logging.debug("single_circle_bilp_time")
    for entry, num, c in zip(single_circle_bilp_time, n, count):
        logging.debug(str(num) + ": " + str(entry))
        logging.debug(str(num) + " mean-value: " + str(entry/c))
