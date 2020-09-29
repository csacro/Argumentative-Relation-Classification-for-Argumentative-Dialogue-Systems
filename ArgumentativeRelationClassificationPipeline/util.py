import logging
import sys


def init_logging(level, filename: str):
    """
    initialises logging
    :param level: logging level
    :param filename: output file, <filename>.log
    """

    logging.getLogger().handlers.clear()

    # logging settings
    logger = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s')
    logger.setLevel(logging.DEBUG)

    # logging to file
    file_handler = logging.FileHandler(filename + '.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # logging to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.level = level
    logger.addHandler(console_handler)
