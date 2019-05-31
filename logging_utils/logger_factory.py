import logging
import sys
from datetime import datetime

file_handler = logging.FileHandler(filename='logs/{:%Y-%m-%d_%H:%M:%S}.log'.format(datetime.now()))
stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=handlers)


def getBasicLogger(classname):
    logger = logging.getLogger(classname)

    return logger
