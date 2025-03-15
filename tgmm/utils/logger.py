import os
import json
import traceback
import sys
from dataclasses import dataclass

from transformers.utils import logging

from .configuration import _get_root


def log_exception_with_traceback(logger=None):
    """Logs an exception with its traceback.

    Args:
        logger: (Optional) A logging.Logger instance. If None, uses a basic
                console logger.

    Example Usage (within a 'try...except' block):

    try:
        # Code that might raise an exception
        1 / 0
    except Exception as e:
        log_exception_with_traceback()
        # or, if you have a logger:
        # log_exception_with_traceback(my_logger)
    """

    if logger is None:
        # Create a basic console logger if one isn't provided.
        logger = create_logger()  # Get the root logger

    # Get the exception information.  This is the crucial part.
    exc_type, exc_value, exc_traceback = sys.exc_info()

    # Log the exception type and message.
    logger.error(f"An exception of type {exc_type.__name__} occurred: {exc_value}")

    # Log the traceback.  This is the *most* important part.
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
    logger.error("Traceback:\n%s", "".join(tb_lines))


def create_logger(verbosity: int = logging.INFO):
    logging.set_verbosity(verbosity)
    logging.enable_default_handler()
    logging.enable_explicit_format()
    logger = logging.get_logger()
    return logger


logger = create_logger()


@dataclass
class WandbProfile(object):
    api_key: str = None
    entity: str = None
    project: str = None


def get_wandb_api_key():
    root = _get_root()
    data_path = os.path.join(root, "data", "wandb.json")
    if os.path.exists(data_path):
        with open(data_path) as f:
            profile_dict = json.load(f)
            return WandbProfile(**profile_dict)
    else:
        return WandbProfile()


wandb_profile = get_wandb_api_key()
