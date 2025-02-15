import traceback
import sys

from transformers.utils import logging


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
