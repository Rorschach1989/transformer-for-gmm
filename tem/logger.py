from transformers.utils import logging


def create_logger(verbosity: int = logging.INFO):
    logging.set_verbosity(verbosity)
    logging.enable_default_handler()
    logging.enable_explicit_format()
    logger = logging.get_logger()
    return logger


logger = create_logger()
