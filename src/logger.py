import logging


def get_logger():
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    init_logger = logging.getLogger()
    return init_logger


logger = get_logger()
