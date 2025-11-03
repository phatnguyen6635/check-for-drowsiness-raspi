import logging
from logging.handlers import TimedRotatingFileHandler

import os


def create_log(log_file: str = "./logs/app.log", backup_days: int = 30) -> logging.Logger:
    """
    Create logger to rotate files by day, automatically delete old logs after backup_days days.
    Args:
        log_file (str): Main log file path 
        backup_days (int): Number of days to retain logs (default: 10)
    Returns:
        logging.Logger
    """

    logger = logging.getLogger("app_logger")

    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s (%(filename)s:%(lineno)d)",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler() # output to console
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    file_handler = TimedRotatingFileHandler(
        log_file,
        when="midnight", # rotate daily
        interval=1, # every 1 day
        backupCount=backup_days, # keep logs for backup_days days
        encoding="ascii",
        utc=False
    ) # output to file
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
 