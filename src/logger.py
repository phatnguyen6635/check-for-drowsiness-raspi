import logging
from logging.handlers import TimedRotatingFileHandler

import cv2
import os
from datetime import datetime

def create_log(log_file: str = "./logs/log_app/app.log", backup_days: int = 30) -> logging.Logger:
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
 
def save_suspected_frame(origin_frame, annotated_frame, save_dir="./logs/log_frame"):
    """
    Save the current frame as an image file in the specified directory.
    
    Args:
        frame: Image frame to save
        save_dir: Directory to save the image file
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    file_path_origin = os.path.join(save_dir, f"suspected_origin_{timestamp}.jpg")
    file_path_annotated = os.path.join(save_dir, f"suspected_annotated_{timestamp}.jpg")
    
    origin_frame = cv2.cvtColor(origin_frame, cv2.COLOR_RGB2BGR)
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(file_path_origin, origin_frame)
    cv2.imwrite(file_path_annotated, annotated_frame)