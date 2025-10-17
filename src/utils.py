
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

import os
import time
import threading
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
import yaml
from yaml import Loader

EYE_BLENDSHAPE_CATEGORIES = [
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "eyeLookUpLeft", "eyeLookUpRight",
    "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight",
]

# Display settings
FONT_FACE = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE_INFO = 0.5
FONT_SCALE_ALERT = 0.8
COLOR_INFO = (50, 205, 50)      # Lime green
COLOR_ALERT = (255, 69, 0)      # Red-orange
FONT_THICKNESS_INFO = 1
FONT_THICKNESS_ALERT = 2
TEXT_LINE_HEIGHT = 22
TEXT_MARGIN = 15

def load_config(config_path='configs/configs.yaml'):
    """Load configuration from YAML file with error handling"""
    try:
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path) as stream:
            config = yaml.load(stream, Loader=Loader)
        
        required_keys = ['model_path', 'blink_threshold']
        for key in required_keys:
            if key not in config:
                raise KeyError(f"Missing required config key: {key}")
        
        return config
    
    except Exception as e:
        raise RuntimeError(f"Failed to load config: {e}")

def create_log(log_file: str = "logs/app.log", backup_days: int = 10) -> logging.Logger:
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

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    file_handler = TimedRotatingFileHandler(
        log_file,
        when="midnight",
        interval=1,
        backupCount=backup_days,
        encoding="utf8",
        utc=False
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def create_face_detector(model_path, logger):
    """Create MediaPipe face detector with error handling"""
    try:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.7
        )
        detector = vision.FaceLandmarker.create_from_options(options)
        logger.info(f"Face detector initialized with model: {model_path}")
        return detector
    
    except Exception as e:
        logger.error(f"Failed to create face detector: {e}")
        raise
    
def draw_face_landmarks(rgb_image, detection_result):
    """
    Draw facial landmarks and mesh on the input image.
    
    Args:
        rgb_image: Input image in RGB format
        detection_result: MediaPipe face detection result
        
    Returns:
        Annotated image with facial landmarks drawn
    """
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Convert landmarks to protobuf format
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) 
            for lm in face_landmarks
        ])

        # Draw face mesh tesselation
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style()
        )
        
        # Draw face contours
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style()
        )
        
        # Draw iris landmarks
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style()
        )

    return annotated_image

def render_blendshape_metrics(image, blendshapes):
    """
    Display eye blendshape metrics on the image.
    
    Args:
        image: Image to draw text on
        blendshapes: List of blendshape categories
        
    Returns:
        Dictionary containing blink scores for both eyes
    """
    y_position = TEXT_MARGIN + TEXT_LINE_HEIGHT
    blink_scores = {"left": 0.0, "right": 0.0}
    
    for category in blendshapes:
        if category.category_name in EYE_BLENDSHAPE_CATEGORIES:
            display_text = f"{category.category_name}: {category.score:.2f}"
            cv2.putText(
                image, display_text, 
                (TEXT_MARGIN, y_position),
                FONT_FACE, FONT_SCALE_INFO, COLOR_INFO, 
                FONT_THICKNESS_INFO, cv2.LINE_AA
            )
            y_position += TEXT_LINE_HEIGHT
        
        # Capture blink scores
        if category.category_name == "eyeBlinkLeft":
            blink_scores["left"] = category.score
        elif category.category_name == "eyeBlinkRight":
            blink_scores["right"] = category.score
    
    return blink_scores, y_position

def display_drowsiness_alert(image, blink_left, blink_right, y_position, blink_threshold):
    """
    Display appropriate alert based on eye closure detection.
    
    Args:
        image: Image to draw alert on
        blink_left: Left eye blink score
        blink_right: Right eye blink score
        y_position: Vertical position to display alert
    """
    
    alert_y = y_position + 10
    drowsy = False
    
    if blink_left > blink_threshold and blink_right > blink_threshold:
        alert_message = "ALERT: Drowsiness Detected - Stay Awake!"
        cv2.putText(
            image, alert_message, 
            (TEXT_MARGIN, alert_y),
            FONT_FACE, FONT_SCALE_ALERT, COLOR_ALERT, 
            FONT_THICKNESS_ALERT, cv2.LINE_AA
        )
        drowsy = True

    return drowsy
         

class CameraManager:
    def __init__(self, logger, camera_index=0, width=1280, height=720, threaded=True, max_fps=0):
        self.logger = logger
        self.index = camera_index
        self.width = width
        self.height = height
        self.cap = None

        # Threaded reader state
        self.threaded = bool(threaded)
        self.max_fps = float(max_fps) if max_fps is not None else 0.0
        self._stop_event = None
        self._thread = None
        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_ret = False
        self._frame_counter = 0
        self._last_read_time = 0.0

    def open(self):
        self.logger and self.logger.info(f"Initializing camera index {self.index}")
        self.cap = cv2.VideoCapture(self.index)

        if not self.cap.isOpened():
            self.logger and self.logger.error(f"Cannot open camera (index={self.index}).")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.logger and self.logger.info(f"Camera is ready ({self.width}x{self.height}).")

        if self.threaded:
            self._start_thread()

        return True

    def _start_thread(self):
        if self._thread and self._thread.is_alive():
            return

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._update_loop, daemon=True, name="CameraManagerReader")
        self._thread.start()
        self.logger and self.logger.info("Camera background reader started.")

    def _update_loop(self):
        sleep_interval = 1.0 / self.max_fps if (self.max_fps and self.max_fps > 0) else 0
        while not (self._stop_event and self._stop_event.is_set()):
            try:
                ret, frame = self.cap.read()
                if not ret:
                    pass

                with self._lock:
                    self._latest_frame = frame
                    self._latest_ret = bool(ret)
                    self._frame_counter += 1
                    self._last_read_time = time.time()

            except Exception as e:
                if self.logger:
                    self.logger.exception(f"Camera background read error: {e}")
            if sleep_interval:
                time.sleep(sleep_interval)

    def read(self):
        if self.threaded:
            with self._lock:
                return self._latest_ret, self._latest_frame
        else:
            if not self.cap:
                self.logger and self.logger.warning("Camera is not open.")
                return False, None
            try:
                ret, frame = self.cap.read()
                if not ret:
                    self.logger and self.logger.error("Unable to read frame from camera.")
                    return False, None
                return True, frame
            except Exception as e:
                self.logger and self.logger.error(f"Error reading frame: {e}")
                return False, None

    def flush(self, n=3, timeout=1.0):
        if not self.threaded:
            last_ret, last_frame = False, None
            t0 = time.time()
            for _ in range(n):
                if time.time() - t0 > timeout:
                    break
                last_ret, last_frame = self.read()
            return last_ret, last_frame

        start = self._frame_counter
        t0 = time.time()
        while (self._frame_counter - start) < n and (time.time() - t0) < timeout:
            time.sleep(0.005)
        with self._lock:
            return self._latest_ret, self._latest_frame

    def is_opened(self):
        return self.cap is not None and self.cap.isOpened()

    def close(self, join_timeout=1.0):
        if self.threaded and self._stop_event:
            self._stop_event.set()
            if self._thread:
                self._thread.join(timeout=join_timeout)
            self.logger and self.logger.info("Camera background reader stopped.")

        if self.cap:
            try:
                self.cap.release()
                self.logger and self.logger.info("Camera connection closed.")
            except Exception as e:
                self.logger and self.logger.error(f"Error releasing camera: {e}")
        with self._lock:
            self._latest_frame = None
            self._latest_ret = False
            self._frame_counter = 0
            self._last_read_time = 0.0
