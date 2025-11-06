import cv2
import numpy as np
import mediapipe as mp

import time
import threading
from collections import deque
from typing import Optional, Tuple
from enum import Enum
import queue
import os
from dataclasses import dataclass


class SystemState(Enum):
    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass
class CameraFrame:
    """Thread-safe camera frame wrapper"""
    frame: np.ndarray
    timestamp: float
    counter: int


class CameraManager:
    """
    Thread-safe webcam manager với proper synchronization.
    """

    def __init__(self, logger, configs: dict, buffer_size: int = 3):
        self.logger = logger
        self.width = configs["frame_width"]
        self.height = configs["frame_height"]
        self.target_fps = configs["frame_rate"]
        self.buffer_size = max(1, buffer_size)
        self.is_windows = os.name == "nt"

        # Thread synchronization
        self._lock = threading.RLock()  # Recursive lock cho nested calls
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ready_event = threading.Event()

        # Frame state (protected by lock)
        self._latest_frame: Optional[CameraFrame] = None
        self._frame_counter: int = 0
        self._fps_timestamps = deque(maxlen=30)
        self._last_error_time: float = 0
        self._error_count: int = 0

        # OpenCV capture
        self.cap: Optional[cv2.VideoCapture] = None
        self._state = SystemState.INITIALIZING

    def _initialize_camera(self) -> bool:
        """Open webcam with best backend for OS."""
        backend = cv2.CAP_DSHOW if self.is_windows else cv2.CAP_V4L2

        for idx in range(20): # Try common indices
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                continue

            # Test read
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                continue

            # Apply settings
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FPS, self.target_fps)

            # Verify
            actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.logger.info(f"Webcam opened: index={idx}, {actual_w}x{actual_h}, backend={'DSHOW' if self.is_windows else 'V4L2'}")

            self.cap = cap
            return True

        self.logger.error("No webcam found.")
        return False

    def open(self) -> bool:
        """Start camera with background thread"""
        if not self._initialize_camera():
            return False

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="CameraCapture"
        )
        self._thread.start()

        # Wait for first frame
        if self._ready_event.wait(timeout=5.0):
            self.logger.info("Camera ready with first frame.")
            return True
        else:
            self.logger.error("Camera timeout waiting for first frame.")
            return False

    def _capture_loop(self) -> None:
        """Main capture loop - ABSOLUTE MAXIMUM SPEED"""
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        # Performance tracking
        capture_times = deque(maxlen=100)
        last_diagnostic = time.time()

        while not self._stop_event.is_set():
            capture_start = time.time()
            
            try:
                # === ZERO DELAY CAPTURE ===
                
                # Check camera health
                if not self.cap or not self.cap.isOpened():
                    if not self._reconnect():
                        time.sleep(1)
                        consecutive_errors += 1
                        if consecutive_errors >= max_consecutive_errors:
                            self.logger.critical("Camera repeatedly failed. Stopping.")
                            self._state = SystemState.STOPPING
                            break
                        continue

                # Use grab() for fastest possible capture
                grabbed = self.cap.grab()
                
                if not grabbed:
                    consecutive_errors += 1
                    if consecutive_errors >= 5:
                        self.logger.warning(f"Grab failed {consecutive_errors} times")
                        self._reconnect()
                    continue
                
                # Retrieve frame (this does the actual decoding)
                ret, frame = self.cap.retrieve()
                
                if ret and frame is not None:
                    # Crop frame (if needed)
                    #
                    #
                    now = time.time()
                    
                    # Thread-safe update - MINIMAL LOCK TIME
                    with self._lock:
                        self._frame_counter += 1
                        self._latest_frame = CameraFrame(
                            frame=frame,
                            timestamp=now,
                            counter=self._frame_counter
                        )
                        self._fps_timestamps.append(now)
                        
                        if not self._ready_event.is_set():
                            self._ready_event.set()
                    
                    consecutive_errors = 0
                    
                    # Track capture time
                    capture_times.append(time.time() - capture_start)
                    
                else:
                    consecutive_errors += 1
                    if consecutive_errors >= 3:
                        self._reconnect()

            except Exception as e:
                self.logger.error(f"Error in capture loop: {e}", exc_info=True)
                consecutive_errors += 1
                time.sleep(0.01)

        self.logger.info("Camera capture loop stopped.")

    def _reconnect(self) -> bool:
        """Thread-safe camera reconnection"""
        with self._lock:
            self.logger.warning("Attempting camera reconnection...")
            
            if self.cap:
                self.cap.release()
                self.cap = None
            
            time.sleep(0.5)
            return self._initialize_camera()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Thread-safe frame read"""
        with self._lock:
            if self._latest_frame is None:
                return False, None
            return True, self._latest_frame.frame.copy()

    def get_latest_frame_info(self) -> Optional[CameraFrame]:
        """Get complete frame info (thread-safe)"""
        with self._lock:
            if self._latest_frame is None:
                return None
            return CameraFrame(
                frame=self._latest_frame.frame.copy(),
                timestamp=self._latest_frame.timestamp,
                counter=self._latest_frame.counter
            )

    def get_fps(self) -> float:
        """Calculate actual FPS"""
        with self._lock:
            if len(self._fps_timestamps) < 2:
                return 0.0
            elapsed = self._fps_timestamps[-1] - self._fps_timestamps[0]
            return (len(self._fps_timestamps) - 1) / max(elapsed, 1e-6)

    def is_healthy(self, timeout: float = 2.0) -> bool:
        """Check camera health"""
        with self._lock:
            if self._latest_frame is None:
                return False
            age = time.time() - self._latest_frame.timestamp
            return age < timeout

    def get_frame_counter(self) -> int:
        """Get current frame counter (thread-safe)"""
        with self._lock:
            return self._frame_counter

    def wait_for_new_frame(self, last_counter: int, timeout: float = 0.1) -> bool:
        """Chờ đợi frame mới (NON-BLOCKING - optimized)"""
        # Quick check first (no sleep)
        with self._lock:
            if self._frame_counter > last_counter:
                return True
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self._lock:
                if self._frame_counter > last_counter:
                    return True
            time.sleep(0.0001)  # Micro sleep - 0.1ms
        
        return False

    def close(self, timeout: float = 2.0) -> None:
        """Graceful shutdown"""
        self.logger.info("Closing camera...")
        self._state = SystemState.STOPPING
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

        with self._lock:
            if self.cap:
                self.cap.release()
                self.cap = None
            
            self._latest_frame = None
            self._frame_counter = 0
            self._fps_timestamps.clear()

        self._state = SystemState.STOPPED
        self.logger.info("Camera closed.")

class FrameWaiter:
    """Helper to wait for a new frame from the camera - OPTIMIZED VERSION"""
    def __init__(self):
        self._last_counter = -1
        self._lock = threading.Lock()
    
    def wait_for_new_frame(self, cam: 'CameraManager', timeout: float = 0.05) -> bool:
        """Wait until a new frame is available - FAST PATH"""
        with self._lock:
            current_counter = cam.get_frame_counter()
            
            # Fast path: new frame already available
            if current_counter > self._last_counter:
                self._last_counter = current_counter
                return True
        
        # Slow path: Wait with micro-sleeps
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self._lock:
                current_counter = cam.get_frame_counter()
                if current_counter > self._last_counter:
                    self._last_counter = current_counter
                    return True
            time.sleep(0.0001)  # 0.1ms micro-sleep
        
        with self._lock:
            self._last_counter = cam.get_frame_counter()
        
        return False

# ==================== MEDIAPIPE PROCESSOR ====================
class MediaPipeProcessor:
    """
    Thread-safe MediaPipe processor với monotonic timestamps.
    """

    def __init__(self, detector, result_queue: queue.Queue, logger):
        self.detector = detector
        self.result_queue = result_queue
        self.logger = logger
        
        # Thread control
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._state = SystemState.INITIALIZING
        
        # Timestamp management (monotonic)
        self._timestamp_ms = 0
        self._frame_interval_ms = 33  # ~30 FPS
        self._last_processed_counter = -1
        self._lock = threading.Lock()

    def start(self, cam: CameraManager) -> None:
        """Start processing thread"""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._process_loop,
            args=(cam,),
            daemon=True,
            name="MediaPipeProcessor"
        )
        self._thread.start()
        self._state = SystemState.RUNNING
        self.logger.info("MediaPipe processor started.")

    def _process_loop(self, cam: CameraManager) -> None:
        """Main processing loop"""
        while not self._stop_event.is_set():
            try:
                # Get latest frame
                frame_info = cam.get_latest_frame_info()
                
                if frame_info is None:
                    time.sleep(0.001)
                    continue

                # Skip if already processed
                with self._lock:
                    if frame_info.counter == self._last_processed_counter:
                        time.sleep(0.001)
                        continue
                    
                    self._last_processed_counter = frame_info.counter
                    # Monotonic timestamp increment
                    self._timestamp_ms += self._frame_interval_ms
                    current_timestamp = self._timestamp_ms

                # Convert to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame_info.frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                # Async detection
                self.detector.detect_async(mp_image, current_timestamp)

            except ValueError as e:
                if "monotonically increasing" in str(e):
                    self.logger.error(f"Timestamp error: {e}")
                    with self._lock:
                        self._timestamp_ms += self._frame_interval_ms * 2
                else:
                    self.logger.error(f"ValueError: {e}", exc_info=True)
                time.sleep(0.01)
                
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}", exc_info=True)
                time.sleep(0.01)

        self.logger.info("MediaPipe processor stopped.")

    def stop(self, timeout: float = 2.0) -> None:
        """Stop processor gracefully"""
        self.logger.info("Stopping MediaPipe processor...")
        self._state = SystemState.STOPPING
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

        self._state = SystemState.STOPPED
