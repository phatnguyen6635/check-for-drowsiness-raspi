import cv2
import numpy as np
import mediapipe as mp

import queue
import time
import threading
from collections import deque
from typing import Optional, Tuple
from enum import Enum
import os
from dataclasses import dataclass


class SystemState(Enum):
    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass
class CameraFrame:
    frame: np.ndarray
    timestamp: float
    counter: int


class CameraManager:
    """
    Thread-safe webcam manager that keeps only the latest frame in a deque (maxlen=1).
    """

    def __init__(self, logger, configs: dict, buffer_size: int = 2):
        self.logger = logger
        self.width = configs["frame_width"]
        self.height = configs["frame_height"]
        self.target_fps = configs["frame_rate"]
        self.buffer_size = max(1, buffer_size)
        self.is_windows = os.name == "nt"

        # Thread synchronization
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ready_event = threading.Event()

        # Keep latest frame in a deque (maxlen=1) to enforce latest-only semantics
        self.cam_deque: deque = deque(maxlen=1)
        self._frame_counter: int = 0

        # FPS tracking
        self._fps_timestamps = deque(maxlen=15)

        # OpenCV capture
        self.cap: Optional[cv2.VideoCapture] = None
        self._state = SystemState.INITIALIZING

    def _initialize_camera(self) -> bool:
        backend = cv2.CAP_DSHOW if self.is_windows else cv2.CAP_V4L2

        for idx in range(20):
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                continue

            # Test read
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FPS, self.target_fps)

            try:
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            except Exception:
                pass

            actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = cap.get(cv2.CAP_PROP_FPS)

            self.logger.info(
                f"Camera opened: index={idx}, {int(actual_w)}x{int(actual_h)}, "
                f"FPS={actual_fps}, backend={'DSHOW' if self.is_windows else 'V4L2'}"
            )

            self.cap = cap
            return True

        self.logger.error("No webcam found.")
        return False

    def open(self) -> bool:
        if not self._initialize_camera():
            return False

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="CameraCapture"
        )
        self._thread.start()

        if self._ready_event.wait(timeout=5.0):
            self.logger.info("Camera ready with first frame.")
            return True
        else:
            self.logger.error("Camera timeout waiting for first frame.")
            return False

    def _capture_loop(self) -> None:
        """Capture loop that keeps only the newest frame in cam_deque, but outputs at target FPS."""
        consecutive_errors = 0
        max_consecutive_errors = 10
        last_diagnostic = time.time()

        # target fps control
        target_fps = getattr(self, "_target_fps", 0) or 0
        if target_fps > 0:
            period = 1.0 / float(target_fps)
        else:
            period = 0.0

        last_push_time = time.perf_counter()
        latest_frame = None
        last_read_time = None

        while not self._stop_event.is_set():
            try:
                if not self.cap or not self.cap.isOpened():
                    if not self._reconnect():
                        time.sleep(1)
                        consecutive_errors += 1
                        if consecutive_errors >= max_consecutive_errors:
                            self.logger.critical("Camera repeatedly failed. Stopping.")
                            self._state = SystemState.STOPPING
                            break
                        continue

                # Read as fast as camera provides (keep latest)
                ret, frame = self.cap.read()
                now_read = time.time()
                if not ret or frame is None:
                    consecutive_errors += 1
                    # tiny sleep to avoid busy spin when camera fails temporarily
                    time.sleep(0.001)
                    continue

                # keep the most recent frame (not yet pushed to deque)
                latest_frame = frame
                last_read_time = now_read

                # If no target fps set => push every frame (original behavior)
                if period <= 0:
                    with self._lock:
                        self._frame_counter += 1
                        cam_frame = CameraFrame(frame=latest_frame.copy(), timestamp=last_read_time, counter=self._frame_counter)
                        self.cam_deque.clear()
                        self.cam_deque.append(cam_frame)
                        self._fps_timestamps.append(last_read_time)
                        if not self._ready_event.is_set():
                            self._ready_event.set()
                    consecutive_errors = 0
                    # yield
                    time.sleep(0.001)
                    continue

                # If it's time to push the next frame according to target_fps
                now = time.perf_counter()
                if (now - last_push_time) >= period:
                    if latest_frame is not None:
                        with self._lock:
                            self._frame_counter += 1
                            # copy frame to avoid race if OpenCV reuses buffer
                            cam_frame = CameraFrame(frame=latest_frame.copy(), timestamp=last_read_time or time.time(), counter=self._frame_counter)
                            # keep latest only
                            self.cam_deque.clear()
                            self.cam_deque.append(cam_frame)
                            self._fps_timestamps.append(cam_frame.timestamp)
                            if not self._ready_event.is_set():
                                self._ready_event.set()
                        last_push_time = now
                    # reset consecutive errors on successful read/push
                    consecutive_errors = 0

                # Small sleep to yield CPU but keep responsive. 
                # Sleep time tuned: if time left until next push, sleep min(remaining/2, 1ms)
                remaining = period - (time.perf_counter() - last_push_time)
                if remaining > 0.002:
                    time.sleep(min(remaining / 2.0, 0.01))
                else:
                    # tiny yield
                    time.sleep(0.0005)

                # Diagnostic / small sleep to avoid starving other threads on some platforms
                if time.time() - last_diagnostic > 5.0:
                    last_diagnostic = time.time()

            except Exception as e:
                self.logger.error(f"Error in capture loop: {e}", exc_info=True)
                consecutive_errors += 1
                time.sleep(0.01)

        self.logger.info("Camera capture loop stopped.")

    def _reconnect(self) -> bool:
        with self._lock:
            self.logger.warning("Attempting camera reconnection...")
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
            time.sleep(0.5)
            return self._initialize_camera()

    def get_latest_frame_info(self) -> Optional[CameraFrame]:
        with self._lock:
            if not self.cam_deque:
                return None
            return self.cam_deque[-1]

    def get_fps(self) -> float:
        with self._lock:
            if len(self._fps_timestamps) < 2:
                return 0.0
            elapsed = self._fps_timestamps[-1] - self._fps_timestamps[0]
            if elapsed <= 0:
                return 0.0
            return (len(self._fps_timestamps) - 1) / elapsed

    def is_healthy(self, timeout: float = 2.0) -> bool:
        with self._lock:
            if not self.cam_deque:
                return False
            age = time.time() - self.cam_deque[-1].timestamp
            return age < timeout

    def get_frame_counter(self) -> int:
        with self._lock:
            return self._frame_counter

    def close(self, timeout: float = 2.0) -> None:
        self.logger.info("Closing camera...")
        self._state = SystemState.STOPPING
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

        with self._lock:
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
            self.cam_deque.clear()
            self._frame_counter = 0
            self._fps_timestamps.clear()

        self._state = SystemState.STOPPED
        self.logger.info("Camera closed.")


class MediaPipeProcessor:
    def __init__(self, detector, result_deque: deque, logger):
        self.detector = detector
        self.result_deque = result_deque
        self.logger = logger

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._state = SystemState.INITIALIZING

        self._timestamp_ms = 0
        self._frame_interval_ms = 33
        self._last_processed_counter = -1
        self._lock = threading.Lock()

        self._error_count = 0
        self._max_errors = 5

    def start(self, cam: CameraManager) -> None:
        self._stop_event.clear()
        if cam.target_fps > 0:
            self._frame_interval_ms = int(1000 / cam.target_fps)

        self._thread = threading.Thread(
            target=self._process_loop,
            args=(cam,),
            daemon=True,
            name="MediaPipeProcessor"
        )
        self._thread.start()
        self._state = SystemState.RUNNING
        self.logger.info(f"MediaPipe processor started (interval: {self._frame_interval_ms}ms)")

    def _process_loop(self, cam: CameraManager) -> None:
        while not self._stop_event.is_set():
            try:
                frame_info = cam.get_latest_frame_info()
                if frame_info is None:
                    time.sleep(0.001)
                    continue

                with self._lock:
                    if frame_info.counter == self._last_processed_counter:
                        time.sleep(0.001)
                        continue

                    self._last_processed_counter = frame_info.counter
                    self._timestamp_ms += self._frame_interval_ms
                    current_timestamp = self._timestamp_ms

                # Drop stale frames
                if time.time() - frame_info.timestamp > 0.5:
                    # too old
                    continue
                
                time.sleep(0.002)  # Yield to other threads

                # Convert color and build mp.Image
                frame_rgb = cv2.cvtColor(frame_info.frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                # Use detector async API; callback should push results to result_deque
                self.detector.detect_async(mp_image, current_timestamp)
                self._error_count = 0

            except ValueError as e:
                if "monotonically increasing" in str(e):
                    self.logger.error(f"Timestamp error: {e}")
                    with self._lock:
                        self._timestamp_ms += self._frame_interval_ms * 5
                        self._error_count += 1
                else:
                    self.logger.error(f"ValueError: {e}", exc_info=True)
                    self._error_count += 1

                if self._error_count > self._max_errors:
                    self.logger.critical("Too many processing errors")
                    self._stop_event.set()
                    
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}", exc_info=True)
                self._error_count += 1
                if self._error_count > self._max_errors:
                    self._stop_event.set()
                time.sleep(0.01)

        self.logger.info("MediaPipe processor stopped.")

    def stop(self, timeout: float = 1.0) -> None:
        self.logger.info("Stopping MediaPipe processor...")
        self._state = SystemState.STOPPING
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._state = SystemState.STOPPED
