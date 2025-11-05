import time
import cv2
import threading
from collections import deque
from typing import Optional, Tuple
import numpy as np
import platform
import os

class CameraManager:
    """
    Webcam-only Camera Manager (USB Camera).
    - Optimized for Windows (CAP_DSHOW) and Linux (CAP_V4L2/DShow).
    - Auto-reconnect, FPS limit, thread-safe.
    - No Raspberry Pi, no picamera2.
    """

    def __init__(
        self,
        logger,
        configs: Optional[dict],
        threaded: bool = True,
        buffer_size: int = 1,
    ):
        self.logger = logger
        self.width = configs["frame_width"]
        self.height = configs["frame_height"]
        self.threaded = threaded
        self.max_fps = configs["frame_rate"]
        self.buffer_size = max(1, buffer_size)

        # Detect OS
        self.is_windows = os.name == "nt"

        # Thread state
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Frame state
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_ret: bool = False
        self._frame_counter: int = 0
        self._last_read_time: float = 0.0
        self._fps_timestamps = deque(maxlen=10)

        # OpenCV capture
        self.cap: Optional[cv2.VideoCapture] = None

        self._initialize_camera()

    def _initialize_camera(self) -> bool:
        """Open webcam with best backend for OS."""
        backend = cv2.CAP_DSHOW if self.is_windows else cv2.CAP_V4L2
        indices = [0, 1, 2, -1]  # Try common indices

        for idx in indices:
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

            # Verify
            actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.logger.info(f"Webcam opened: index={idx}, {actual_w}x{actual_h}, backend={'DSHOW' if self.is_windows else 'V4L2'}")

            self.cap = cap
            return True

        self.logger.error("No webcam found.")
        return False

    def open(self) -> bool:
        if not self.cap:
            if not self._initialize_camera():
                return False

        if self.threaded:
            self._start_thread()

        self.logger.info("Webcam ready.")
        return True

    def _start_thread(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._update_loop, daemon=True, name="WebcamReader")
        self._thread.start()
        self.logger.info("Webcam background thread started.")

    def _update_loop(self) -> None:
        min_interval = 1.0 / self.max_fps if self.max_fps > 0 else 0.0
        last_time = time.time()

        while not self._stop_event.is_set():
            now = time.time()
            elapsed = now - last_time

            if min_interval == 0 or elapsed >= min_interval:
                if not self.cap or not self.cap.isOpened():
                    if not self._reconnect():
                        time.sleep(1)
                    continue

                ret, frame = self.cap.read()
                if ret and frame is not None:
                    frame = frame[90:740, 0:800]
                    with self._lock:
                        self._latest_frame = frame
                        self._latest_ret = True
                        self._frame_counter += 1
                        self._last_read_time = now
                        self._fps_timestamps.append(now)
                    last_time = now
                else:
                    self.logger.warning("Frame dropped. Reconnecting...")
                    self._reconnect()
            else:
                time.sleep(max(0, min_interval - elapsed))

    def _reconnect(self) -> bool:
        """Reconnect to webcam."""
        self.logger.warning("Reconnecting to webcam...")
        if self.cap:
            self.cap.release()
        time.sleep(0.5)
        return self._initialize_camera()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.threaded:
            with self._lock:
                return self._latest_ret, self._latest_frame.copy() if self._latest_frame is not None else None
        else:
            if not self.cap or not self.cap.isOpened():
                return False, None
            return self.cap.read()

    def get_fps(self) -> float:
        if len(self._fps_timestamps) < 2:
            return 0.0
        elapsed = self._fps_timestamps[-1] - self._fps_timestamps[0]
        return (len(self._fps_timestamps) - 1) / max(elapsed, 1e-6)

    def is_healthy(self, timeout: float = 2.0) -> bool:
        return self._latest_ret and (time.time() - self._last_read_time < timeout)

    def flush(self, n: int = 3, timeout: float = 1.0) -> Tuple[bool, Optional[np.ndarray]]:
        start = self._frame_counter
        t0 = time.time()
        while self._frame_counter - start < n and time.time() - t0 < timeout:
            time.sleep(0.001)
        return self.read()

    def close(self, join_timeout: float = 1.0) -> None:
        if self.threaded and self._stop_event:
            self._stop_event.set()
            if self._thread:
                self._thread.join(timeout=join_timeout)
            self.logger.info("Webcam thread stopped.")

        if self.cap:
            self.cap.release()
            self.logger.info("Webcam released.")

        with self._lock:
            self._latest_frame = None
            self._latest_ret = False
            self._frame_counter = 0
            self._fps_timestamps.clear()

        self.logger.info("CameraManager closed.")
