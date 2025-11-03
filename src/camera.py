import time
import cv2
import threading

class CameraManager:
    def __init__(self, logger, width=1280, height=720, threaded=True, max_fps=0):
        self.logger = logger
        self.width = width
        self.height = height

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
        
        self.cap = self.get_real_camera()


    def get_real_camera(self):
        for i in range(20):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.logger.info(f"Using camera at index {i}")
                    return cap
            cap.release()
        self.logger.error("Cannot find any camera")
        return None

    def open(self):
        
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
    
    
    def get_fps(self):
        return self._frame_counter / (time.time() - self._last_read_time + 1e-6)
