import tkinter as tk
from tkinter import ttk
import threading
import queue
import time
import cv2
import yaml
from collections import deque
from PIL import Image, ImageTk
import sys
import numpy as np
from dataclasses import dataclass
from typing import Optional

# Local modules (same as in main.py)
from src.threadflow import CameraManager, MediaPipeProcessor
from src.logger import create_log, save_suspected_frame
from src.utils import initialize_gpio, set_relay, cleanup_resources
from src.models import (
    create_face_detector,
    draw_face_landmarks,
    calculate_gaze_direction,
    draw_gaze_arrows,
    render_blendshape_metrics,
    display_eyes_status,
    display_info,
    get_head_orientation,
    display_head_orientation,
)

# ==================== SHARED UI QUEUE ====================
ui_frame_queue = queue.Queue(maxsize=1)

# ==================== DATA CLASS ====================
@dataclass
class DetectionResult:
    frame_rgb: np.ndarray
    detection: any
    timestamp_ms: int
    frame_counter: int

# ==================== THREAD-SAFE CONFIG MANAGER (HOT-RELOADABLE) ====================
class ConfigManager:
    def __init__(self, yaml_path="configs/configs.yaml"):
        self.yaml_path = yaml_path
        self.lock = threading.Lock()
        self.config = {
        "model_path": "./models/face_landmarker_v2_with_blendshapes.task",
        "num_faces": 1,
        "min_face_detection_confidence": 0.4,
        "min_face_presence_confidence": 0.6,
        "blink_threshold_pitch": 0.7,
        "blink_threshold_wo_pitch": 0.4,
        "pitch_threshold_positive": 40,
        "pitch_threshold_negative": -40,
        "delay_drowsy_threshold": 2.0,
        "perclos_window_size": 100,
        "perclos_threshold": 0.5,
        "frame_width": 1280,
        "frame_height": 720,
        "frame_rate": 30,
        "led_pin": 21
        }
        try:
            self.load()
        except Exception:
            pass

    def load(self):
        with open(self.yaml_path, "r") as f:
            data = yaml.safe_load(f) or {}
        with self.lock:
            self.config.update(data)

    def get(self, key, default=None):
        with self.lock:
            return self.config.get(key, default)

    def snapshot(self):
        with self.lock:
            return dict(self.config)

    def set_many(self, new_dict: dict):
        with self.lock:
            # disallow cam_id/display_fps modifications (they are intentionally not supported)
            new_dict = {k: v for k, v in new_dict.items() if k not in ("cam_id", "display_fps")}
            self.config.update(new_dict)

    def save(self):
        try:
            with open(self.yaml_path, "w") as f:
                with self.lock:
                    yaml.safe_dump(self.config, f)
        except Exception as e:
            print(f"[CONFIG] WARNING: Could not save config: {e}")

# ==================== RESULT CALLBACK (same logic as main.py) ====================

def create_result_callback(result_deque: deque, result_lock: threading.Lock, stop_event: threading.Event, logger):
    def result_callback(result, output_image, timestamp_ms: int) -> None:
        if stop_event.is_set():
            return
        try:
            frame_rgb = output_image.numpy_view()
            with result_lock:
                result_deque.append(
                    DetectionResult(
                        frame_rgb=frame_rgb,
                        detection=result,
                        timestamp_ms=timestamp_ms,
                        frame_counter=0,
                    )
                )
                time.sleep(0.001)
        except Exception as e:
            logger.error(f"Error in result callback: {e}")

    return result_callback

# ==================== DISPLAY & PROCESS (reads config live from ConfigManager) ====================

def display_and_process(
    result_deque: deque,
    result_lock: threading.Lock,
    stop_event: threading.Event,
    config_mgr: ConfigManager,
    logger,
    gpio_enabled: bool,
    led_pin: int,
    ui_queue: queue.Queue,
) -> None:
    logger.info("Display & logic loop started.")

    drowsy_prev = False
    delay_drowsy = None
    eye_closed_history = deque(maxlen=30)
    last_detection_result: Optional[DetectionResult] = None
    last_displayed_frame = None
    relay_on = False
    main_loop_fps_timestamps = deque(maxlen=30)

    while not stop_event.is_set():
        loop_start = time.time()

        # read config snapshot (hot-reloadable)
        configs = config_mgr.snapshot()
        blink_threshold_wo_pitch = configs.get("blink_threshold_wo_pitch", 0.15)
        blink_threshold_pitch = configs.get("blink_threshold_pitch", 0.25)
        pitch_threshold_positive = configs.get("pitch_threshold_positive", 5.0)
        pitch_threshold_negative = configs.get("pitch_threshold_negative", -5.0)
        delay_drowsy_threshold = configs.get("delay_drowsy_threshold", 3.0)
        perclos_window_size = configs.get("perclos_window_size", 30)
        perclos_threshold = configs.get("perclos_threshold", 0.8)

        # adapt PERCLOS window if changed
        if eye_closed_history.maxlen != perclos_window_size:
            eye_closed_history = deque(maxlen=perclos_window_size)

        main_target_fps = configs.get("frame_rate", 15)
        interval = 1.0 / max(1.0, float(main_target_fps))

        with result_lock:
            if result_deque:
                last_detection_result = result_deque[-1]
            else:
                last_detection_result = None
        
        is_alert = False
        if last_detection_result is not None:
            frame_rgb = last_detection_result.frame_rgb
            detection_result = last_detection_result.detection

            if detection_result and getattr(detection_result, 'face_landmarks', None):
                try:
                    blendshapes = detection_result.face_blendshapes[0]
                    face_landmarks = detection_result.face_landmarks[0]

                    annotated_frame = draw_face_landmarks(frame_rgb, detection_result)
                    gaze_info = calculate_gaze_direction(face_landmarks, blendshapes, annotated_frame.shape)
                    annotated_frame = draw_gaze_arrows(annotated_frame, gaze_info)
                    annotated_frame = cv2.flip(annotated_frame, 1)

                    main_fps = len(main_loop_fps_timestamps) / max(
                        main_loop_fps_timestamps[-1] - main_loop_fps_timestamps[0], 1e-6
                    ) if len(main_loop_fps_timestamps) >= 2 else 0.0

                    annotated_frame = display_info(annotated_frame, main_fps)

                    annotated_frame, blink_scores, text_end_y = render_blendshape_metrics(annotated_frame, blendshapes)

                    annotated_frame = display_eyes_status(
                        annotated_frame,
                        blink_scores["left"],
                        blink_scores["right"],
                        text_end_y,
                        blink_threshold_wo_pitch,
                    )

                    head_orientation = get_head_orientation(detection_result.facial_transformation_matrixes[0])
                    annotated_frame = display_head_orientation(annotated_frame, head_orientation)

                    pitch = head_orientation.get("pitch", 0)
                    blink_threshold = (
                        blink_threshold_wo_pitch
                        if pitch_threshold_negative < pitch < pitch_threshold_positive
                        else blink_threshold_pitch
                    )

                    drowsy = (
                        blink_scores["left"] > blink_threshold and
                        blink_scores["right"] > blink_threshold
                    )

                    if drowsy:
                        if not drowsy_prev:
                            delay_drowsy = time.time()
                        elapsed = time.time() - delay_drowsy if delay_drowsy else 0
                        is_alert = elapsed > delay_drowsy_threshold
                    else:
                        delay_drowsy = None
                        is_alert = False
                    drowsy_prev = drowsy

                    eye_closed_history.append(drowsy)
                    if eye_closed_history and len(eye_closed_history) >= perclos_window_size:
                        perclos = sum(eye_closed_history) / len(eye_closed_history)
                    else:
                        perclos = 0
                    
                    if perclos >= perclos_threshold:
                        is_alert = True

                    # small yield
                    time.sleep(0.001)
                    if is_alert:
                        logger.warning("DROWSINESS DETECTED!")
                        try:
                            save_suspected_frame(
                                origin_frame=display_info(cv2.flip(frame_rgb, 1), main_fps),
                                annotated_frame=annotated_frame,
                            )
                        except Exception as e:
                            logger.error(f"Error saving suspected frame: {e}")
                        eye_closed_history.clear()
                        

                    display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                    last_displayed_frame = display_frame
                except Exception as e:
                    logger.error(f"Error in display processing: {e}", exc_info=True)
                    display_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    last_displayed_frame = display_frame
            else:
                annotated_frame = cv2.flip(frame_rgb, 1)
                main_fps = len(main_loop_fps_timestamps) / max(
                    main_loop_fps_timestamps[-1] - main_loop_fps_timestamps[0], 1e-6
                ) if len(main_loop_fps_timestamps) >= 2 else 0.0
                annotated_frame = display_info(annotated_frame, main_fps)
                display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                last_displayed_frame = display_frame
        else:
            if last_displayed_frame is None:
                placeholder = np.zeros((configs.get("frame_height", 720), configs.get("frame_width", 1280), 3), dtype=np.uint8)
                placeholder = display_info(placeholder, 0)
                display_frame = placeholder
                last_displayed_frame = display_frame
            else:
                display_frame = last_displayed_frame
                
        if is_alert and not relay_on:
            set_relay(gpio_enabled, led_pin, logger, True)
            relay_on = True
        elif not is_alert and relay_on:
            set_relay(gpio_enabled, led_pin, logger, False)
            relay_on = False
            
        try:
            if ui_queue.full():
                try:
                    _ = ui_queue.get_nowait()
                except Exception:
                    pass
            ui_queue.put_nowait(display_frame)
        except Exception:
            pass

        main_loop_fps_timestamps.append(time.time())
        elapsed = time.time() - loop_start
        to_sleep = interval - elapsed
        if to_sleep > 0:
            time.sleep(to_sleep)

    logger.info("Display loop terminated.")

# ==================== UI: CameraView (centered image, background fill) ====================
class CameraView(tk.Canvas):
    def __init__(self, parent, config_mgr: ConfigManager):
        initial_w = config_mgr.get("frame_width", 640)
        initial_h = config_mgr.get("frame_height", 480)
        super().__init__(parent, width=initial_w, height=initial_h, bg="#2b2b2b", highlightthickness=1, relief=tk.SUNKEN)
        self.config_mgr = config_mgr
        self._photo_image = None
        self.image_id = None
        self.last_displayed_frame = None
        self.update_frame_loop_id = None
        self.bind("<Configure>", self.on_resize)
        self.update_frame()

    def on_resize(self, event):
        self.redraw_last_frame()

    def redraw_last_frame(self):
        if self.last_displayed_frame is None:
            return
        try:
            canvas_w = max(1, self.winfo_width())
            canvas_h = max(1, self.winfo_height())
            frame_rgb = self.last_displayed_frame
            fh, fw = frame_rgb.shape[:2]

            scale = min(canvas_w / fw, canvas_h / fh, 1.0)
            target_w = int(fw * scale)
            target_h = int(fh * scale)

            # 🩹 fix: tránh lỗi resize 0
            if target_w <= 0 or target_h <= 0:
                return

            img = Image.fromarray(frame_rgb)
            img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            self._photo_image = imgtk

            x = (canvas_w - target_w) // 2
            y = (canvas_h - target_h) // 2

            if self.image_id is None:
                self.image_id = self.create_image(x, y, anchor=tk.NW, image=imgtk)
            else:
                self.coords(self.image_id, x, y)
                self.itemconfig(self.image_id, image=imgtk)
        except Exception as e:
            print("[CameraView] redraw error:", e)

    def update_frame(self):
        try:
            frame = None
            try:
                frame = ui_frame_queue.get_nowait()
            except queue.Empty:
                frame = None

            if frame is not None:
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                except Exception:
                    frame_rgb = frame
                self.last_displayed_frame = frame_rgb
                self.redraw_last_frame()
            else:
                if self.last_displayed_frame is None:
                    h = self.config_mgr.get("frame_height", 720)
                    w = self.config_mgr.get("frame_width", 1280)
                    placeholder = np.zeros((h, w, 3), dtype=np.uint8)
                    placeholder = display_info(placeholder, 0)
                    self.last_displayed_frame = cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
                    self.redraw_last_frame()
        except Exception as e:
            print("[CameraView] update error:", e)
        finally:
            self.update_frame_loop_id = self.after(30, self.update_frame)

# ==================== UI: Config Editor with Vietnamese labels ====================
class ConfigEditor(ttk.Frame):
    
    DISPLAY_LABELS = {
        "model_path": "Model Path",
        "led_pin": "LED Output Pin",
        "blink_threshold_wo_pitch": "Blink Threshold (No Head Tilt)",
        "blink_threshold_pitch": "Blink Threshold (With Head Tilt)",
        "pitch_threshold_negative": "Negative Pitch Threshold (deg)",
        "pitch_threshold_positive": "Positive Pitch Threshold (deg)",
        "delay_drowsy_threshold": "Drowsiness Alert Delay (s)",
        "perclos_threshold": "PERCLOS Threshold",
        "perclos_window_size": "PERCLOS Window Size",
        "frame_height": "Frame Height (px)",
        "frame_width": "Frame Width (px)",          
        "frame_rate": "Frame Rate (FPS)",
    }

    HIDDEN_KEYS = {"min_face_detection_confidence", "min_face_presence_confidence", "num_faces"}

    def __init__(self, parent, config_mgr: ConfigManager):
        super().__init__(parent, padding=10, relief=tk.RIDGE)
        self.config_mgr = config_mgr
        self.entries = {}

        ttk.Label(self, text="Configuration", font=("Arial", 10, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(0, 10), sticky="w"
        )

        row_idx = 1
        keys = list(self.config_mgr.snapshot().keys())

        for key in keys:
            if key in self.HIDDEN_KEYS:
                continue

            ttk.Label(self, text=self.DISPLAY_LABELS.get(key, key)).grid(
                row=row_idx, column=0, sticky="w", padx=5, pady=2
            )
            entry = ttk.Entry(self, width=20)
            entry.insert(0, str(self.config_mgr.get(key)))
            entry.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=2)
            self.entries[key] = entry
            row_idx += 1

        self.columnconfigure(1, weight=1)

    def collect_changes(self):
        new_vals = {}
        for key, entry in self.entries.items():
            raw = entry.get()
            try:
                if '.' in raw:
                    val = float(raw)
                else:
                    val = int(raw)
            except ValueError:
                val = raw
            new_vals[key] = val
        return new_vals

# ==================== MAIN UI APP ====================

def main_ui():
    logger = create_log()
    logger.info("Start UI (hot reload + full restart on apply)")

    config_mgr = ConfigManager("configs/configs.yaml")
    led_pin = config_mgr.get("led_pin")
    gpio_enabled = initialize_gpio(led_pin, logger)

    # Shared resources (these will be re-created on full restart)
    stop_event = threading.Event()
    result_deque = deque(maxlen=3)
    result_lock = threading.Lock()

    # initial camera and processor setup
    cam = CameraManager(logger=logger, configs=config_mgr.snapshot())
    cam_opened = cam.open()
    if not cam_opened:
        logger.warning("The camera could not be opened after restart — a placeholder will be shown.")
        cam = None

    detector = None
    processor = None

    if cam is not None:
        callback = create_result_callback(result_deque, result_lock, stop_event, logger)
        detector = create_face_detector(config_mgr.get("model_path"), config_mgr.snapshot(), logger, callback)
        processor = MediaPipeProcessor(detector, result_deque, logger)
        processor.start(cam)

    # display thread
    display_thread = threading.Thread(
        target=display_and_process,
        args=(result_deque, result_lock, stop_event, config_mgr, logger, gpio_enabled, led_pin, ui_frame_queue),
        daemon=True,
    )
    display_thread.start()

    # BUILD UI
    root = tk.Tk()
    root.title("AI Drowsiness Detector")
    style = ttk.Style()
    try:
        style.theme_use('clam')
    except Exception:
        pass

    cam_w = config_mgr.get("frame_width", 1280)
    control_w = 360
    root.minsize(cam_w + control_w, config_mgr.get("frame_height", 720) + 20)

    main_frame = ttk.Frame(root, padding=8)
    main_frame.pack(fill="both", expand=True)
    main_frame.columnconfigure(0, weight=1)
    main_frame.columnconfigure(1, weight=0)
    main_frame.rowconfigure(0, weight=1)

    cam_view = CameraView(main_frame, config_mgr)
    cam_view.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

    control_panel = ttk.Frame(main_frame, padding=8)
    control_panel.grid(row=0, column=1, sticky="ns", padx=8, pady=8)

    editor = ConfigEditor(control_panel, config_mgr)
    editor.pack(fill="x", expand=False)

    btn_apply = ttk.Button(control_panel, text="Apply Configuration (Save & Restart)")
    btn_apply.pack(fill="x", pady=8)

    def restart_full():
        # runs in background thread to avoid blocking UI
        nonlocal stop_event, result_deque, result_lock, cam, detector, processor, display_thread
        btn_apply.config(state=tk.DISABLED)
        try:
            # collect changes from editor
            new_vals = editor.collect_changes()
            # update and save config
            config_mgr.set_many(new_vals)
            config_mgr.save()

            # stop current components
            stop_event.set()
            try:
                if processor:
                    processor.stop(timeout=2.0)
            except Exception:
                pass
            try:
                if cam:
                    cam.close(timeout=2.0)
            except Exception:
                pass
            # give threads a moment
            time.sleep(0.2)

            # create new shared resources
            stop_event = threading.Event()
            result_deque = deque(maxlen=3)
            result_lock = threading.Lock()

            # recreate camera
            cam = CameraManager(logger=logger, configs=config_mgr.snapshot())
            cam_opened = cam.open()
            if not cam_opened:
                logger.warning("The camera could not be opened after restart — a placeholder will be shown.")
                cam = None

            # recreate detector & processor
            detector = None
            processor = None
            if cam is not None:
                callback = create_result_callback(result_deque, result_lock, stop_event, logger)
                detector = create_face_detector(config_mgr.get("model_path"), config_mgr.snapshot(), logger, callback)
                processor = MediaPipeProcessor(detector, result_deque, logger)
                processor.start(cam)

            # restart display thread
            display_thread = threading.Thread(
                target=display_and_process,
                args=(result_deque, result_lock, stop_event, config_mgr, logger, gpio_enabled, led_pin, ui_frame_queue),
                daemon=True,
            )
            display_thread.start()
        finally:
            try:
                btn_apply.config(state=tk.NORMAL)
            except Exception:
                pass

    def on_apply():
        threading.Thread(target=restart_full, daemon=True).start()

    btn_apply.config(command=on_apply)

    def on_quit():
        logger.info("App close")
        stop_event.set()
        try:
            if processor:
                processor.stop(timeout=2.0)
        except Exception:
            pass
        config_mgr.save()
        cleanup_resources(cam=cam, detector=detector, led_pin=led_pin, gpio_enabled=gpio_enabled, logger=logger)
        root.destroy()

    ttk.Button(control_panel, text="Exit and Save Configuration", command=on_quit).pack(fill="x", pady=(6, 0))

    root.protocol("WM_DELETE_WINDOW", on_quit)
    root.mainloop()

if __name__ == "__main__":
    main_ui()