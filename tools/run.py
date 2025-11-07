"""
Drowsiness Detection System - OPTIMIZED FOR RASPBERRY PI 4
Giảm tải: resolution thấp, skip frame, UI đơn giản, MediaPipe sync
"""

import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
import threading
import queue
import time
import sys
import numpy as np
from typing import Optional
from collections import deque
from dataclasses import dataclass
from PIL import Image, ImageTk

# Import modules gốc
from src.threadflow import CameraManager, MediaPipeProcessor, FrameWaiter
from src.logger import create_log, save_suspected_frame
from src.utils import load_config, initialize_gpio, flash_led, cleanup_resources
from src.models import (
    create_face_detector,
    draw_face_landmarks,
    calculate_gaze_direction,
    draw_gaze_arrows,
    render_blendshape_metrics,
    display_eyes_status,
    display_info,
    get_head_orientation,
    display_head_orientation
)


# ==================== OPTIMIZATIONS ====================
# 1. REDUCE RESOLUTION - Instead of 1280x720, use 640x480 or 480x360
# 2. SKIP FRAMES - Process every 2-3 frames instead of every frame
# 3. LOWER FPS - Cap at 15-20 FPS instead of 30 FPS
# 4. SYNC DETECTION - Use sync instead of async (better for Pi)
# 5. SIMPLER UI - Only essential UI, no PIL resizing every frame


@dataclass
class DetectionResult:
    """Kết quả phát hiện từ MediaPipe"""
    frame_rgb: np.ndarray
    detection: any
    timestamp_ms: int
    frame_counter: int


# ==================== LIGHTWEIGHT UI ====================
class AlertPanel(tk.Frame):
    """Panel cảnh báo - đơn giản hóa"""
    
    def __init__(self, parent):
        super().__init__(parent, bg='#1a1a1a', height=50)
        self.pack_propagate(False)
        
        self.is_alert = False
        self.blink_state = False
        
        # Only essential label
        self.alert_label = tk.Label(
            self,
            text="● Ready",
            font=('Arial', 14, 'bold'),
            fg='#4CAF50',
            bg='#1a1a1a'
        )
        self.alert_label.pack(expand=True)
        
        self._start_animation()
    
    def set_alert(self, is_alert: bool):
        """Bật/tắt cảnh báo"""
        self.is_alert = is_alert
        if not is_alert:
            self.config(bg='#1a1a1a')
            self.alert_label.config(bg='#1a1a1a', fg='#4CAF50', text="● Normal")
        else:
            self.config(bg='#FF1744')
            self.alert_label.config(bg='#FF1744', fg='#FFEB3B', text="⚠️ DROWSY!")
    
    def _start_animation(self):
        """Blink animation khi cảnh báo"""
        if self.is_alert:
            self.blink_state = not self.blink_state
            color = '#FF5252' if self.blink_state else '#FF1744'
            self.config(bg=color)
            self.alert_label.config(bg=color)
        
        self.after(500, self._start_animation)  # 500ms blink (slower)


class SimpleInfoPanel(ttk.Frame):
    """Panel thông tin tối giản"""
    
    def __init__(self, parent):
        super().__init__(parent, padding="5")
        
        title = tk.Label(
            self,
            text="System Info",
            font=('Arial', 10, 'bold'),
            bg='#2196F3',
            fg='white',
            pady=5
        )
        title.pack(fill='x')
        
        stats_frame = tk.Frame(self, bg='white')
        stats_frame.pack(fill='both', expand=True, pady=(3, 0))
        
        # FPS label
        self.fps_label = tk.Label(
            stats_frame,
            text="FPS: 0.0",
            font=('Arial', 9),
            bg='white',
            fg='#333'
        )
        self.fps_label.pack(anchor='w', padx=5, pady=2)
        
        # PERCLOS label
        self.perclos_label = tk.Label(
            stats_frame,
            text="PERCLOS: 0%",
            font=('Arial', 9),
            bg='white',
            fg='#333'
        )
        self.perclos_label.pack(anchor='w', padx=5, pady=2)
        
        # Status label
        self.status_label = tk.Label(
            stats_frame,
            text="Face: No",
            font=('Arial', 9),
            bg='white',
            fg='#FF9800'
        )
        self.status_label.pack(anchor='w', padx=5, pady=2)
    
    def update_stats(self, fps: float, perclos: float, has_face: bool):
        """Cập nhật thông số"""
        self.fps_label.config(text=f"FPS: {fps:.1f}")
        self.perclos_label.config(text=f"PERCLOS: {perclos*100:.0f}%")
        
        face_text = "Face: Yes" if has_face else "Face: No"
        face_color = '#4CAF50' if has_face else '#FF9800'
        self.status_label.config(text=face_text, fg=face_color)


class SimpleCameraView(tk.Label):
    """Camera display - tối ưu cho Pi"""
    
    def __init__(self, parent, width=640, height=480):
        super().__init__(
            parent, 
            relief=tk.RIDGE, 
            borderwidth=1, 
            bg='#000000',
            width=width,
            height=height
        )
        self.target_width = width
        self.target_height = height
        self.imgtk = None
    
    def update_frame_direct(self, frame_rgb: np.ndarray):
        """Update frame từ BGR NumPy array - KHÔNG resize"""
        try:
            # If frame is not exact size, use OpenCV resize (faster)
            if frame_rgb.shape[:2] != (self.target_height, self.target_width):
                frame_rgb = cv2.resize(frame_rgb, (self.target_width, self.target_height), 
                                      interpolation=cv2.INTER_LINEAR)
            
            # Convert RGB to PhotoImage - nhanh hơn PIL
            img = Image.fromarray(frame_rgb)
            self.imgtk = ImageTk.PhotoImage(image=img)
            self.config(image=self.imgtk)
        except Exception as e:
            print(f"[CameraView] Error: {e}")


# ==================== OPTIMIZED PROCESSING ====================
class DrowsinessProcessorLite:
    """Xử lý buồn ngủ - OPTIMIZED cho Pi"""
    
    def __init__(self, configs: dict, logger, gpio_enabled: bool, led_pin: int):
        self.configs = configs
        self.logger = logger
        self.gpio_enabled = gpio_enabled
        self.led_pin = led_pin
        
        # State
        self.drowsy_prev = False
        self.delay_drowsy = None
        self.eye_closed_history = deque(maxlen=configs["perclos_window_size"])
        self.last_detection_result: Optional[DetectionResult] = None
        self.no_result_counter = 0
        self.max_no_result = 15  # Reduced from 30
        self.last_processed_counter = -1
        self.main_loop_fps_timestamps = deque(maxlen=20)  # Reduced from 30
        
        # SKIP FRAMES - process every Nth frame
        self.frame_skip = 2  # Process every 2nd frame on Pi4
        self.skip_counter = 0
        
        self.frame_waiter = FrameWaiter()
    
    def process_frame(self, cam, result_queue, stop_event) -> tuple:
        """Process frame with skipping"""
        try:
            # SKIP FRAMES optimization
            self.skip_counter += 1
            if self.skip_counter % self.frame_skip != 0:
                return None, False, 0.0, 0.0, False
            
            # Wait for frame
            self.frame_waiter.wait_for_new_frame(cam, timeout=0.01)
            
            # Get latest result (skip old ones)
            has_new_result = False
            temp_result = None
            
            try:
                temp_result = result_queue.get_nowait()
                has_new_result = True
            except queue.Empty:
                pass
            
            if has_new_result and temp_result:
                self.last_detection_result = temp_result
                self.no_result_counter = 0
            
            # Get current frame
            frame_info = cam.get_latest_frame_info()
            if frame_info is None:
                return None, False, 0.0, 0.0, False
            
            if frame_info.counter == self.last_processed_counter:
                return None, False, 0.0, 0.0, False
            
            self.last_processed_counter = frame_info.counter
            current_frame = frame_info.frame
            
            # Use result or convert current frame
            if self.last_detection_result is not None:
                frame_rgb = self.last_detection_result.frame_rgb
                detection_result = self.last_detection_result.detection
            else:
                frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                detection_result = None
                self.no_result_counter += 1
            
            is_alert = False
            perclos = 0.0
            has_face = False
            
            # Process detection
            if detection_result and detection_result.face_landmarks:
                has_face = True
                
                blendshapes = detection_result.face_blendshapes[0]
                face_landmarks = detection_result.face_landmarks[0]
                
                annotated_frame = draw_face_landmarks(frame_rgb, detection_result)
                gaze_info = calculate_gaze_direction(
                    face_landmarks, blendshapes, annotated_frame.shape
                )
                annotated_frame = draw_gaze_arrows(annotated_frame, gaze_info)
                annotated_frame = cv2.flip(annotated_frame, 1)
                
                # FPS
                main_fps = self._calculate_fps()
                annotated_frame = display_info(annotated_frame, main_fps)
                
                # Blink scores
                annotated_frame, blink_scores, text_end_y = render_blendshape_metrics(
                    annotated_frame, blendshapes
                )
                
                annotated_frame = display_eyes_status(
                    annotated_frame,
                    blink_scores["left"],
                    blink_scores["right"],
                    text_end_y,
                    self.configs["blink_threshold_wo_pitch"],
                )
                
                # Head orientation
                head_orientation = get_head_orientation(
                    detection_result.facial_transformation_matrixes[0]
                )
                annotated_frame = display_head_orientation(annotated_frame, head_orientation)
                
                # Drowsiness detection
                pitch = head_orientation["pitch"]
                blink_threshold = (
                    self.configs["blink_threshold_wo_pitch"] 
                    if self.configs["pitch_threshold_negative"] < pitch < self.configs["pitch_threshold_positive"]
                    else self.configs["blink_threshold_pitch"]
                )
                
                drowsy = (
                    blink_scores["left"] > blink_threshold and 
                    blink_scores["right"] > blink_threshold
                )
                
                # Timing
                if drowsy:
                    if not self.drowsy_prev:
                        self.delay_drowsy = time.time()
                    elapsed = time.time() - self.delay_drowsy
                    is_alert = elapsed > self.configs["delay_drowsy_threshold"]
                else:
                    self.delay_drowsy = None
                    is_alert = False
                
                self.drowsy_prev = drowsy
                
                # PERCLOS
                self.eye_closed_history.append(drowsy)
                perclos = sum(self.eye_closed_history) / len(self.eye_closed_history) if self.eye_closed_history else 0
                
                if perclos >= self.configs["perclos_threshold"]:
                    is_alert = True
                
                # Alert action
                if is_alert:
                    self.logger.warning("DROWSINESS DETECTED!")
                    save_suspected_frame(
                        origin_frame=display_info(cv2.flip(frame_rgb, 1), cam.get_fps()),
                        annotated_frame=annotated_frame,
                    )
                    flash_led(self.led_pin, self.gpio_enabled, self.logger)
                
            else:
                # No face
                annotated_frame = cv2.flip(frame_rgb, 1)
                main_fps = self._calculate_fps()
                annotated_frame = display_info(annotated_frame, main_fps)
            
            self.main_loop_fps_timestamps.append(time.time())
            
            return annotated_frame, is_alert, self._calculate_fps(), perclos, has_face
            
        except Exception as e:
            self.logger.error(f"Error in process_frame: {e}", exc_info=True)
            return None, False, 0.0, 0.0, False
    
    def _calculate_fps(self) -> float:
        """Calculate FPS"""
        if len(self.main_loop_fps_timestamps) >= 2:
            return len(self.main_loop_fps_timestamps) / max(
                self.main_loop_fps_timestamps[-1] - self.main_loop_fps_timestamps[0], 1e-6
            )
        return 0.0


# ==================== MAIN APP ====================
class DrowsinessDetectionAppLite:
    """Lightweight app cho Pi4"""
    
    def __init__(self):
        self.logger = None
        self.cam = None
        self.processor_thread = None
        self.detector = None
        self.configs = None
        self.gpio_enabled = False
        
        self.stop_event = threading.Event()
        self.result_queue = queue.Queue(maxsize=3)  # Reduced from 5
        
        self.root = None
        self.camera_view = None
        self.alert_panel = None
        self.info_panel = None
        self.drowsiness_processor = None
    
    def initialize(self):
        """Initialize system"""
        self.logger = create_log()
        self.logger.info("=" * 60)
        self.logger.info("DROWSINESS DETECTION - Pi4 OPTIMIZED")
        self.logger.info("=" * 60)
        
        self.configs = load_config()
        
        # OPTIMIZE: Reduce resolution for Pi4
        self.configs["frame_width"] = 640   # Changed from 1280
        self.configs["frame_height"] = 480  # Changed from 720
        self.configs["frame_rate"] = 30     # Changed from 30
        
        self.logger.info(f"Resolution: {self.configs['frame_width']}x{self.configs['frame_height']} @ {self.configs['frame_rate']} FPS")
        
        self.gpio_enabled = initialize_gpio(self.configs["led_pin"], self.logger)
        
        # Camera
        self.cam = CameraManager(logger=self.logger, configs=self.configs, buffer_size=2)
        if not self.cam.open():
            self.logger.critical("Camera failed")
            sys.exit(1)
        
        # MediaPipe - use sync callback
        callback = self._create_sync_callback()
        self.detector = create_face_detector(
            self.configs["model_path"], self.configs, self.logger, callback
        )
        
        # Processor
        self.processor_thread = MediaPipeProcessor(self.detector, self.result_queue, self.logger)
        self.processor_thread.start(self.cam)
        
        self.drowsiness_processor = DrowsinessProcessorLite(
            self.configs, self.logger, self.gpio_enabled, self.configs["led_pin"]
        )
    
    def _create_sync_callback(self):
        """Create result callback"""
        def callback(result, output_image, timestamp_ms: int):
            if self.stop_event.is_set():
                return
            try:
                frame_rgb = output_image.numpy_view().copy()
                try:
                    self.result_queue.put_nowait(
                        DetectionResult(
                            frame_rgb=frame_rgb,
                            detection=result,
                            timestamp_ms=timestamp_ms,
                            frame_counter=0
                        )
                    )
                except queue.Full:
                    pass
            except Exception as e:
                self.logger.error(f"Callback error: {e}")
        return callback
    
    def create_ui(self):
        """Create minimal UI"""
        self.root = tk.Tk()
        self.root.title("Drowsiness Detection - Pi4")
        self.root.configure(bg='#f0f0f0')
        
        # Window size - fit 640x480 + small info panel
        self.root.geometry("800x540")
        self.root.minsize(800, 540)
        
        main_container = tk.Frame(self.root, bg='#f0f0f0')
        main_container.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Alert
        self.alert_panel = AlertPanel(main_container)
        self.alert_panel.pack(fill="x", pady=(0, 5))
        
        # Content
        content_frame = tk.Frame(main_container, bg='#f0f0f0')
        content_frame.pack(fill="both", expand=True)
        
        # Camera (left)
        self.camera_view = SimpleCameraView(content_frame, width=640, height=480)
        self.camera_view.pack(side="left", padx=(0, 5))
        
        # Right panel - minimal
        right_panel = tk.Frame(content_frame, bg='white', width=150)
        right_panel.pack(side="right", fill="both", expand=True)
        right_panel.pack_propagate(False)
        
        # Info
        self.info_panel = SimpleInfoPanel(right_panel)
        self.info_panel.pack(fill="x", padx=5, pady=5)
        
        # Quit button
        quit_btn = tk.Button(
            right_panel,
            text="Quit",
            bg='#F44336',
            fg='white',
            font=('Arial', 10, 'bold'),
            command=self.quit_application,
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        quit_btn.pack(fill="x", padx=5, pady=5)
        
        self.root.protocol("WM_DELETE_WINDOW", self.quit_application)
    
    def update_ui_loop(self):
        """Update UI loop"""
        if self.stop_event.is_set():
            return
        
        try:
            result = self.drowsiness_processor.process_frame(
                self.cam, 
                self.result_queue,
                self.stop_event
            )
            
            if result[0] is not None:
                annotated_frame, is_alert, fps, perclos, has_face = result
                
                self.camera_view.update_frame_direct(annotated_frame)
                self.alert_panel.set_alert(is_alert)
                self.info_panel.update_stats(fps, perclos, has_face)
        
        except Exception as e:
            self.logger.error(f"UI loop error: {e}", exc_info=True)
        
        # Update less frequently on Pi4
        self.root.after(50, self.update_ui_loop)
    
    def quit_application(self):
        """Shutdown"""
        self.logger.info("Shutting down...")
        self.stop_event.set()
        
        if self.processor_thread:
            self.processor_thread.stop(timeout=1.0)
        
        if self.cam:
            self.cam.close(timeout=1.0)
        
        cleanup_resources(
            cam=None,
            detector=self.detector,
            led_pin=self.configs["led_pin"] if self.configs else 0,
            gpio_enabled=self.gpio_enabled,
            logger=self.logger,
        )
        
        self.logger.info("=" * 60)
        self.logger.info("TERMINATED")
        self.logger.info("=" * 60)
        
        self.root.destroy()
    
    def run(self):
        """Run app"""
        try:
            self.initialize()
            self.create_ui()
            self.root.after(100, self.update_ui_loop)
            self.root.mainloop()
        except KeyboardInterrupt:
            self.logger.info("Interrupted")
        except Exception as e:
            if self.logger:
                self.logger.critical(f"Fatal error: {e}", exc_info=True)
            sys.exit(1)


def main():
    app = DrowsinessDetectionAppLite()
    app.run()


if __name__ == "__main__":
    main()  