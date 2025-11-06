import cv2
import mediapipe as mp

import threading
import queue
import time
import sys
import os
import numpy as np
from typing import Optional, Tuple
from collections import deque
from dataclasses import dataclass

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


# ==================== ENUMS & DATA CLASSES ====================
@dataclass
class DetectionResult:
    """Thread-safe detection result wrapper"""
    frame_rgb: np.ndarray
    detection: any
    timestamp_ms: int
    frame_counter: int


# ==================== RESULT CALLBACK ====================
def create_result_callback(result_queue: queue.Queue, stop_event: threading.Event, logger):
    """Factory function to create callback with proper closure"""
    def result_callback(result, output_image, timestamp_ms: int) -> None:
        if stop_event.is_set():
            return
        
        try:
            frame_rgb = output_image.numpy_view().copy()
            
            # Non-blocking put with timeout
            try:
                result_queue.put(
                    DetectionResult(
                        frame_rgb=frame_rgb,
                        detection=result,
                        timestamp_ms=timestamp_ms,
                        frame_counter=0  # Will be updated by processor
                    ),
                    timeout=0.001
                )
            except queue.Full:
                pass  # Drop frame if queue full
                
        except Exception as e:
            logger.error(f"Error in result callback: {e}")
    
    return result_callback


# ==================== MAIN DISPLAY & LOGIC ====================
def display_and_process(
    cam: CameraManager,
    result_queue: queue.Queue,
    stop_event: threading.Event,
    configs: dict,
    logger,
    gpio_enabled: bool,
    led_pin: int,
) -> None:
    """Main display loop with proper synchronization"""
    logger.info("Display & logic loop started.")

    # Config
    blink_threshold_wo_pitch = configs["blink_threshold_wo_pitch"]
    blink_threshold_pitch = configs["blink_threshold_pitch"]
    pitch_threshold_positive = configs["pitch_threshold_positive"]
    pitch_threshold_negative = configs["pitch_threshold_negative"]
    delay_drowsy_threshold = configs["delay_drowsy_threshold"]
    perclos_window_size = configs["perclos_window_size"]
    perclos_threshold = configs["perclos_threshold"]

    # State
    drowsy_prev = False
    delay_drowsy = None
    eye_closed_history = deque(maxlen=perclos_window_size)
    last_detection_result: Optional[DetectionResult] = None
    no_result_counter = 0
    max_no_result = 30
    
    # Frame synchronization
    last_processed_counter = -1
    frame_waiter = FrameWaiter()
    
    # FPS calculation for main loop
    main_loop_fps_timestamps = deque(maxlen=30)

    while not stop_event.is_set():
        try:
            loop_start = time.time()
            
            # ===== WAIT FOR NEW CAMERA FRAME (with short timeout) =====
            # Wait for new frame from camera
            frame_waiter.wait_for_new_frame(cam, timeout=0.01)  # 10ms timeout
            
            # Drain queue to get latest result
            has_new_result = False
            temp_result = None
            
            while True:
                try:
                    temp_result = result_queue.get_nowait()
                    has_new_result = True
                except queue.Empty:
                    break
            
            if has_new_result and temp_result:
                last_detection_result = temp_result
                no_result_counter = 0

            # Get current camera frame
            frame_info = cam.get_latest_frame_info()
            
            if frame_info is None:
                time.sleep(0.001)
                continue
            
            # Skip if no new frame
            if frame_info.counter == last_processed_counter:
                time.sleep(0.001)
                continue
            
            last_processed_counter = frame_info.counter
            current_frame = frame_info.frame

            # Use detection result if available
            if last_detection_result is not None:
                frame_rgb = last_detection_result.frame_rgb
                detection_result = last_detection_result.detection
            else:
                frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                detection_result = None
                no_result_counter += 1

            # Check for stale results
            if no_result_counter > max_no_result:
                logger.warning("No detection results for too long.")

            # Process detection
            if detection_result and detection_result.face_landmarks:
                # === FACE ANALYSIS ===
                blendshapes = detection_result.face_blendshapes[0]
                face_landmarks = detection_result.face_landmarks[0]
                
                annotated_frame = draw_face_landmarks(frame_rgb, detection_result)
                gaze_info = calculate_gaze_direction(
                    face_landmarks, blendshapes, annotated_frame.shape
                )
                annotated_frame = draw_gaze_arrows(annotated_frame, gaze_info)
                annotated_frame = cv2.flip(annotated_frame, 1)
                
                main_fps = len(main_loop_fps_timestamps) / max(
                    main_loop_fps_timestamps[-1] - main_loop_fps_timestamps[0], 1e-6
                ) if len(main_loop_fps_timestamps) >= 2 else 0.0
                
                annotated_frame = display_info(annotated_frame, main_fps)

                annotated_frame, blink_scores, text_end_y = render_blendshape_metrics(
                    annotated_frame, blendshapes
                )

                annotated_frame = display_eyes_status(
                    annotated_frame,
                    blink_scores["left"],
                    blink_scores["right"],
                    text_end_y,
                    blink_threshold_wo_pitch,
                )
                
                head_orientation = get_head_orientation(
                    detection_result.facial_transformation_matrixes[0]
                )
                annotated_frame = display_head_orientation(annotated_frame, head_orientation)
                
                # === DROWSINESS DETECTION ===
                pitch = head_orientation["pitch"]
                blink_threshold = (
                    blink_threshold_wo_pitch 
                    if pitch_threshold_negative < pitch < pitch_threshold_positive 
                    else blink_threshold_pitch
                )
                
                drowsy = (
                    blink_scores["left"] > blink_threshold and 
                    blink_scores["right"] > blink_threshold
                )
                
                # Timing logic
                if drowsy:
                    if not drowsy_prev:
                        delay_drowsy = time.time()
                    elapsed = time.time() - delay_drowsy
                    is_alert = elapsed > delay_drowsy_threshold
                else:
                    delay_drowsy = None
                    is_alert = False
                
                drowsy_prev = drowsy

                # PERCLOS
                eye_closed_history.append(drowsy)
                perclos = sum(eye_closed_history) / len(eye_closed_history) if eye_closed_history else 0
                print(f"PERCLOS: {perclos:.2f}")
                
                if perclos >= perclos_threshold:
                    is_alert = True
                    
                # Alert
                if is_alert:
                    logger.warning("DROWSINESS DETECTED!")
                    save_suspected_frame(
                        origin_frame=display_info(cv2.flip(frame_rgb, 1), cam.get_fps()),
                        annotated_frame=annotated_frame,
                    )
                    flash_led(led_pin, gpio_enabled, logger)
                
            else:
                # No face detected
                annotated_frame = cv2.flip(frame_rgb, 1)
                cam_fps = cam.get_fps()
                main_fps = len(main_loop_fps_timestamps) / max(
                    main_loop_fps_timestamps[-1] - main_loop_fps_timestamps[0], 1e-6
                ) if len(main_loop_fps_timestamps) >= 2 else 0.0
                
                annotated_frame = display_info(annotated_frame, cam_fps)
            
            # === DISPLAY ===
            display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Drowsiness Detection", display_frame)

            # === KEYBOARD ===
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                logger.info("User requested quit.")
                stop_event.set()
            
            # Track main loop FPS
            main_loop_fps_timestamps.append(time.time())

        except Exception as e:
            logger.error(f"Error in display loop: {e}", exc_info=True)
            time.sleep(0.01)

    cv2.destroyAllWindows()
    logger.info("Display loop terminated.")


# ==================== MAIN ====================
def main() -> None:
    logger = None
    cam: Optional[CameraManager] = None
    processor: Optional[MediaPipeProcessor] = None
    detector = None
    gpio_enabled = False
    configs = None

    # Shared resources
    stop_event = threading.Event()
    result_queue = queue.Queue(maxsize=5)

    try:
        # Initialize
        logger = create_log()
        logger.info("=" * 60)
        logger.info("DROWSINESS DETECTION SYSTEM STARTED")
        logger.info("=" * 60)

        configs = load_config()
        gpio_enabled = initialize_gpio(configs["led_pin"], logger)

        # Camera
        cam = CameraManager(logger=logger, configs=configs)
        if not cam.open():
            logger.critical("Failed to open camera.")
            sys.exit(1)

        # MediaPipe
        callback = create_result_callback(result_queue, stop_event, logger)
        detector = create_face_detector(
            configs["model_path"], configs, logger, callback
        )

        # Processor
        processor = MediaPipeProcessor(detector, result_queue, logger)
        processor.start(cam)

        # Main loop
        display_and_process(
            cam, result_queue, stop_event, configs, logger,
            gpio_enabled, configs["led_pin"]
        )

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        if logger:
            logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Cleanup
        if logger:
            logger.info("Shutting down...")
        
        stop_event.set()

        if processor:
            processor.stop(timeout=2.0)

        if cam:
            cam.close(timeout=2.0)

        cleanup_resources(
            cam=None,
            detector=detector,
            led_pin=configs["led_pin"] if configs else 0,
            gpio_enabled=gpio_enabled,
            logger=logger or create_log(),
        )

        if logger:
            logger.info("=" * 60)
            logger.info("SYSTEM TERMINATED")
            logger.info("=" * 60)


if __name__ == "__main__":
    main()