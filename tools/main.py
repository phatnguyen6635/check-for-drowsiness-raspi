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

from src.threadflow import CameraManager, MediaPipeProcessor
from src.logger import create_log, save_suspected_frame
from src.utils import load_config, initialize_gpio, set_relay, cleanup_resources
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
def create_result_callback(result_deque: deque, result_lock: threading.Lock, stop_event: threading.Event, logger):
    """Factory function to create callback with proper closure"""
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
                        frame_counter=0  # Will be updated by processor if needed
                    )
                )
                time.sleep(0.001)

        except Exception as e:
            logger.error(f"Error in result callback: {e}")

    return result_callback

# ==================== MAIN DISPLAY & LOGIC (deque-based) ====================
def display_and_process(
    result_deque: deque,
    result_lock: threading.Lock,
    stop_event: threading.Event,
    configs: dict,
    logger,
    gpio_enabled: bool,
    led_pin: int,
) -> None:
    """Main display loop that only reads latest result from result_deque. Main controls total FPS."""
    logger.info("Display & logic loop (deque-based) started.")

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
    last_displayed_frame = None

    # Main controls display fps
    main_target_fps = configs.get("display_fps", configs.get("frame_rate", 15))
    interval = 1.0 / max(1.0, float(main_target_fps))
    relay_on = False
    main_loop_fps_timestamps = deque(maxlen=30)

    while not stop_event.is_set():
        loop_start = time.time()

        # Grab latest result (non-blocking)
        with result_lock:
            if result_deque:
                last_detection_result = result_deque[-1]
            else:
                last_detection_result = None

        is_alert = False
        if last_detection_result is not None:
            # Unpack
            frame_rgb = last_detection_result.frame_rgb
            detection_result = last_detection_result.detection

            if detection_result and getattr(detection_result, 'face_landmarks', None):
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
                if eye_closed_history and len(eye_closed_history) >= perclos_window_size:
                    perclos = sum(eye_closed_history) / len(eye_closed_history)
                else:
                    perclos = 0

                if perclos >= perclos_threshold:
                    is_alert = True

                # Alert
                time.sleep(0.001)
                if is_alert:
                    logger.warning("DROWSINESS DETECTED!")
                    save_suspected_frame(
                        origin_frame=display_info(cv2.flip(frame_rgb, 1), main_fps),
                        annotated_frame=annotated_frame,
                    )
                    eye_closed_history.clear()

                display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                last_displayed_frame = display_frame

            else:
                # No face detected
                annotated_frame = cv2.flip(frame_rgb, 1)
                main_fps = len(main_loop_fps_timestamps) / max(
                    main_loop_fps_timestamps[-1] - main_loop_fps_timestamps[0], 1e-6
                ) if len(main_loop_fps_timestamps) >= 2 else 0.0

                annotated_frame = display_info(annotated_frame, main_fps)
                display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                last_displayed_frame = display_frame

        else:
            # No detection result available: re-show last frame or display placeholder
            if last_displayed_frame is None:
                # black frame placeholder
                placeholder = np.zeros((configs["frame_height"], configs["frame_width"], 3), dtype=np.uint8)
                placeholder = display_info(placeholder, 0)
                display_frame = placeholder
            else:
                display_frame = last_displayed_frame

        if is_alert and not relay_on:
            set_relay(gpio_enabled, led_pin, logger, True)
            relay_on = True
        elif not is_alert and relay_on:
            set_relay(gpio_enabled, led_pin, logger, False)
            relay_on = False

        # Show on screen
        # cv2.imshow("Drowsiness Detection", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), ord('Q'), 27]:
            logger.info("User requested quit.")
            stop_event.set()

        main_loop_fps_timestamps.append(time.time())

        # sleep to enforce target fps
        elapsed = time.time() - loop_start
        to_sleep = interval - elapsed
        if to_sleep > 0:
            time.sleep(to_sleep)

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
    result_deque = deque(maxlen=3)
    result_lock = threading.Lock()  # Added for thread safety

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
        callback = create_result_callback(result_deque, result_lock, stop_event, logger)
        detector = create_face_detector(
            configs["model_path"], configs, logger, callback
        )

        # Processor
        processor = MediaPipeProcessor(detector, result_deque, logger)
        processor.start(cam)

        # Main loop
        display_and_process(
            result_deque, result_lock, stop_event, configs, logger,
            gpio_enabled, configs["led_pin"]
        )

    except KeyboardInterrupt:
        if logger:
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

        cleanup_resources(
            cam=cam,
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