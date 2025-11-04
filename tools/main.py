import cv2
import mediapipe as mp
import threading
import queue
import time
import sys
from typing import Optional

from src.logger import create_log
from src.camera import CameraManager
from src.utils import load_config, initialize_gpio, set_led, cleanup_resources
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


# ==================== GLOBAL STATE ====================
result_queue: queue.Queue = queue.Queue(maxsize=5)
stop_event = threading.Event()
process_thread: Optional[threading.Thread] = None


# ==================== MEDIAPIPE CALLBACK ====================
def result_callback(result, output_image, timestamp_ms: int) -> None:
    """Async callback from MediaPipe → push result to queue."""
    if stop_event.is_set():
        return
    try:
        frame_rgb = output_image.numpy_view().copy()
        result_queue.put_nowait((frame_rgb, result, timestamp_ms))
    except queue.Full:
        pass  # Drop old frame → prioritize real-time


# ==================== PROCESS THREAD: MEDIAPIPE INFERENCE ====================
def process_frames(detector, cam: CameraManager, logger) -> None:
    """Continuously feed latest frame from CameraManager to MediaPipe."""
    logger.info("MediaPipe processing thread started.")
    
    # CRITICAL: Use monotonic clock for timestamps
    timestamp_ms = 0  # Start from 0
    frame_interval_ms = 33  # ~30 FPS interval (1000ms / 30fps)
    last_frame_counter = -1  # Track if we got a new frame
    
    while not stop_event.is_set():
        try:
            ret, frame_bgr = cam.read()
            if not ret or frame_bgr is None:
                time.sleep(0.001)
                continue

            # Skip if same frame (avoid duplicate timestamps)
            current_counter = cam._frame_counter
            if current_counter == last_frame_counter:
                time.sleep(0.001)
                continue
            last_frame_counter = current_counter

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # FIXED: Always increment timestamp monotonically
            timestamp_ms += frame_interval_ms
            
            # Async detection → result via callback
            detector.detect_async(mp_image, timestamp_ms)

        except ValueError as e:
            # Catch timestamp errors specifically
            if "monotonically increasing" in str(e):
                logger.error(f"Timestamp error at {timestamp_ms}ms. Resetting detector...")
                # Force timestamp forward
                timestamp_ms += frame_interval_ms * 2
            else:
                logger.error(f"ValueError in MediaPipe thread: {e}", exc_info=True)
            time.sleep(0.01)
        except Exception as e:
            logger.error(f"Error in MediaPipe thread: {e}", exc_info=True)
            time.sleep(0.01)

    logger.info("MediaPipe processing thread stopped.")


# ==================== MAIN THREAD: DISPLAY + LOGIC ====================
def display_and_process(
    cam: CameraManager,
    detector,
    configs: dict,
    logger,
    gpio_enabled: bool,
    led_pin: int,
) -> None:
    """Main loop: display frame, process logic, handle alerts."""
    logger.info("Display & logic loop started.")

    # State tracking
    delay_worker: Optional[float] = None
    delay_drowsy: Optional[float] = None
    blink_threshold_wo_pitch = configs["blink_threshold_wo_pitch"]
    blink_threshold_pitch = configs["blink_threshold_pitch"]
    
    processed_frames = 0

    while not stop_event.is_set():
        try:
            # Get latest detection result
            try:
                frame_rgb, detection_result, _ = result_queue.get_nowait()
            except queue.Empty:
                time.sleep(0.001)
                continue

            processed_frames += 1
            
            # === DRAW LANDMARKS ===
            if detection_result and detection_result.face_landmarks:
                annotated_frame = draw_face_landmarks(frame_rgb, detection_result)
            else:
                annotated_frame = frame_rgb
                
            # === FACE DETECTION & BLINK ANALYSIS ===
            if detection_result and detection_result.face_landmarks:
                blendshapes = detection_result.face_blendshapes[0]
                face_landmarks = detection_result.face_landmarks[0]
                
                annotated_frame = draw_face_landmarks(frame_rgb, detection_result)
                gaze_info = calculate_gaze_direction(
                    face_landmarks, blendshapes, annotated_frame.shape
                )
                annotated_frame = draw_gaze_arrows(annotated_frame, gaze_info)
                annotated_frame = cv2.flip(annotated_frame, 1)
                annotated_frame = display_info(annotated_frame, cam.get_fps())

                
                annotated_frame, blink_scores, text_end_y  = render_blendshape_metrics(annotated_frame, blendshapes)

                annotated_frame =display_eyes_status(
                    annotated_frame,
                    blink_scores["left"],
                    blink_scores["right"],
                    text_end_y,
                    blink_threshold_wo_pitch,
                )
                if detection_result.facial_transformation_matrixes:
                    head_orientation = get_head_orientation(detection_result.facial_transformation_matrixes[0])
                    annotated_frame = display_head_orientation(annotated_frame, head_orientation)
                    
                    
            
            else:
                annotated_frame = frame_rgb
                annotated_frame = cv2.flip(annotated_frame, 1)
                annotated_frame = display_info(annotated_frame, cam.get_fps())

            # # === DROWSINESS ALERT LOGIC ===
            # Logic 1: Perclos
            # Logic 2: EAR
            # Logic 3: Pitch + EAR
            # if drowsy:
            #     if not drowsy_prev:
            #         delay_drowsy = time.time()
            #     if delay_drowsy and time.time() - delay_drowsy > 2.0:
            #         logger.warning("WORKER IS DROWSY - ALERT!")
            #         set_led(led_pin, True, gpio_enabled, logger)
            #         cam.flush(3)  # Clear camera buffer
            #     else:
            #         set_led(led_pin, False, gpio_enabled, logger)
            # else:
            #     set_led(led_pin, False, gpio_enabled, logger)
            #     delay_drowsy = None

            # === DISPLAY FRAME ===
            display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Drowsiness Detection", display_frame)

            # === KEYBOARD INPUT ===
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:  # q, Q, ESC
                logger.info("Quit requested by user.")
                stop_event.set()
                

        except Exception as e:
            logger.error(f"Error in display loop: {e}", exc_info=True)
            time.sleep(0.01)

    cv2.destroyAllWindows()
    logger.info("Display loop terminated.")


# ==================== MAIN ENTRY POINT ====================
def main() -> None:
    global process_thread

    logger = None
    cam: Optional[CameraManager] = None
    detector = None
    gpio_enabled = False
    config = None

    try:
        # === INITIALIZE LOGGER ===
        logger = create_log()
        logger.info("=" * 60)
        logger.info("DROWSINESS DETECTION SYSTEM STARTED")
        logger.info("=" * 60)

        # === LOAD CONFIG ===
        configs = load_config()
        model_path = configs["model_path"]
        led_pin = configs["led_pin"]
        logger.info(f"Config loaded: LED={led_pin}")

        # === INITIALIZE GPIO ===
        gpio_enabled = initialize_gpio(led_pin, logger)

        # === INITIALIZE CAMERA (with internal thread & FPS limit) ===
        cam = CameraManager(
            logger=logger,
            configs=configs,
            threaded=True,
            buffer_size=1,
        )
        if not cam.open():
            logger.critical("Failed to open camera. Exiting.")
            sys.exit(1)

        # === INITIALIZE MEDIAPIPE DETECTOR ===
        detector = create_face_detector(model_path, configs, logger, result_callback)

        # === START MEDIAPIPE PROCESSING THREAD ===
        process_thread = threading.Thread(
            target=process_frames,
            args=(detector, cam, logger),
            daemon=True,
            name="MediaPipeProcessor",
        )
        process_thread.start()
        logger.info("MediaPipe processing thread launched.")

        # === RUN DISPLAY & LOGIC IN MAIN THREAD ===
        display_and_process(cam, detector, configs, logger, gpio_enabled, led_pin)

    except KeyboardInterrupt:
        logger.info("System interrupted by user (Ctrl+C).")
    except Exception as e:
        logger.critical(f"Fatal error in main: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # === GRACEFUL SHUTDOWN ===
        logger.info("Initiating shutdown sequence...")
        stop_event.set()

        if process_thread and process_thread.is_alive():
            process_thread.join(timeout=2.0)
            if process_thread.is_alive():
                logger.warning("MediaPipe thread did not terminate gracefully.")

        cleanup_resources(
            cam=cam,
            detector=detector,
            led_pin=config["led_pin"] if config else 0,
            gpio_enabled=gpio_enabled,
            logger=logger or create_log(),
        )

        logger.info("=" * 60)
        logger.info("SYSTEM TERMINATED")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()