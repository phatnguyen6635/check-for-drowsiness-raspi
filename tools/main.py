import cv2
import mediapipe as mp

from src.camera import CameraManager
from src.logger import create_log
from src.utils import (load_config, initialize_gpio, set_led, cleanup_resources)
from src.models import (create_face_detector, draw_face_landmarks, render_blendshape_metrics,
                        calculate_gaze_direction, draw_gaze_arrows, display_drowsiness_alert,
                        display_time_info)

import sys
import time


def main():
    """Main function with comprehensive error handling"""
    logger = None
    cam = None
    detector = None
    gpio_enabled = False
    config = None
    
    # Shared state for async callback
    latest_result = {'detection': None, 'timestamp': 0}
    
    def result_callback(result, output_image, timestamp_ms: int):
        """Callback for async detection results"""
        latest_result['detection'] = result
        latest_result['timestamp'] = timestamp_ms
    
    try:
        # Initialize logger
        logger = create_log()
        logger.info("=" * 60)
        logger.info("DROWSINESS DETECTION SYSTEM STARTED")
        logger.info("=" * 60)
        
        # Load configuration
        config = load_config()
        model_path = config['model_path']
        blink_threshold = config['blink_threshold_wo_pitch']
        led_pin = config['led_pin']
        logger.info(f"Config loaded - Threshold: {blink_threshold} ; Led pin {led_pin}")
        
        # Initialize GPIO
        gpio_enabled = initialize_gpio(led_pin, logger)
        
        # Initialize camera
        cam = CameraManager(logger, width=1280, height=720, threaded=True, max_fps=0)
        
        if not cam.open():
            logger.error("Cannot open camera. Exiting.")
            return
        
        logger.info("Camera opened successfully")
        
        # Initialize face detector with callback
        detector = create_face_detector(model_path, config, logger, result_callback)
        
        # Initialize video recorder
        logger.info("Video recorder initialized")
        
        # Main loop
        logger.info("Starting detection loop. Press 'Q' or ESC to quit.")
        
        # State variables
        drowsy_pred = False
        worker_pred = True
        no_worker_count = 0
        frame_id = 0
        delay_worker = None
        delay_drowsy = None
        
        while True:
            try:
                # Read frame
                ret, frame = cam.read()
                if frame is None:
                    logger.warning("Failed to read frame, skipping...")
                    time.sleep(0.01)
                    continue
                
                # Crop frame
                frame = frame[90:640, 0:800]
                
                # Convert to RGB for mediapipe
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                frame_id += 1

                # Send frame for async detection
                detector.detect_async(mp_image, frame_id)
                
                # Get latest detection result from callback
                detection_result = latest_result['detection']
                print(detection_result)
                
                
                # Process detection results
                drowsy = False
                worker = False
                
                # Detect face
                if detection_result and detection_result.face_landmarks:
                    worker = True
                    blendshapes = detection_result.face_blendshapes[0]
                    
                    # Render metrics and get blink scores
                    blink_scores, text_end_y = render_blendshape_metrics(
                        rgb, blendshapes
                    )
                    
                    # Check drowsiness
                    drowsy = (blink_scores["left"] > blink_threshold and 
                             blink_scores["right"] > blink_threshold)
                    
                    # Display drowsiness alert on frame
                    rgb = display_drowsiness_alert(
                        rgb, 
                        blink_scores["left"], 
                        blink_scores["right"],
                        text_end_y,
                        blink_threshold
                    )
                    
                    # Reset no worker count
                    no_worker_count = 0
                
                # Handle no worker detection
                if not worker:
                    # Start timer when worker disappears
                    if worker_pred and not worker:
                        delay_worker = time.time()
                    
                    # Alert if no worker for > 1.5 seconds
                    if delay_worker and time.time() - delay_worker > 1.5:
                        if no_worker_count < 5:
                            logger.info("No worker detected in the frame.")
                            no_worker_count += 1
                
                # Handle drowsiness detection
                if drowsy:
                    # Start timer when drowsiness begins
                    if not drowsy_pred:
                        delay_drowsy = time.time()
                    
                    # Alert if drowsy for > 2 seconds
                    if delay_drowsy and time.time() - delay_drowsy > 2:
                        logger.info("Worker is drowsy - ALERT!")
                        set_led(led_pin, True, gpio_enabled, logger)
                        cam.flush(3)
                    else:
                        set_led(led_pin, False, gpio_enabled, logger)
                else:
                    # Turn off LED when not drowsy
                    set_led(led_pin, False, gpio_enabled, logger)
                    delay_drowsy = None  # Reset drowsy timer
                
                # Update previous states
                drowsy_pred = drowsy
                worker_pred = worker
                
                # Draw face landmarks (only if we have results)
                if detection_result.face_landmarks:
                    annotated_frame = draw_face_landmarks(rgb, detection_result)
                else:
                    annotated_frame = rgb
                
                # Convert back to BGR for display
                display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                
                # Display frame
                cv2.imshow("Drowsiness Detection", display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(5) & 0xFF
                if key in [ord('q'), ord('Q'), 27]:
                    logger.info("User requested quit")
                    break
                
                time.sleep(0.1)
                
            except cv2.error as e:
                logger.error(f"OpenCV error in loop: {e}")
                continue
            
            except Exception as e:
                logger.error(f"Error in processing loop: {e}", exc_info=True)
                continue
    
    except KeyboardInterrupt:
        if logger:
            logger.info("Interrupted by user (Ctrl+C)")
    
    except FileNotFoundError as e:
        if logger:
            logger.error(f"File not found: {e}")
        else:
            print(f"ERROR: {e}")
        sys.exit(1)
    
    except Exception as e:
        if logger:
            logger.exception(f"Fatal error: {e}")
        else:
            print(f"FATAL ERROR: {e}")
        sys.exit(1)
    
    finally:
        if cam and detector:
            cleanup_resources(cam, detector, 
                            config['led_pin'] if config else 0, 
                            gpio_enabled, logger or create_log())
        if logger:
            logger.info("=" * 60)
            logger.info("PROGRAM TERMINATED")
            logger.info("=" * 60)


if __name__ == "__main__":
    main()