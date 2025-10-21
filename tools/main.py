import cv2
import mediapipe as mp
from src.utils import (load_config, create_log, create_face_detector, draw_face_landmarks,
                       render_blendshape_metrics, display_drowsiness_alert)
from src.utils import CameraManager

import os
import sys
import time

try:
    import RPi.GPIO as GPIO
    RPI_AVAILABLE = True
except ImportError:
    GPIO = None
    RPI_AVAILABLE = False

def initialize_gpio(led_pin, logger):
    """Initialize GPIO with error handling"""
    if not RPI_AVAILABLE:
        logger.warning("RPI.GPIO not available. GPIO features disabled.")
        return False
    
    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(led_pin, GPIO.OUT)
        GPIO.output(led_pin, GPIO.LOW)
        logger.info(f"GPIO initialized on pin {led_pin}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to initialize GPIO: {e}")
        return False

def set_led(led_pin, state, gpio_enabled, logger):
    """Control led with error handling"""
    if not gpio_enabled:
        return
    try:
        GPIO.output(led_pin, GPIO.HIGH if state else GPIO.LOW)
    except Exception as e:
        logger.error(f"GPIO output error: {e}")
        
def cleanup_resources(cam, detector, led_pin, gpio_enabled, logger):
    """Cleanup all resources safely"""
    logger.info("Cleaning up resources...")
    
    try:
        cam.close()
    except Exception as e:
        logger.error(f"Error closing camera: {e}")
    
    try:
        cv2.destroyAllWindows()
    except Exception as e:
        logger.error(f"Error destroying windows: {e}")
    
    try:
        detector.close()
    except Exception as e:
        logger.error(f"Error closing detector: {e}")
    
    if gpio_enabled:
        try:
            GPIO.output(led_pin, GPIO.LOW)
            GPIO.cleanup()
            logger.info("GPIO cleaned up")
        except Exception as e:
            logger.error(f"GPIO cleanup error: {e}")

def main():
    """Main function with comprehensive error handling"""
    logger = None
    cam = None
    detector = None
    gpio_enabled = False
    config = None
    
    try:
        # Initialize logger
        logger = create_log()
        logger.info("=" * 60)
        logger.info("DROWSINESS DETECTION SYSTEM STARTED")
        logger.info("=" * 60)
        
        # Load configuration
        config = load_config()
        model_path = config['model_path']
        blink_threshold = config['blink_threshold']
        led_pin = config['led_pin']
        logger.info(f"Config loaded - Threshold: {blink_threshold} ; Led pin {led_pin}")
        
        
        # Initialize camera
        cam = CameraManager(logger, camera_index=0, width=1280, height=720, threaded=True, max_fps=0)
        
        if not cam.open():
            logger.error("Cannot open camera. Exiting.")
            return
        
        logger.info("Camera opened successfully")
        
        # Initialize face detector
        detector = create_face_detector(model_path, logger)
        
        # Main loop
        logger.info("Starting detection loop. Press 'Q' or ESC to quit.")
        
        drowsy_pred = False
        worker_pred = True
        count = 0
        count = 0
        
        while True:
            try:
                # Read frame
                ret, frame = cam.read()
                if frame is None:
                    logger.warning("Failed to read frame, skipping...")
                    time.sleep(0.01)
                    continue
                                
                # Using mediapipe for face detection
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                detection_result = detector.detect(mp_image)
                
                # Process detection results
                drowsy = False
                
                # Detect face
                if detection_result.face_landmarks:
                    blendshapes = detection_result.face_blendshapes[0]
                    
                    # Render metrics and get blink scores
                    blink_scores, text_end_y = render_blendshape_metrics(
                        rgb, blendshapes
                    )
                    
                    # Display drowsiness alerts
                    drowsy = display_drowsiness_alert(
                        rgb, 
                        blink_scores["left"], 
                        blink_scores["right"],
                        text_end_y,
                        blink_threshold
                    )
                    count = 0
                    worker = True
                    
                else:
                    worker = False

                    if worker_pred and not worker:
                        delay_worker = time.time()
                    if count < 5 and time.time() - delay_worker > 1.5:
                        set_led(led_pin, True, gpio_enabled, logger)
                        logger.info("No woker detected in the frame.")
                        # os.system("ffplay -nodisp -autoexit -volume 100 /home/pi/Documents/project/check-for-drowsiness-raspi/voice/voice1.wav")
                        time.sleep(2)
                        set_led(led_pin, False, gpio_enabled, logger)
                        cam.flush(3)
                        count += 1
                        
                if drowsy and not drowsy_pred:
                    delay_drowsy = time.time()
                if drowsy and time.time() - delay_drowsy > 2:
                    logger.info("Worker is sleeping")
                    set_led(led_pin, True, gpio_enabled, logger)
                    os.system("ffplay -nodisp -autoexit /home/pi/Documents/project/check-for-drowsiness-raspi/voice/voice2.wav")
                    set_led(led_pin, False, gpio_enabled, logger)
                    cam.flush(3)
                    
                drowsy_pred = drowsy
                worker_pred = worker
                
                annotated_frame = draw_face_landmarks(rgb, detection_result)
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
