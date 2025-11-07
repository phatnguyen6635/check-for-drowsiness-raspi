import cv2
from pathlib import Path
import yaml
from yaml import Loader
from pathlib import Path
import time
import threading

try:
    import RPi.GPIO as GPIO
    RPI_AVAILABLE = True
except ModuleNotFoundError:
    GPIO = None
    RPI_AVAILABLE = False

def load_config(config_path='configs/configs.yaml'):
    """Load configuration from YAML file with error handling"""
    try:
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path) as stream:
            config = yaml.load(stream, Loader=Loader)
        
        required_keys = ['model_path', 'num_faces', 'min_face_detection_confidence',
                         'min_face_presence_confidence', 'blink_threshold_pitch',
                         'blink_threshold_wo_pitch', 'camera_id', 'frame_width',
                         'frame_height', 'frame_rate', 'led_pin']
        
        for key in required_keys:
            if key not in config:
                raise KeyError(f"Missing required config key: {key}")
        
        return config
    
    except Exception as e:
        raise RuntimeError(f"Failed to load config: {e}")

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

def flash_led(led_pin, gpio_enabled, logger, duration=2):
    """Flash LED for a duration (non-blocking)."""
    if not gpio_enabled:
        return
    def _blink():
        try:
            GPIO.output(led_pin, GPIO.HIGH)
            time.sleep(duration)
            GPIO.output(led_pin, GPIO.LOW)
        except Exception as e:
            logger.error(f"GPIO blink error: {e}")
    threading.Thread(target=_blink, daemon=True).start()
        
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
