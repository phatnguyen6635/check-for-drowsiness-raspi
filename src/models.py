import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions.drawing_utils import DrawingSpec

import numpy as np
from pathlib import Path
from datetime import datetime

EYE_BLENDSHAPE_CATEGORIES = [
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "eyeLookUpLeft", "eyeLookUpRight",
    "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight",
]

# Display setting
FONT_FACE = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE_INFO = 0.5
FONT_SCALE_ALERT = 0.8
COLOR_INFO = (0, 0, 139)      # Dark blue
COLOR_ALERT = (255, 69, 0)      # Red-orange
COLOR_GAZE_LEFT = (255, 100, 255) # Pink
COLOR_GAZE_RIGHT = (255, 100, 255)
FONT_THICKNESS_INFO = 1
FONT_THICKNESS_ALERT = 2
TEXT_LINE_HEIGHT = 22
TEXT_MARGIN = 15

# Gaze arrow settings
ARROW_LENGTH = 60
ARROW_THICKNESS = 3
ARROW_TIP_LENGTH = 0.3

# Iris landmark indices in MediaPipe Face Mesh
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473
LEFT_EYE_CENTER = 33
RIGHT_EYE_CENTER = 263

def create_face_detector(model_path, configs, logger, result_callback):
    """Create MediaPipe face detector with error handling"""
    try:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            running_mode = vision.RunningMode.LIVE_STREAM,
            result_callback=result_callback,
            num_faces=configs['num_faces'],
            min_face_detection_confidence=configs['min_face_detection_confidence'],
            min_face_presence_confidence=configs['min_face_presence_confidence']
        )
        detector = vision.FaceLandmarker.create_from_options(options)
        logger.info(f"Face detector initialized with model: {model_path}")
        return detector
    
    except Exception as e:
        logger.error(f"Failed to create face detector: {e}")
        raise

def draw_face_landmarks(rgb_image, detection_result):
    """
    Draw facial landmarks and mesh on the input image.
    
    Args:
        rgb_image: Input image in RGB format
        detection_result: MediaPipe face detection result
        
    Returns:
        Annotated image with facial landmarks drawn
    """
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Convert landmarks to protobuf format
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) 
            for lm in face_landmarks
        ])

        # Convert landmarks to protobuf format
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) 
            for lm in face_landmarks
        ])

        # Draw face mesh tesselation (toàn bộ khuôn mặt)
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=DrawingSpec(color=(175,238,238), thickness=1)
        )
        
        # Draw face contours (viền mặt rõ hơn)
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=DrawingSpec(color=(135,206,235), thickness=2)
        )
        
        # Draw left iris
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_LEFT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=DrawingSpec(color=(244,164,96), thickness=2)
        )
        
        # Draw right iris
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_RIGHT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=DrawingSpec(color=(244,164,96), thickness=2)
        )

    return annotated_image

def render_blendshape_metrics(frame, blendshapes):
    """
    Display eye blendshape metrics on the image.
    
    Args:
        image: Image to draw text on
        blendshapes: List of blendshape categories
        
    Returns:
        Dictionary containing blink scores for both eyes
    """
    y_position = TEXT_MARGIN + TEXT_LINE_HEIGHT
    blink_scores = {"left": 0.0, "right": 0.0}
    
    for category in blendshapes:
        if category.category_name in EYE_BLENDSHAPE_CATEGORIES:
            display_text = f"{category.category_name}: {category.score:.2f}"
            cv2.putText(
                frame, display_text, 
                (TEXT_MARGIN, y_position),
                FONT_FACE, FONT_SCALE_INFO, COLOR_INFO, 
                FONT_THICKNESS_INFO, cv2.LINE_AA
            )
            y_position += TEXT_LINE_HEIGHT
        
        # Capture blink scores
        if category.category_name == "eyeBlinkLeft":
            blink_scores["left"] = category.score
        elif category.category_name == "eyeBlinkRight":
            blink_scores["right"] = category.score
    
    return blink_scores, y_position

def calculate_gaze_direction(face_landmarks, blendshapes, image_shape):
    """
    Calculate gaze direction based on iris position and eye look blendshapes.
    
    Args:
        face_landmarks: List of facial landmarks
        blendshapes: Face blendshapes data
        image_shape: Shape of the image (height, width)
        
    Returns:
        Dictionary with gaze vectors for left and right eyes
    """
    height, width = image_shape[:2]
    
    # Get iris center positions
    left_iris = face_landmarks[LEFT_IRIS_CENTER]
    right_iris = face_landmarks[RIGHT_IRIS_CENTER]
    
    left_iris_px = (int(left_iris.x * width), int(left_iris.y * height))
    right_iris_px = (int(right_iris.x * width), int(right_iris.y * height))
    
    # Extract eye look blendshapes
    gaze_data = {
        'left_up': 0.0, 'left_down': 0.0, 'left_in': 0.0, 'left_out': 0.0,
        'right_up': 0.0, 'right_down': 0.0, 'right_in': 0.0, 'right_out': 0.0
    }
    
    for category in blendshapes:
        name = category.category_name
        score = category.score
        
        if name == "eyeLookUpLeft":
            gaze_data['left_up'] = score
        elif name == "eyeLookDownLeft":
            gaze_data['left_down'] = score
        elif name == "eyeLookInLeft":
            gaze_data['left_in'] = score
        elif name == "eyeLookOutLeft":
            gaze_data['left_out'] = score
        elif name == "eyeLookUpRight":
            gaze_data['right_up'] = score
        elif name == "eyeLookDownRight":
            gaze_data['right_down'] = score
        elif name == "eyeLookInRight":
            gaze_data['right_in'] = score
        elif name == "eyeLookOutRight":
            gaze_data['right_out'] = score
    
    # Calculate gaze vectors (x, y)
    left_gaze_x = (gaze_data['left_in'] - gaze_data['left_out']) * ARROW_LENGTH
    left_gaze_y = (gaze_data['left_down'] - gaze_data['left_up']) * ARROW_LENGTH
    
    right_gaze_x = (gaze_data['right_out'] - gaze_data['right_in']) * ARROW_LENGTH
    right_gaze_y = (gaze_data['right_down'] - gaze_data['right_up']) * ARROW_LENGTH
    
    return {
        'left_eye': {
            'center': left_iris_px,
            'vector': (left_gaze_x, left_gaze_y)
        },
        'right_eye': {
            'center': right_iris_px,
            'vector': (right_gaze_x, right_gaze_y)
        }
    }

def draw_gaze_arrows(frame, gaze_info):
    """
    Draw gaze direction arrows on the image.
    
    Args:
        image: Image to draw arrows on
        gaze_info: Dictionary containing gaze direction information
    """
    # Draw left eye gaze arrow
    left_center = gaze_info['left_eye']['center']
    left_vector = gaze_info['left_eye']['vector']
    left_end = (
        int(left_center[0] + left_vector[0]),
        int(left_center[1] + left_vector[1])
    )
    
    cv2.arrowedLine(
        frame, left_center, left_end,
        COLOR_GAZE_LEFT, ARROW_THICKNESS, 
        tipLength=ARROW_TIP_LENGTH
    )
    
    # Draw right eye gaze arrow
    right_center = gaze_info['right_eye']['center']
    right_vector = gaze_info['right_eye']['vector']
    right_end = (
        int(right_center[0] + right_vector[0]),
        int(right_center[1] + right_vector[1])
    )
    
    cv2.arrowedLine(
        frame, right_center, right_end,
        COLOR_GAZE_RIGHT, ARROW_THICKNESS,
        tipLength=ARROW_TIP_LENGTH
    )

def display_drowsiness_alert(image, blink_left, blink_right, y_position, blink_threshold):
    """
    Display appropriate alert based on eye closure detection.
    
    Args:
        image: Image to draw alert on
        blink_left: Left eye blink score
        blink_right: Right eye blink score
        y_position: Vertical position to display alert
    """
    
    alert_y = y_position + 10
    drowsy = False
    
    if blink_left > blink_threshold and blink_right > blink_threshold:
        alert_message = "ALERT: DROWSINESS DETECTED - STAY AWAKE!"
        cv2.putText(
            image, alert_message, 
            (TEXT_MARGIN, alert_y),
            FONT_FACE, FONT_SCALE_ALERT, COLOR_ALERT, 
            FONT_THICKNESS_ALERT, cv2.LINE_AA
        )
        drowsy = True

    return drowsy

def display_time_info(frame):
    """
    Display timestamp info on the frame.
    
    Args:
        frame: frame to draw time info on
    """
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, current_datetime, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, current_datetime, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return frame