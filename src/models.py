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
TEXT_MARGIN = 10

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
            output_facial_transformation_matrixes=True,
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
    y_position = TEXT_MARGIN + TEXT_LINE_HEIGHT + 20
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
    
    return frame, blink_scores, y_position

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
            gaze_data['right_up'] = score
        elif name == "eyeLookDownLeft":
            gaze_data['right_down'] = score
        elif name == "eyeLookInLeft":
            gaze_data['right_in'] = score
        elif name == "eyeLookOutLeft":
            gaze_data['right_out'] = score
        elif name == "eyeLookUpRight":
            gaze_data['left_up'] = score
        elif name == "eyeLookDownRight":
            gaze_data['left_down'] = score
        elif name == "eyeLookInRight":
            gaze_data['left_in'] = score
        elif name == "eyeLookOutRight":
            gaze_data['left_out'] = score
    
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
    return frame

def display_eyes_status(frame, blink_left, blink_right, y_position, blink_threshold):
    """
    Display appropriate alert based on eye closure detection.
    
    Args:
        image: Image to draw alert on
        blink_left: Left eye blink score
        blink_right: Right eye blink score
        y_position: Vertical position to display alert
    """
    
    alert_y = y_position + 10
    
    if blink_left > blink_threshold and blink_right > blink_threshold:
        alert_message = "YOU ARE CLOSING BOTH EYES"
        cv2.putText(
            frame, alert_message, 
            (TEXT_MARGIN, alert_y),
            FONT_FACE, FONT_SCALE_ALERT, COLOR_ALERT, 
            FONT_THICKNESS_ALERT, cv2.LINE_AA
        )
    elif blink_left > blink_threshold:
        alert_message = "YOU ARE CLOSING YOUR LEFT EYE"
        cv2.putText(
            frame, alert_message, 
            (TEXT_MARGIN, alert_y),
            FONT_FACE, FONT_SCALE_ALERT, COLOR_ALERT, 
            FONT_THICKNESS_ALERT, cv2.LINE_AA
        )
    elif blink_right > blink_threshold:
        alert_message = "YOU ARE CLOSING YOUR RIGHT EYE"
        cv2.putText(
            frame, alert_message, 
            (TEXT_MARGIN, alert_y),
            FONT_FACE, FONT_SCALE_ALERT, COLOR_ALERT, 
            FONT_THICKNESS_ALERT, cv2.LINE_AA
        )
        
    return frame

def display_info(frame, fps):
    """
    Display extra infor on frame.
    
    Args:
        frame: frame to draw extra info.
    """
    # Add timestamp and FPS info
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    fps_text = f"FPS: {int(round(fps))}"
    (text_width, text_height), _ = cv2.getTextSize(
        fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    )
    
    x1 = 10
    x2 = frame.shape[1] - text_width - 10
    y = 30
    
    cv2.putText(frame, current_datetime, (x1, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3) 
    cv2.putText(frame, current_datetime, (x1, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(frame, fps_text, (x2, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3) 
    cv2.putText(frame, fps_text, (x2, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add logo info
    logo_size=60
    margin=10
    alpha_scale = 1
    logo = cv2.imread("./logo/logo_mvp.png", cv2.IMREAD_UNCHANGED)
    if logo is None:
        print("Logo not found")
        return frame
    if logo.shape[2] == 4:
        logo = cv2.cvtColor(logo, cv2.COLOR_BGRA2RGBA)
    else:
        logo = cv2.cvtColor(logo, cv2.COLOR_BGR2RGB)

    # Resize logo keeping aspect ratio
    h_logo, w_logo = logo.shape[:2]
    scale = logo_size / float(w_logo)
    new_w = int(w_logo * scale)
    new_h = int(h_logo * scale)
    if new_w <= 0 or new_h <= 0:
        return frame
    logo = cv2.resize(logo, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Separate color and alpha channels safely
    if logo.shape[2] == 4:
        logo_bgr = logo[:, :, :3].astype(np.float32)
        alpha = logo[:, :, 3].astype(np.float32) / 255.0
    else:
        logo_bgr = logo[:, :, :3].astype(np.float32)
        alpha = np.ones((logo.shape[0], logo.shape[1]), dtype=np.float32)

    # Apply global alpha_scale and clip
    alpha = np.clip(alpha * float(alpha_scale), 0.0, 1.0)
    # Make alpha shape H x W x 1 for broadcasting over BGR channels
    alpha_3 = alpha[:, :, None]

    # ROI coordinates
    fh, fw = frame.shape[:2]
    y1 = fh - new_h - margin
    y2 = y1 + new_h
    x1 = fw - new_w - margin
    x2 = x1 + new_w

    if y1 < 0 or x1 < 0:
        # Not enough space; skip
        return frame

    roi = frame[y1:y2, x1:x2].astype(np.float32)

    # Blend: out = alpha * logo + (1-alpha) * roi
    blended = (alpha_3 * logo_bgr) + ((1.0 - alpha_3) * roi)

    # Ensure uint8 and write back
    frame[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return frame

def get_head_orientation(matrix: np.ndarray) -> dict[str, float]:
    """
    Compute head orientation (roll, pitch, yaw) from a 4x4 transformation matrix.
    
    Parameters
    ----------
    matrix : np.ndarray
        4x4 transformation matrix (e.g., from MediaPipe or camera pose estimation).
        
    Returns
    -------
    dict[str, float]
        {
            "roll": float,   # Head tilt left/right (degrees)
            "pitch": float,  # Head nod up/down (degrees)
            "yaw": float,    # Head turn left/right (degrees)
        }
    """
    # Extract the 3x3 rotation matrix
    R = matrix[:3, :3]
    
    # Compute Euler angles from the rotation matrix
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6  # Check for gimbal lock (singular case)

    if not singular:
        pitch = np.arctan2(-R[2, 1], R[2, 2])
        yaw   = np.arctan2(R[2, 0], sy)
        roll  = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Handle singularity (when sy is near zero)
        pitch = np.arctan2(-R[2, 1], R[2, 2])
        yaw   = np.arctan2(R[0, 1], R[1, 1])
        roll  = 0.0

    # Convert radians to degrees
    return {
        "roll": np.degrees(roll),
        "pitch": np.degrees(pitch),
        "yaw": np.degrees(yaw),
    }

def display_head_orientation(frame: np.ndarray, orientation: dict[str, float]) -> np.ndarray:
    """
    Display head orientation (roll, pitch, yaw) in the top-right corner of the frame.
    
    Parameters
    ----------
    frame : np.ndarray
        Input video frame (BGR).
    orientation : dict[str, float]
        Dictionary containing roll, pitch, yaw values (in degrees).

    Returns
    -------
    np.ndarray
        Frame with overlay text.
    """
    # Prepare display text
    roll_text  = f"Roll : {orientation['roll']:.1f} deg"
    pitch_text = f"Pitch: {orientation['pitch']:.1f} deg"
    yaw_text   = f"Yaw  : {orientation['yaw']:.1f} deg"
    lines = [roll_text, pitch_text, yaw_text]
    
    # Text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    line_height = 22
    margin = 10

    (text_w, text_h), _ = cv2.getTextSize("Yaw  : 00.0 deg", font, font_scale, thickness)

    # Start position (top-right corner)
    x = frame.shape[1] - text_w - 10  # adjust horizontal position if needed
    y = margin + 50

    for line in lines:
        # White outline
        cv2.putText(frame, line, (x, y), font, font_scale, (255, 255, 255), thickness + 1, cv2.LINE_AA)
        # Black inner text
        cv2.putText(frame, line, (x, y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        y += line_height

    return frame
