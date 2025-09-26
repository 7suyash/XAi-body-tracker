import cv2
import mediapipe as mp
import math
import pyttsx3
import time
from typing import Tuple, Optional, List, Dict, Any

# ==============================================================================
# Constants and Configuration
# ==============================================================================

# --- Pose Landmarks ---
mp_pose = mp.solutions.pose
PoseLandmark = mp_pose.PoseLandmark

# --- Drawing ---
mp_drawing = mp.solutions.drawing_utils
DRAWING_SPEC = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)

# --- Squat Form Angles (in degrees) ---
KNEE_ANGLE_IDEAL = 90.0
KNEE_ANGLE_RANGE = (70.0, 110.0)
HIP_ANGLE_IDEAL = 95.0
HIP_ANGLE_RANGE = (70.0, 120.0)

# --- Visibility & Confidence Thresholds ---
MIN_VISIBILITY_THRESHOLD = 0.7  # Increased for more reliable detection
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# --- Feedback Timing ---
VOICE_FEEDBACK_COOLDOWN = 3  # seconds

# --- Text & Color Settings ---
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_INFO = (255, 255, 0)      # Cyan for angles and XAI
COLOR_GOOD = (0, 255, 0)        # Green for good form
COLOR_ERROR = (0, 0, 255)       # Red for adjustments

# ==============================================================================
# Helper Functions
# ==============================================================================

def initialize_voice_engine() -> pyttsx3.Engine:
    """Initializes and configures the text-to-speech engine."""
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)  # Slightly faster for more fluid feedback
    return engine

def calculate_angle(a: Dict[str, float], b: Dict[str, float], c: Dict[str, float]) -> float:
    """
    Calculates the angle between three 2D points (e.g., elbow angle).
    Points a, b, and c are landmark coordinates. Angle is calculated at point b.
    """
    try:
        # Vector subtraction to get vectors from b to a and b to c
        vec_ba = (a['x'] - b['x'], a['y'] - b['y'])
        vec_bc = (c['x'] - b['x'], c['y'] - b['y'])

        # Dot product of the two vectors
        dot_product = vec_ba[0] * vec_bc[0] + vec_ba[1] * vec_bc[1]

        # Magnitude (length) of each vector
        mag_ba = math.sqrt(vec_ba[0]**2 + vec_ba[1]**2)
        mag_bc = math.sqrt(vec_bc[0]**2 + vec_bc[1]**2)

        # Cosine of the angle
        cosine_angle = dot_product / (mag_ba * mag_bc)
        
        # Ensure the value is within the valid range for acos to avoid math domain errors
        cosine_angle = max(-1.0, min(1.0, cosine_angle))

        # Calculate angle in radians and convert to degrees
        angle = math.acos(cosine_angle)
        return math.degrees(angle)
    except (ZeroDivisionError, ValueError):
        # Return a neutral angle if calculation is not possible
        return 0.0

def are_landmarks_visible(landmarks: List[PoseLandmark], landmark_data: Any) -> bool:
    """Checks if a list of essential landmarks have high visibility."""
    return all(landmark_data[lm.value].visibility > MIN_VISIBILITY_THRESHOLD for lm in landmarks)


def provide_voice_feedback(engine: pyttsx3.Engine, text: str, last_feedback: Dict[str, Any]) -> None:
    """
    Provides voice feedback if the message is new and cooldown has passed.
    """
    current_time = time.time()
    if (text != last_feedback['text'] or 
        (current_time - last_feedback['time']) > VOICE_FEEDBACK_COOLDOWN):
        engine.say(text)
        engine.runAndWait()
        last_feedback['text'] = text
        last_feedback['time'] = current_time

def generate_xai_explanation(knee_angle: float, hip_angle: float) -> str:
    """
    Generates a simple, mock Explainable AI (XAI) feedback message.
    Prioritizes feedback on the joint with the largest deviation from ideal.
    """
    knee_diff = abs(knee_angle - KNEE_ANGLE_IDEAL)
    hip_diff = abs(hip_angle - HIP_ANGLE_IDEAL)
    
    # Give a small tolerance before providing feedback
    if knee_diff < 5 and hip_diff < 5:
        return "Excellent squat form! Keep it up!"

    if knee_diff > hip_diff:
        return f"Focus on your knee. It's off by {int(knee_diff)} degrees."
    else:
        return f"Focus on your hips. They're off by {int(hip_diff)} degrees."

def draw_text(image: Any, text: str, position: Tuple[int, int], color: Tuple[int, int, int], scale: float = 1.0, thickness: int = 2):
    """Utility to draw text on the image with a consistent style."""
    cv2.putText(image, text, position, FONT, scale, color, thickness, cv2.LINE_AA)

# ==============================================================================
# Main Application Logic
# ==============================================================================

def main():
    """Main function to run the squat analysis application."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return
        
    engine = initialize_voice_engine()
    
    # State tracking variables
    last_feedback = {'text': "", 'time': 0}
    full_body_in_frame_spoken = False

    # Define essential landmarks for squat analysis
    essential_landmarks = [
        PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER,
        PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP,
        PoseLandmark.LEFT_KNEE, PoseLandmark.RIGHT_KNEE,
        PoseLandmark.LEFT_ANKLE, PoseLandmark.RIGHT_ANKLE,
    ]

    with mp_pose.Pose(min_detection_confidence=MIN_DETECTION_CONFIDENCE,
                      min_tracking_confidence=MIN_TRACKING_CONFIDENCE) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # --- Image Processing ---
            frame = cv2.flip(frame, 1) # Mirror image for a more natural feel
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False # Performance optimization
            results = pose.process(image_rgb)
            image_rgb.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # --- Landmark Detection and Analysis ---
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Draw landmarks on the image
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=DRAWING_SPEC,
                    connection_drawing_spec=DRAWING_SPEC)

                # Check if the full body is visible
                if are_landmarks_visible(essential_landmarks, landmarks):
                    if not full_body_in_frame_spoken:
                        provide_voice_feedback(engine, "Full body detected. You are ready to squat.", last_feedback)
                        full_body_in_frame_spoken = True

                    # Extract landmark coordinates
                    shoulder_r = landmarks[PoseLandmark.RIGHT_SHOULDER.value]
                    hip_r = landmarks[PoseLandmark.RIGHT_HIP.value]
                    knee_r = landmarks[PoseLandmark.RIGHT_KNEE.value]
                    ankle_r = landmarks[PoseLandmark.RIGHT_ANKLE.value]
                    
                    # Calculate angles
                    knee_angle = calculate_angle({'x': hip_r.x, 'y': hip_r.y}, 
                                                 {'x': knee_r.x, 'y': knee_r.y}, 
                                                 {'x': ankle_r.x, 'y': ankle_r.y})
                    hip_angle = calculate_angle({'x': shoulder_r.x, 'y': shoulder_r.y}, 
                                                {'x': hip_r.x, 'y': hip_r.y}, 
                                                {'x': knee_r.x, 'y': knee_r.y})

                    # Display angle info
                    draw_text(image, f'Knee Angle: {int(knee_angle)}', (50, 50), COLOR_INFO)
                    draw_text(image, f'Hip Angle: {int(hip_angle)}', (50, 100), COLOR_INFO)

                    # --- Form Evaluation ---
                    is_knee_good = KNEE_ANGLE_RANGE[0] <= knee_angle <= KNEE_ANGLE_RANGE[1]
                    is_hip_good = HIP_ANGLE_RANGE[0] <= hip_angle <= HIP_ANGLE_RANGE[1]

                    feedback_text = ""
                    if is_knee_good and is_hip_good:
                        draw_text(image, 'Good Squat Form!', (50, 150), COLOR_GOOD, scale=1.2, thickness=3)
                        feedback_text = "Good form."
                    else:
                        if not is_knee_good:
                            draw_text(image, 'Adjust Knee Angle', (50, 150), COLOR_ERROR, thickness=3)
                        if not is_hip_good:
                            draw_text(image, 'Adjust Hip Angle', (50, 200), COLOR_ERROR, thickness=3)

                    # Generate and display XAI feedback
                    xai_explanation = generate_xai_explanation(knee_angle, hip_angle)
                    draw_text(image, xai_explanation, (50, 250), COLOR_INFO, scale=0.7)
                    
                    # Provide voice feedback based on XAI
                    provide_voice_feedback(engine, xai_explanation, last_feedback)

                else:
                    # Reset state if body goes out of frame
                    full_body_in_frame_spoken = False
                    draw_text(image, 'Please make your full body visible', (50, 50), COLOR_ERROR, thickness=2)
                    provide_voice_feedback(engine, "Please position your full body in the frame.", last_feedback)

            # --- Display Frame ---
            cv2.imshow('Optimized Squat Form Analyzer', image)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    engine.stop()

if __name__ == '__main__':
    main()

