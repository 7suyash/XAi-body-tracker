import cv2
import mediapipe as mp
import math
import pyttsx3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only output errors
import tensorflow as tf  # If you use TF explicitly somewhere


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speech speed
last_spoken = ""  # Track last spoken feedback to avoid repeats
entered_frame_spoken = False  # To track if half-body entry message was spoken

def calculate_angle(a, b, c):
    a = [a.x, a.y]
    b = [b.x, b.y]
    c = [c.x, c.y]

    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]

    dot_product = ba[0]*bc[0] + ba[1]*bc[1]
    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)

    angle = math.acos(dot_product / (mag_ba * mag_bc))
    return math.degrees(angle)

def mock_xai_explanation(knee_angle, hip_angle):
    KNEE_IDEAL = 90
    HIP_IDEAL = 95

    knee_diff = abs(knee_angle - KNEE_IDEAL)
    hip_diff = abs(hip_angle - HIP_IDEAL)

    if knee_diff > hip_diff:
        return f"Knee angle off by {int(knee_diff)} degrees, adjust your knee."
    elif hip_diff > knee_diff:
        return f"Hip angle off by {int(hip_diff)} degrees, adjust your hips."
    else:
        return "Good squat form! Keep it up!"

#    Instead of local camera (0)

cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)  # try 0, 1, 2, 3

with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror horizontally
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Half body visibility detection
            important_landmarks = [
                mp_pose.PoseLandmark.RIGHT_SHOULDER,
                mp_pose.PoseLandmark.LEFT_SHOULDER,
                mp_pose.PoseLandmark.RIGHT_HIP,
                mp_pose.PoseLandmark.LEFT_HIP,
                mp_pose.PoseLandmark.RIGHT_KNEE,
                mp_pose.PoseLandmark.LEFT_KNEE,
                mp_pose.PoseLandmark.RIGHT_ANKLE,
                mp_pose.PoseLandmark.LEFT_ANKLE,
            ]

            visible_count = sum(1 for lm in important_landmarks if landmarks[lm].visibility > 0.5)
            half_body_visible = visible_count >= len(important_landmarks) // 2

            # Announce once when half body comes into frame
            if half_body_visible and not entered_frame_spoken:
                engine.say("Half body detected, you are well in frame.")
                engine.runAndWait()
                entered_frame_spoken = True
            elif not half_body_visible:
                entered_frame_spoken = False

            hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
            ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

            if hip.visibility > 0.5 and knee.visibility > 0.5 and ankle.visibility > 0.5:
                knee_angle = calculate_angle(hip, knee, ankle)
                hip_angle = calculate_angle(knee, hip, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER])

                cv2.putText(image, f'Knee Angle: {int(knee_angle)}', (50,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
                cv2.putText(image, f'Hip Angle: {int(hip_angle)}', (50,100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)

                KNEE_ANGLE_MIN, KNEE_ANGLE_MAX = 70, 110
                HIP_ANGLE_MIN, HIP_ANGLE_MAX = 70, 120

                knee_good = KNEE_ANGLE_MIN <= knee_angle <= KNEE_ANGLE_MAX
                hip_good = HIP_ANGLE_MIN <= hip_angle <= HIP_ANGLE_MAX

                if not knee_good:
                    cv2.putText(image, 'Adjust knee angle', (50, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                if not hip_good:
                    cv2.putText(image, 'Adjust hip angle', (50, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                if knee_good and hip_good:
                    cv2.putText(image, 'Good squat form!', (50, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)

                explanation = mock_xai_explanation(knee_angle, hip_angle)
                cv2.putText(image, explanation, (50, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

                # Voice feedback for form correction
                if explanation != last_spoken:
                    engine.say(explanation)
                    engine.runAndWait()
                    last_spoken = explanation

            else:
                cv2.putText(image, 'Body part not fully visible', (50,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Squat Angle Detection with Voice Feedback', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
