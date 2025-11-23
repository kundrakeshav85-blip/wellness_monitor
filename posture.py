import cv2
import mediapipe as mp
import math
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def angle_3pt(a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy = c

    ba = (ax - bx, ay - by)
    bc = (cx - bx, cy - by)

    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag1 = math.sqrt(ba[0]**2 + ba[1]**2)
    mag2 = math.sqrt(bc[0]**2 + bc[1]**2)

    if mag1 * mag2 == 0:
        return 0

    return math.degrees(math.acos(dot / (mag1 * mag2)))


def vertical_diff(a, b):
    return abs(a[1] - b[1])   # y-axis difference


def horizontal_diff(a, b):
    return abs(a[0] - b[0])   # x-axis difference


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        status = "No Person Detected"
        color = (0, 255, 255)

        if results.pose_landmarks:

            lm = results.pose_landmarks.landmark

            # Key points
            left_sh = (int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                       int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))

            right_sh = (int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                        int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))

            left_ear = (int(lm[mp_pose.PoseLandmark.LEFT_EAR].x * w),
                        int(lm[mp_pose.PoseLandmark.LEFT_EAR].y * h))

            right_ear = (int(lm[mp_pose.PoseLandmark.RIGHT_EAR].x * w),
                         int(lm[mp_pose.PoseLandmark.RIGHT_EAR].y * h))

            left_hip = (int(lm[mp_pose.PoseLandmark.LEFT_HIP].x * w),
                        int(lm[mp_pose.PoseLandmark.LEFT_HIP].y * h))

            # 1) NECK FORWARD ANGLE
            neck_angle = angle_3pt(left_ear, left_sh, left_hip)

            # 2) SHOULDER TILT
            shoulder_tilt = vertical_diff(left_sh, right_sh)

            # 3) HEAD TILT (ear height difference)
            head_tilt = vertical_diff(left_ear, right_ear)

            # 4) BACK STRAIGHTNESS (distance shoulder → hip should be vertical)
            back_slant = horizontal_diff(left_sh, left_hip)

            # === POSTURE CONDITIONS ===
            good_posture = True

            if neck_angle < 135:                # leaning forward
                good_posture = False
            if shoulder_tilt > 45:
                          # shoulder slouch
                good_posture = False
            if head_tilt > 40:                  # head tilted sideways
                good_posture = False
            if back_slant > 60:                 # sideways bending
                good_posture = False

            if good_posture:
                status = "Correct Posture"
                color = (0, 255, 0)
            else:
                status = "Incorrect Posture"
                color = (0, 0, 255)

            # Display info
            cv2.putText(frame, f"Neck Angle: {int(neck_angle)}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.putText(frame, f"Head Tilt: {int(head_tilt)}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.putText(frame, f"Shoulder Tilt: {int(shoulder_tilt)}", (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.putText(frame, status, (20, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow("Posture Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
