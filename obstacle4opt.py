from collections import deque
from ultralytics import YOLO
import mediapipe as mp
import numpy as np
import torch
import time
import cv2

EXTENSION_MULTIPLIER = 10000
YOLO_CONFIDENCE = 0.5
OBSTACLE_CLASSES = {56, 57, 58, 59, 60, 61}
IN_PATH_ANGLE_THRESHOLD = 30
YOLO_DOWNSCALE_FACTOR = 0.5
SMOOTHING_WINDOW = 5
orientation_history = deque(maxlen = SMOOTHING_WINDOW)

FONT = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Error: Could not open webcam.")
    exit()

model = YOLO("yolo11n.pt")

if torch.cuda.is_available():

    model.to("cuda")

model.conf = YOLO_CONFIDENCE

mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(static_image_mode = False, model_complexity = 1)
mp_drawing = mp.solutions.drawing_utils

def to_pixel(landmark, width, height):

    return np.array([int(landmark.x * width), int(landmark.y * height)])

while True:

    start_time = time.time()
    ret, frame = cap.read()

    if not ret:

        break

    frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]

    small_frame = cv2.resize(frame, (int(width * YOLO_DOWNSCALE_FACTOR), int(height * YOLO_DOWNSCALE_FACTOR)))
    results = model(small_frame)
    boxes = results[0].boxes
    detections = boxes.data.cpu().numpy() if boxes.data is not None else np.empty((0, 6))

    obstacle_detections = []
    scale = 1 / YOLO_DOWNSCALE_FACTOR

    for det in detections:

        if int(det[5]) in OBSTACLE_CLASSES:

            x1 = int(det[0] * scale)
            y1 = int(det[1] * scale)
            x2 = int(det[2] * scale)
            y2 = int(det[3] * scale)
            obstacle_detections.append((x1, y1, x2, y2, det[4], int(det[5])))

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose_detector.process(frame_rgb)

    line_color = (0, 255, 0)
    guidance = "Path Clear"
    obstacle_in_path = False

    if pose_results.pose_landmarks:

        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = pose_results.pose_landmarks.landmark

        try:

            left_heel = landmarks[29]
            left_foot_index = landmarks[31]
            right_heel = landmarks[30]
            right_foot_index = landmarks[32]
            left_ankle = landmarks[27]
            right_ankle = landmarks[28]

        except IndexError:

            cv2.imshow("Walking Orientation & Obstacle Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):

                break

            continue

        left_heel_pt = to_pixel(left_heel, width, height)
        left_foot_pt = to_pixel(left_foot_index, width, height)
        right_heel_pt = to_pixel(right_heel, width, height)
        right_foot_pt = to_pixel(right_foot_index, width, height)
        left_ankle_pt = to_pixel(left_ankle, width, height)
        right_ankle_pt = to_pixel(right_ankle, width, height)

        left_vector = left_foot_pt - left_heel_pt
        right_vector = right_foot_pt - right_heel_pt
        vectors = [v for v in (left_vector, right_vector) if np.linalg.norm(v) > 1e-3]

        if vectors:

            orientation_vector = np.mean(vectors, axis = 0)
            norm_val = np.linalg.norm(orientation_vector)
            orientation_unit = orientation_vector / norm_val if norm_val > 0 else np.array([0, 0])

        else:

            orientation_unit = np.array([0, 0])

        if np.linalg.norm(orientation_unit) > 1e-3:

            orientation_history.append(orientation_unit)
            smoothed = np.mean(orientation_history, axis = 0)
            norm_smoothed = np.linalg.norm(smoothed)
            orientation_unit = smoothed / norm_smoothed if norm_smoothed > 0 else np.array([0, 0])

        start_pt = ((left_ankle_pt + right_ankle_pt) / 2).astype(int)
        cv2.circle(frame, tuple(start_pt), 5, (255, 0, 0), -1)

        extended_pt = start_pt + (orientation_unit * EXTENSION_MULTIPLIER).astype(int)
        ret_clip, clipped_start, clipped_end = cv2.clipLine((0, 0, width, height), tuple(start_pt), tuple(extended_pt))

        if ret_clip:

            cv2.line(frame, clipped_start, clipped_end, line_color, 3)

        for (ox1, oy1, ox2, oy2, conf, cls) in obstacle_detections:

            obs_center = np.array([(ox1 + ox2) // 2, (oy1 + oy2) // 2])
            vec_to_obs = obs_center - start_pt
            norm_vec = np.linalg.norm(vec_to_obs)

            if norm_vec < 1e-3:

                continue

            vec_to_obs_unit = vec_to_obs / norm_vec
            dot = np.dot(orientation_unit, vec_to_obs_unit)
            dot = np.clip(dot, -1.0, 1.0)
            angle = np.degrees(np.arccos(dot))

            if angle < IN_PATH_ANGLE_THRESHOLD and dot > 0:

                obstacle_in_path = True
                cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), (0, 0, 255), 3)
                cv2.putText(frame, "Obstacle", (ox1, oy1 - 10), FONT, 0.8, (0, 0, 255), 2)

            else:

                cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), (0, 255, 0), 1)

        if obstacle_in_path:

            guidance = "Obstacle Ahead"

            cv2.line(frame, clipped_start, clipped_end, (0, 0, 255), 3)

        cv2.putText(frame, guidance, (start_pt[0], start_pt[1] - 20), FONT, 1.0, (255, 255, 0), 2)

    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, height - 10), FONT, 0.7, (255, 255, 255), 2)
    cv2.imshow("Walking Orientation & Obstacle Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break

cap.release()
cv2.destroyAllWindows()