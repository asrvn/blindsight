from ultralytics import YOLO
import concurrent.futures
import mediapipe as mp
import numpy as np
import asyncio
import logging
import torch
import time
import cv2

logging.getLogger('ultralytics').setLevel(logging.ERROR)

def run_detection(frame_rgb, yolo_model):

    return yolo_model(frame_rgb)

def run_depth_estimation(frame_rgb, midas, transform, device):

    input_image = transform(frame_rgb).to(device)

    with torch.no_grad():

        depth = midas(input_image)

    depth = torch.nn.functional.interpolate(

        depth.unsqueeze(1),
        size=frame_rgb.shape[:2],
        mode = "bilinear",
        align_corners = False

    ).squeeze()

    return depth

def draw_detections(frame, detection_info, depth_threshold):

    for det in detection_info:

        x1, y1, x2, y2 = det["bbox"]
        score = det["score"]
        cls_id = det["class_id"]
        mean_depth = det["mean_depth"]
        color = (0, 255, 0) if mean_depth < depth_threshold else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"Class {cls_id}: {score:.2f}, Depth: {mean_depth:.1f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

def process_frame_concurrent(frame, yolo_model, midas, transform, device):

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with concurrent.futures.ThreadPoolExecutor(max_workers = 2) as executor:

        future_det = executor.submit(run_detection, frame_rgb, yolo_model)
        future_depth = executor.submit(run_depth_estimation, frame_rgb, midas, transform, device)
        detection_result = future_det.result()
        depth_tensor = future_depth.result()

    if isinstance(detection_result, list):

        detection_result = detection_result[0]

    boxes = detection_result.boxes.data.cpu().numpy()
    depth_map = depth_tensor.cpu().numpy()

    detection_info = []
    for box in boxes:

        x1, y1, x2, y2, score, cls_id = box
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, frame.shape[1] - 1)
        y2 = min(y2, frame.shape[0] - 1)
        bbox_region = depth_map[y1 : y2, x1 : x2]

        if bbox_region.size == 0:

            continue

        mean_depth = np.mean(bbox_region)
        detection_info.append({

            "bbox": (x1, y1, x2, y2),
            "score": score,
            "class_id": int(cls_id),
            "mean_depth": mean_depth,
            "center": ((x1 + x2) // 2, (y1 + y2) // 2)

        })

    return detection_info, depth_map

def load_models(device):

    yolo_model = YOLO('yolo11s.pt')
    yolo_model.fuse()
    yolo_model.to(device).eval()
    yolo_model.model.half()

    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
    midas.to(device).eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform

    return yolo_model, midas, transform

def compute_heading(landmarks, image_width, image_height):

    try:

        nose = landmarks[0]
        left_foot_index = landmarks[31]
        right_foot_index = landmarks[32]
        mid_foot = (

            (left_foot_index.x + right_foot_index.x) / 2.0,
            (left_foot_index.y + right_foot_index.y) / 2.0

        )
        mid_foot_px = (int(mid_foot[0] * image_width), int(mid_foot[1] * image_height))
        nose_px = (int(nose.x * image_width), int(nose.y * image_height))
        left_foot_px = (int(left_foot_index.x * image_width), int(left_foot_index.y * image_height))
        right_foot_px = (int(right_foot_index.x * image_width), int(right_foot_index.y * image_height))
        foot_baseline = np.array([

            right_foot_px[0] - left_foot_px[0],
            right_foot_px[1] - left_foot_px[1]

        ], dtype = float)
        candidate1 = np.array([foot_baseline[1], -foot_baseline[0]], dtype = float)
        candidate2 = np.array([-foot_baseline[1], foot_baseline[0]], dtype = float)
        nose_vector = np.array([

            nose_px[0] - mid_foot_px[0],
            nose_px[1] - mid_foot_px[1]

        ], dtype = float)

        if np.dot(nose_vector, candidate1) >= np.dot(nose_vector, candidate2):

            forward_vector = candidate1

        else:

            forward_vector = candidate2

        norm = np.linalg.norm(forward_vector)

        if norm == 0:

            forward_vector = np.array([1, 0], dtype = float)

        else:

            forward_vector /= norm

        arrow_length = 150
        end_point = (

            int(mid_foot_px[0] + forward_vector[0] * arrow_length),
            int(mid_foot_px[1] + forward_vector[1] * arrow_length)

        )

        return mid_foot_px, end_point

    except Exception:

        return (0, 0), (0, 0)

def call(update_callback, loop):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo_model, midas, transform = load_models(device)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(

        static_image_mode = False,
        model_complexity = 1,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5

    )

    cap = cv2.VideoCapture(0)
    depth_threshold = 50
    box_size = 60
    margin = 50
    prev_direction = None

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:

            break

        detection_info, depth_map = process_frame_concurrent(frame, yolo_model, midas, transform, device)
        _ = draw_detections(frame.copy(), detection_info, depth_threshold)

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        direction = None

        if results.pose_landmarks:

            start_point, end_point = compute_heading(

                results.pose_landmarks.landmark,
                frame.shape[1],
                frame.shape[0]

            )

            box_half = box_size // 2
            obstacle_box = (

                end_point[0] - box_half,
                end_point[1] - box_half,
                end_point[0] + box_half,
                end_point[1] + box_half

            )

            obstacle_detected = False
            for det in detection_info:

                x1, y1, x2, y2 = det["bbox"]

                if (x1 < obstacle_box[2] and x2 > obstacle_box[0] and y1 < obstacle_box[3] and y2 > obstacle_box[1]):

                    obstacle_detected = True
                    break

            frame_center = frame.shape[1] / 2

            if obstacle_detected:

                direction = "stop"

            else:

                if end_point[0] < frame_center - margin:

                    direction = "left"

                elif end_point[0] > frame_center + margin:

                    direction = "right"

                else:

                    direction = "forward"

        else:

            direction = "stop"

        if direction and direction != prev_direction:

            try:

                asyncio.run_coroutine_threadsafe(update_callback(direction), loop)

            except Exception:

                pass

            prev_direction = direction