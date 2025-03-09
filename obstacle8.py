from ultralytics import YOLO
import concurrent.futures
import numpy as np
import torch
import cv2

def load_models(device):

    yolo_model = YOLO('yolo11s.pt')
    yolo_model.fuse()
    yolo_model.to(device).eval()

    yolo_model.model.half()

    midas_model_type = "DPT_Hybrid"
    midas = torch.hub.load("intel-isl/MiDaS", midas_model_type, trust_repo = True)
    midas.to(device).eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo = True)

    if "DPT" in midas_model_type:

        transform = midas_transforms.dpt_transform

    else:

        transform = midas_transforms.small_transform

    return yolo_model, midas, transform

def run_detection(frame_rgb, yolo_model):

    results = yolo_model(frame_rgb)[0]
    return results

def run_depth_estimation(frame_rgb, midas, transform, device):

    input_tensor = transform(frame_rgb).to(device)

    with torch.no_grad():

        depth = midas(input_tensor)
        depth = torch.nn.functional.interpolate(

            depth.unsqueeze(1),
            size = frame_rgb.shape[:2],
            mode = "bicubic",
            align_corners = False

        ).squeeze()

    return depth

def process_frame_concurrent(frame, yolo_model, midas, transform, device):

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with concurrent.futures.ThreadPoolExecutor(max_workers = 2) as executor:

        future_det = executor.submit(run_detection, frame_rgb, yolo_model)
        future_depth = executor.submit(run_depth_estimation, frame_rgb, midas, transform, device)
        detection_result = future_det.result()
        depth_tensor = future_depth.result()

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
            "mean_depth": mean_depth

        })

    return detection_info, depth_map, boxes

def draw_detections(frame, detection_info, yolo_model):

    for det in detection_info:

        x1, y1, x2, y2 = det["bbox"]
        score = det["score"]
        mean_depth = det["mean_depth"]
        cls_id = det["class_id"]
        class_name = yolo_model.model.names[cls_id]
        label = f"{class_name} {score:.2f}, depth: {mean_depth:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    yolo_model, midas, transform = load_models(device)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("Error: Could not open video stream.")
        return

    try:

        while True:

            ret, frame = cap.read()

            if not ret:

                print("Frame capture failed. Exiting loop.")
                break

            detection_info, depth_map, _ = process_frame_concurrent(frame, yolo_model, midas, transform, device)
            annotated_frame = draw_detections(frame.copy(), detection_info, yolo_model)
            depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
            depth_norm = np.uint8(depth_norm)
            depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)

            combined_view = np.hstack((annotated_frame, depth_color))
            cv2.imshow("Object Detection & Depth Estimation", combined_view)

            if cv2.waitKey(1) & 0xFF == ord('q'):

                break

    except Exception as e:

        print("An error occurred:", e)

    finally:

        cap.release()
        cv2.destroyAllWindows()

        torch.cuda.empty_cache()

if __name__ == "__main__":

    main()
