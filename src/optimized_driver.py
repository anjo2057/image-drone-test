import cv2
import numpy as np

confh = 0.7  # High confidence threshold
confl = 0.3  # Low confidence threshold
iou_threshold = 0.5
cross_corr_threshold = 0.8

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def extract_patch(frame, box, scale=1.2):
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w // 2, y1 + h // 2
    nw, nh = int(w * scale), int(h * scale)
    nx1 = max(0, cx - nw // 2)
    ny1 = max(0, cy - nh // 2)
    nx2 = min(frame.shape[1], cx + nw // 2)
    ny2 = min(frame.shape[0], cy + nh // 2)
    return frame[ny1:ny2, nx1:nx2]

def cross_correlation_score(prev_patch, curr_patch):
    if prev_patch.shape != curr_patch.shape:
        curr_patch = cv2.resize(curr_patch, (prev_patch.shape[1], prev_patch.shape[0]))
    result = cv2.matchTemplate(curr_patch, prev_patch, cv2.TM_CCOEFF_NORMED)
    return np.max(result)

def track_with_correlation(prev_frame, curr_frame, prev_box):
    prev_patch = extract_patch(prev_frame, prev_box)
    search_area = extract_patch(curr_frame, prev_box, scale=1.5)
    result = cv2.matchTemplate(search_area, prev_patch, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    dx, dy = max_loc
    new_x1 = prev_box[0] + dx
    new_y1 = prev_box[1] + dy
    new_x2 = prev_box[2] + dx
    new_y2 = prev_box[3] + dy
    return [new_x1, new_y1, new_x2, new_y2], max_val

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    prev_detections = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = run_yolo(frame)  # Replace with actual YOLOv7 inference
        valid = [d for d in detections if d[4] > confh]
        candidates = [d for d in detections if confl < d[4] <= confh]

        if valid:
            prev_detections = valid
            prev_frame = frame.copy()
            continue

        for cand in candidates:
            for prev in prev_detections:
                pred_box, corr_score = track_with_correlation(prev_frame, frame, prev[:4])
                iou = compute_iou(pred_box, cand[:4])

                if iou > iou_threshold or corr_score > cross_corr_threshold:
                    cand[4] = confh  # Boost confidence
                    break
            else:
                cand[4] = 0  # Consider as lost

        prev_detections = [d for d in detections if d[4] >= confh]
        prev_frame = frame.copy()

    cap.release()

def run_yolo(frame):
    # Placeholder: Replace this with actual YOLOv7 inference code
    # Expected return: list of [x1, y1, x2, y2, confidence]
    
    return []

if __name__ == "__main__":
    main("path_to_video.mp4")
