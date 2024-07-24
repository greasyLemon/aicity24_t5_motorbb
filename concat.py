import os
import cv2 as cv
import numpy as np
from pathlib import Path

img_width = 1920
img_height = 1080

def concatenate_bb(bboxes):
    x_min = min(bbox[0] for bbox in bboxes)
    y_min = min(bbox[1] for bbox in bboxes)
    x_max = max(bbox[2] for bbox in bboxes)
    y_max = max(bbox[3] for bbox in bboxes)
    return [x_min, y_min, x_max, y_max]

def bb_assign(rider_bboxes, motorcycle_bboxes, threshold=100):
    assignments = []
    oddr = []
    oddm = []
    for rider in rider_bboxes:
        rider_x_min, rider_y_min, rider_x_max, rider_y_max = rider
        best_match = None
        min_distance = float('inf')
        for motorcycle in motorcycle_bboxes:
            moto_x_min, moto_y_min, moto_x_max, moto_y_max = motorcycle
            rider_center = ((rider_x_min + rider_x_max) / 2, (rider_y_min + rider_y_max) / 2)
            moto_center = ((moto_x_min + moto_x_max) / 2, (moto_y_min + moto_y_max) / 2)
            distance = np.sqrt((rider_center[0] - moto_center[0]) ** 2 + (rider_center[1] - moto_center[1]) ** 2)
            if distance != 0 and distance < threshold and distance < min_distance and iou(rider, motorcycle) > 0.1:
                min_distance = distance
                best_match = motorcycle
            else:
                oddm.append(motorcycle)
        if best_match:
            assignments.append((rider, best_match))
        else:
            oddr.append(rider)
    return assignments, oddr, oddm


def save_yolo(frame_id, bb, output_path):
    yolo_bb = []
    for bbox in bb:
        x_min, y_min, x_max, y_max = bbox
        center_x = (x_min + x_max) / 2.0 / img_width
        center_y = (y_min + y_max) / 2.0 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        yolo_bb.append(f"0 {center_x} {center_y} {width} {height}")

    new_dir = output_path / "labels"
    os.makedirs(new_dir, exist_ok=True)
    output_path = new_dir / f"{frame_id:04d}.txt"
    with open(output_path, 'w') as f:
        f.write("\n".join(yolo_bb))


def iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = box1_area + box2_area - inter_area
    iou_value = inter_area / union_area

    return iou_value

def nms(bboxes, iou_threshold):
    if len(bboxes) == 0:
        return []

    bboxes = sorted(bboxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)
    selected_bboxes = []

    while bboxes:
        best_bbox = bboxes.pop(0)
        selected_bboxes.append(best_bbox)
        bboxes = [bbox for bbox in bboxes if iou(best_bbox, bbox) < iou_threshold]

    return selected_bboxes

def concat_sv(root_path, gt_path, video_path):
    chosen_vid = int(video_path.stem)
    anno_path = root_path / f"{chosen_vid:03d}"

    vidcap = cv.VideoCapture(video_path)

    with open(gt_path, 'r') as file:
        data = file.readlines()

    annotations = {}
    for line in data:
        parts = line.strip().split(',')
        id = int(parts[0])
        if id == chosen_vid:
            frame_id = int(parts[1])
            x = int(parts[2])
            y = int(parts[3])
            width = int(parts[4])
            height = int(parts[5])
            class_id = int(parts[6])
            center_x = x + width / 2.0
            center_y = y + height / 2.0
            x_min = (center_x - width / 2.0)
            y_min = (center_y - height / 2.0)
            x_max = (center_x + width / 2.0)
            y_max = (center_y + height / 2.0)
            if frame_id not in annotations:
                annotations[frame_id] = []
            annotations[frame_id].append((x_min, y_min, x_max, y_max, class_id))

    while True:
        success, frame = vidcap.read()
        if not success:
            break
        frame_id = int(vidcap.get(cv.CAP_PROP_POS_FRAMES))
        motor_bb = []
        rider_bb = []
        ps_bb = []
        tmp_bb = []
        anno = []
        area_sum = 0
        if frame_id in annotations:
            for obj in annotations[frame_id]:
                x_min, y_min, x_max, y_max, class_id = obj
                if class_id == 1:
                    motor_bb.append([x_min, y_min, x_max, y_max])
                elif class_id in (2, 3):
                    rider_bb.append([x_min, y_min, x_max, y_max])
                elif class_id in (4, 5, 6, 7, 8, 9):
                    ps_bb.append([x_min, y_min, x_max, y_max])

            threshold = 70
            for rider in rider_bb:
                x_min, y_min, x_max, y_max = rider
                rider_area = (x_max - x_min) * (y_max - y_min)
                area_sum += rider_area

            if len(rider_bb) != 0:
                mean_rider_area = area_sum / len(rider_bb)
                if mean_rider_area < 70000:
                    threshold = 150
                else:
                    threshold = 250

            assignmentrp , oddr, oddrp = bb_assign(rider_bb, ps_bb, threshold)

            for rp_pair in assignmentrp:
                x_min, y_min, x_max, y_max = concatenate_bb(rp_pair)
                tmp_bb.append([x_min, y_min, x_max, y_max])

            rider_bb = tmp_bb + oddr

            assignment, odd, oddm = bb_assign(rider_bb, motor_bb, threshold)
            assignment1, odd1, _ = bb_assign(oddrp, oddm, threshold)

            for bb_pair in assignment:
                x_min, y_min, x_max, y_max = concatenate_bb(bb_pair)
                anno.append((x_min, y_min, x_max, y_max))

            for bb_pair in assignment1:
                x_min, y_min, x_max, y_max = concatenate_bb(bb_pair)
                anno.append((x_min, y_min, x_max, y_max))

            for odd_pair in odd:
                x_min, y_min, x_max, y_max = odd_pair
                anno.append((x_min, y_min, x_max, y_max))

            anno = nms(anno, iou_threshold=0.7)

            save_yolo(frame_id - 1, anno, anno_path)

        else:
            save_yolo(frame_id - 1, [], anno_path)

    print("Merge: ok")

