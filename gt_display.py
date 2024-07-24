import os
import cv2 as cv
from pathlib import Path

img_width = 1920
img_height = 1080

gt_path = Path("D:/aicity2024_track5_train/gt.txt")
vid_path = Path("D:/aicity2024_track5_train/videos/054.mp4")
chosen_vid = int(vid_path.stem)

vidcap = cv.VideoCapture(vid_path)

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
    frame_id = 1
    while True:
        success, frame = vidcap.read()
        if not success:
            vidcap.set(cv.CAP_PROP_POS_FRAMES, 0)
            frame_id = 1
            continue

        if frame_id in annotations:
            for (x_min, y_min, x_max, y_max, class_id) in annotations[frame_id]:
                # if class_id in (1,2,3):
                cv.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                cv.putText(frame, f'Class {class_id}', (int(x_min), int(y_min)-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv.imshow('1', frame)
        cv.waitKey(100)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        frame_id += 1

vidcap.release()
cv.destroyAllWindows()
