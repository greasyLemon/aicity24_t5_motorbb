import os
import cv2 as cv
import numpy as np
from pathlib import Path

img_width = 1920
img_height = 1080

def draw_bb(image, bbox, color=(0, 255, 0)):
    x_min, y_min, x_max, y_max = bbox
    x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
    cv.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

def yolo_to_bbox(yolo_bb, img_width, img_height):
    bbox = []
    for bb in yolo_bb:
        _, center_x, center_y, width, height = bb
        center_x *= img_width
        center_y *= img_height
        width *= img_width
        height *= img_height

        x_min = center_x - (width / 2.0)
        y_min = center_y - (height / 2.0)
        x_max = center_x + (width / 2.0)
        y_max = center_y + (height / 2.0)

        bbox.append([x_min, y_min, x_max, y_max])
    return bbox

def test(root_path, chosen_vid):
    img_path = Path(root_path / chosen_vid, 'images')
    label_path = Path(root_path / chosen_vid, 'labels')
    for img_file in img_path.glob("*.jpg"):
        image = cv.imread(str(img_file))
        label_file = label_path / (img_file.stem + ".txt")

        if label_file.exists():
            with open(label_file, 'r') as file:
                yolo_bb = [list(map(float, line.strip().split())) for line in file.readlines()]
            bbox = yolo_to_bbox(yolo_bb, img_width, img_height)

            for bb in bbox:
                draw_bb(image, bb)

        cv.imshow('Image with BB', image)
        cv.waitKey(100)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
