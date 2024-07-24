import cv2 as cv
import os
from pathlib import Path

root_path = Path("D:/aicity2024_track5_train")
video_path = Path("D:/aicity2024_track5_train/videos/089.mp4")

def extract_frms(root_path, video_path):
    vidcap = cv.VideoCapture(video_path)
    count = 0

    while True:
        _, frame = vidcap.read()
        if not _:
            break

        new_path = root_path / video_path.stem
        os.makedirs(new_path, exist_ok=True)

        img_path = new_path / "images"
        os.makedirs(img_path, exist_ok=True)
        frame_path = img_path / f"{count:04d}.jpg"
        cv.imwrite(frame_path, frame)
        count += 1

    print("Number of frames:", count)
    print("Extract: ok")