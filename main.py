import os
from pathlib import Path
import argparse
from extract_frms import extract_frms
from concat import concat_sv
from tvt_split import split
from test import test

def main(video_dir, root_path, gt_path):
    video_dir = Path(video_dir)
    root_path = Path(root_path)
    gt_path = Path(gt_path)

    for i in range(101):
        video_path = video_dir / f"{i:03}.mp4"
        chosen_vid = video_path.stem

        extract_frms(root_path, video_path)

        concat_sv(root_path, gt_path, video_path)

        # split(root_path, train=0.7, val=0.15, test=0.15)
        # test(root_path, chosen_vid)
        print("Done vid: ", i)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some videos.")
    parser.add_argument('--video_dir', type=str, required=True, help='Path to the video directory')
    parser.add_argument('--root_path', type=str, required=True, help='Root path for the project, where the images and labels will be saved in')
    parser.add_argument('--gt_path', type=str, required=True, help='Path to the ground truth file')

    args = parser.parse_args()
    main(args.video_dir, args.root_path, args.gt_path)
