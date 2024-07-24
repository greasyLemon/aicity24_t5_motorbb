import os
import shutil
from pathlib import Path
import random

def split(root_path, train=0.7, val=0.15, test=0.15):
    image_path = Path(root_path, "images")
    label_path = Path(root_path, "labels")

    for subset in ['train', 'test', 'val']:
        img_subset = image_path / subset
        label_subset = label_path / subset

        if img_subset.exists():
            shutil.rmtree(img_subset)

        if label_subset.exists():
            shutil.rmtree(label_subset)

        img_subset.mkdir(parents=True, exist_ok=True)
        label_subset.mkdir(parents=True, exist_ok=True)

    images = list(image_path.rglob('*.jpg'))
    labels = list(label_path.rglob('*.txt'))

    random.shuffle(images)

    total = len(images)
    train_end = int(total * train)
    val_end = int(train_end + total * val)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    for subset, img_files in zip(['train', 'test', 'val'],[train_images, test_images, val_images]):
        desimg_dir = Path(image_path / subset)
        deslbl_dir = Path(label_path / subset)
        for file in img_files:
            shutil.copy(file, desimg_dir)
            lbl_file = label_path / file.name.replace('.jpg', '.txt')
            if lbl_file in labels:
                shutil.copy(lbl_file, deslbl_dir)

    print("split: ok")
