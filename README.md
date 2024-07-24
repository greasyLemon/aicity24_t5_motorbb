# aicity24_t5_motorbb

**The AI City 2024 Challenge-Track 5 contains the ground truth for 9 classes:**

	motorbike
	DHelmet
	DNoHelmet
	P1Helmet
	P1NoHelmet
	P2Helmet
	P2NoHelmet
	P0Helmet
	P0NoHelmet

**This repository merges the 9 ground truth classes into 1 single motocycle class for custom object detection training.**

`gt_display`: shows videos with the original 9-class bounding boxes (OPTIONAL).

`extract_frms`: extracts and saves frames from the videos.

`concat.py`: contains all the merging code. 

`tvt_split.py`: splits the data into train, validation and test subsets after merging (OPTIONAL).

`test.py`: show videos with the motorcycle bounding boxes after merging is done (OPTIONAL).

You can comment out the optional functionalities in main.py.

**How the merging works:**

1. Group the 9 classes into 3 main classes: **motorcycle**, **driver** and **passenger**.
2. Assign and merge the 2 closest (if any)** driver and passenger** bounding box into 1 driver bounding box.
3. Assign and merger the 2 closest (if any)** driver and motorcycle** bounding box into 1 motorcycle bounding box.
4. Apply **Non-Maximum Supression** to filter out duplicate or remaining bounding boxes.

**Main function in concat.py:**

`iou(box1, box2)`: calculate the Intersection of Union between 2 bounding box, which will later be used as thresholds in `bb_assign` and `nms`.

`bb_assign(bboxes1, bboxes2, threshold)`: assign the 2 closest bounding boxes into a pair via calculating their center distance. `threshold` is the maximum distance between 2 assigned boxes.

`concatenate_bb(bboxes)`: merge the assigned bounding boxes into 1 larger box with the `x_min, y_min, x_max, y_max` taken from the original bounding box coordinates.

`nsm(bboxes, iou_threshold)`: Non-Maximum Supression algorithm for filtering out the duplicate bounding boxes. `iou_threshold` is the maximum iou value for the bounding box to be kept.

**How to run:**

Run **main.py**:

	python main.py --video_dir "YOUR_VIDEO_PATH" --root_path "YOUR_ROOT_PATH" --gt_path "YOUR_GROUND_TRUTH"

`--root-path` is where the images and labels will be stored.

**Merging visualization:**

![Screenshot 2024-07-24 160642](https://github.com/user-attachments/assets/4124527e-87aa-4c68-b5b1-be0416705dab)
![Screenshot 2024-07-24 160738](https://github.com/user-attachments/assets/8b55e222-f30c-4af9-8656-1c223a5fed67)
![Screenshot 2024-07-24 160809](https://github.com/user-attachments/assets/eab27f07-da91-4338-b1c3-08baa5b49268)

See more images examples: [Notebook](https://colab.research.google.com/drive/1_6zpFOJCghHVUVm1xAutQ_tBUsgvv_sp?usp=sharing)




 
