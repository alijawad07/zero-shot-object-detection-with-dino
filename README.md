# Zero-Shot Object Detection with DINO Grounding and save the annotations in the yolo format

This repository contains a script for performing zero-shot object detection using DINO (Transformers for Dense Object Recognition) grounding. The script detects objects in images without prior training on those specific objects, and it utilizes the DINO framework for self-supervised learning to extract meaningful visual representations. Also if we want to run the inference on a folder of images or if we want to make a darknet format dataset for yolo model we can do that as well.


## Installation

1. Clone this repository:

```bash
git clone https://github.com/alijawad07/zero-shot-object-detection-with-dino
```
2.  install the following
```bash
pip install -r requirements.txt
pip install -q -e .

#if you want to use dataset from roboflow
pip install -q roboflow

```
3. execute the following
```bash
export CUDA_LAUNCH_BLOCKING=1
```

## Usage
```bash
python3 main.py --src --classes --annot_path --save_yolo --yolo_path --text_prompt --box_thresh --text_thresh
```
- **--src** = type=str, required=True,
                    help='Path to the input data'
- **--classes** = type=str, required=True,
                    help='path to the .names file'
- **--annot_path** = type=str, default='annotation',
                    help='path to the save the annotated images, annotation is the default folder'
- **--save_yolo** = type=bool, default=False,
                    help='if the predictions are to be saved in the yolo format as well'
- **--yolo_path** = type=str, default='labels',
                    help='path to the save the yolo files, labels is the default folder'
- **--text_prompt** = type=str, required=True,
                    help='text prompt, the objects you want to detect in an image, for multiple objects it should be like this: person . car . chair .'
- **--box_thresh** = type=float, default=0.35,
                    help='box threshold, by default it is 0.35'
- **--text_thresh** = type=float, default=0.25,
                    help='text threshold, by default it is 0.25'

## Acknowledgments
- This project utilizes the [groundingdino](https://github.com/IDEA-Research/GroundingDINO) for inference.
