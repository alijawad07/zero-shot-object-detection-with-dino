import os
from groundingdino.util.inference import load_model, load_image, predict, annotate
import supervision as sv
import cv2
import glob
import shutil
from tqdm import tqdm
import argparse
import subprocess

parser = argparse.ArgumentParser(description='Training script for PP-YOLO-E model on custom dataset.')
parser.add_argument('--src', type=str, required=True,
                    help='Path to the input data')
parser.add_argument('--classes', type=str, required=True,
                    help='path to the .names file')
parser.add_argument('--annot_path', type=str, default='annotation',
                    help='path to the save the annotated images, annotation is the default folder')
parser.add_argument('--save_yolo', type=bool, default=False,
                    help='if the predictions are to be saved in the yolo format as well')
parser.add_argument('--yolo_path', type=str, default='labels',
                    help='path to the save the yolo files, labels is the default folder')
parser.add_argument('--text_prompt', type=str, required=True,
                    help='text prompt, the objects you want to detect in an image, for multiple objects it should be like this: person . car . chair .')
parser.add_argument('--box_thresh', type=float, default=0.35,
                    help='box threshold, by default it is 0.35')
parser.add_argument('--text_thresh', type=float, default=0.25,
                    help='text threshold, by default it is 0.25')

args = parser.parse_args()


# Create the output and labels folders if they don't exist
if args.annot_path:
    annotation_path = args.annot_path
    dir_exists = os.path.isdir(annotation_path)
    if dir_exists:
        shutil.rmtree(annotation_path)
        os.makedirs(annotation_path, exist_ok=True)
    else:
        os.makedirs(annotation_path, exist_ok=True)
else:
    annotation_path = 'annotation'
    dir_exists = os.path.isdir(annotation_path)
    if dir_exists:
        shutil.rmtree(annotation_path)
        os.makedirs(annotation_path, exist_ok=True)
    else:
        os.makedirs(annotation_path, exist_ok=True)
        
if args.yolo_path:
    labels_path = args.yolo_path
    dir_exists = os.path.isdir(labels_path)
    if dir_exists:
        shutil.rmtree(labels_path)
        os.makedirs(labels_path, exist_ok=True)
    else:
        os.makedirs(labels_path, exist_ok=True)
else:
    labels_path = 'labels'
    dir_exists = os.path.isdir(labels_path)
    if dir_exists:
        shutil.rmtree(labels_path)
        os.makedirs(labels_path, exist_ok=True)
    else:
        os.makedirs(labels_path, exist_ok=True)

# Define a function to read the label names from a file
def read_label_names(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    if all(line.isdigit() for line in lines):
        return {int(key): value for key, value in enumerate(lines)}
    else:
        return {key: value for key, value in enumerate(lines)}

# Load the label names from a file
label_names = read_label_names(args.classes)

#path configuration, no need to change anything if you have no idea what it is
HOME =os.getcwd()

WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
WEIGHTS_PATH = os.path.join(HOME, "weights", WEIGHTS_NAME)
print(WEIGHTS_PATH, "; exist:", os.path.isfile(WEIGHTS_PATH))

# Download the weights file if it doesn't exist
if not os.path.isfile(WEIGHTS_PATH):
    weights_url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    subprocess.run(["wget", "-q", weights_url, "-P", "weights"])
    
CONFIG_PATH = os.path.join(HOME, "groundingdino/config/GroundingDINO_SwinT_OGC.py")
print(CONFIG_PATH, "; exist:", os.path.isfile(CONFIG_PATH))

# load model
model = load_model(CONFIG_PATH, WEIGHTS_PATH)

# input images directory
IMAGE_DIR = args.src 

# Text prompt
TEXT_PROMPT = args.text_prompt
#print(TEXT_PROMPT)
BOX_TRESHOLD = args.box_thresh
TEXT_TRESHOLD = args.text_thresh

# Get a list of image files in the directory
image_files = glob.glob(os.path.join(IMAGE_DIR, '*.jpg')) + glob.glob(os.path.join(IMAGE_DIR, '*.jpeg')) + glob.glob(os.path.join(IMAGE_DIR, '*.png'))

for image_file in tqdm(image_files):

    image_source, image = load_image(image_file)

    boxes, logits, phrases = predict(
        model=model, 
        image=image, 
        caption=TEXT_PROMPT, 
        box_threshold=BOX_TRESHOLD, 
        text_threshold=TEXT_TRESHOLD
    )

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

    # Save the annotated image
    annotated_image_file = os.path.splitext(os.path.basename(image_file))[0] + '_annotated.'+ os.path.splitext(os.path.basename(image_file))[1]
    annotated_image_file = os.path.join(annotation_path, annotated_image_file)
    cv2.imwrite(annotated_image_file, annotated_frame)

    #print("Annotated image saved:", annotated_image_file)
    
    if args.save_yolo:
        # Create the label file name based on the image name
        image_name = os.path.basename(image_file)
        label_file_name = os.path.splitext(image_name)[0] + f".txt"
        label_file_name = os.path.join(labels_path, label_file_name)
        #print(label_file_name)
        for i in range(len(boxes)):
            # Get the bounding box coordinates and phrase for the current object
            box = boxes[i].tolist()
            phrase = phrases[i]

            # Get the class ID from label_names based on the phrase
            class_id = list(label_names.keys())[list(label_names.values()).index(phrase)]

            # Create the YOLO line: class_id x_center y_center width height
            x_center = box[0] + box[2] / 2
            y_center = box[1] + box[3] / 2
            width = box[2]
            height = box[3]
            yolo_line = f"{class_id} {x_center} {y_center} {width} {height} \n"

            # Write the YOLO line to the label file
            with open(label_file_name, "a") as f:
                f.write(yolo_line)
