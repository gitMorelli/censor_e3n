#!/usr/bin/env python3
import argparse
import logging
import json
import os
import io
from time import perf_counter
import re
import math
import cv2
import shutil

from label_studio_sdk.client import LabelStudio

import requests
import urllib.parse
import random
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.utils.file_utils import get_basename, create_folder, remove_folder, load_annotation_tree

URL = "http://localhost:8080"
API_KEY = "b24196523ec2086771fcf26a4c5b4a57d99cd75b"
IMG_INPUT_PATH = "//vms-e34n-databr/2025-handwriting\\data"

PROJECT_ID = 35 #39 #single digits #35 #handwriting only #30 #label questinnaires 31 #label partial regions

#OUTPUT_PATH = "//vms-e34n-databr/2025-handwriting\\downloads"
#OUTPUT_PATH = "//vms-e34n-databr/2025-handwriting\\downloads\\partial_regions_export"
OUTPUT_PATH = "//vms-e34n-databr/2025-handwriting\\downloads\\handwriting_only_export_v2"
#OUTPUT_PATH = "//vms-e34n-databr/2025-handwriting\\downloads\\single_digits_export"

#JSON_PATH= "//vms-e34n-databr/2025-handwriting\\vscode\censor_e3n\data\q5_tests\\annotazioni" 
# --- EXECUTION ---
# Map your label names to IDs (0, 1, 2...)
MY_CLASSES = {
    "Checkmark": 0,
    "Filled": 1,
    "Handwriting": 2,
    "Number": 3,
}
'''MY_CLASSES = {
    "Handwriting": 0,
}'''
SKIP_DOWNLOAD = False
SKIP_CONVERT = False
VISUALIZE = True

def convert_json_to_yolo_labels(json_path, output_folder, class_mapping, with_images = True):
    """
    Converts Label Studio JSON to YOLO .txt files with or without images based on the flag.
    """
    slash_char = '%5C'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    labels_folder = os.path.join(output_folder, "labels")
    images_folder = os.path.join(output_folder, "images")
    if not os.path.exists(labels_folder):
        os.makedirs(labels_folder)
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)

    with open(json_path, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    
    print('Number of labelled tasks: ',len(tasks))

    file_list = []
    n_duplicate_names = 0

    for task in tasks:
        # 1. Sanitize Filename (Remove ?d= and URL encoding)
        raw_path = task['data']['image']
        #print(raw_path)
        parts = raw_path.split(slash_char)

        # 3. Take the last two pieces and join them with an underscore
        # parts[-2:] gets the second-to-last and last elements

        new_filename = "_".join(parts[-2:]) #change for different exports (-2 if generated from folder with subfolders, -1 if from folder with files)

        new_filename = new_filename.rsplit('.', 1)[0]
        #print(new_filename)
        #return 0
        #count how many times new_filename already exists in file_list and if it does, add a suffix _1, _2, etc. until it's unique
        #n_copies=0
        if new_filename in file_list:
            n_duplicate_names += 1
            suffix = 1
            while f"{new_filename}_{suffix}" in file_list:
                suffix += 1
            new_filename = f"{new_filename}_{suffix}"
            #n_copies = suffix
        file_list.append(new_filename)
        
        label_file_path = os.path.join(labels_folder, f"{new_filename}.txt")

        if with_images:
            # 4. Copy the image to the output folder with the new filename
            # Note: This assumes the image path is accessible from this script
            image_path = raw_path.split('?d=')[-1]
            #substitute the slash char in the image path with the actual slash char
            image_path = image_path.replace(slash_char, os.sep)
            image_path = os.path.join(IMG_INPUT_PATH, image_path)
            #copy the image from the image path to the output folder with the new filename and the same extension
            if os.path.exists(image_path):
                img_extension = os.path.splitext(image_path)[1]
                new_image_path = os.path.join(images_folder, f"{new_filename}{img_extension}")
                shutil.copy(image_path,new_image_path)

        # 2. Create the file (This ensures an empty file exists even if no boxes)
        with open(label_file_path, 'w') as f_out:
            # Check if there are any annotations at all
            if not task.get('annotations'):
                continue # The file remains empty
                
            for ann in task['annotations']:
                # Optional: Only process if the annotation isn't 'skipped'
                # if ann.get('was_cancelled'): continue 
                
                for res in ann.get('result', []):
                    if res['type'] != 'rectanglelabels':
                        continue
                    
                    val = res['value']
                    class_name = val['rectanglelabels'][0]
                    class_id = class_mapping.get(class_name, 0)

                    # Normalized YOLO coordinates (0.0 to 1.0)
                    x = (val['x'] + val['width'] / 2.0) / 100.0
                    y = (val['y'] + val['height'] / 2.0) / 100.0
                    w = val['width'] / 100.0
                    h = val['height'] / 100.0

                    f_out.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
    # also save the classes.txt file for yolo
    with open(os.path.join(output_folder, "classes.txt"), 'w') as f:
        for class_name in class_mapping:
            f.write(f"{class_name}\n")
    print("Duplicate files with same name (before adding suffix): ", n_duplicate_names)

def visualize_yolo_data(dataset_path, class_names, n=5):
    """
    Visualizes n random images with their bounding boxes.
    
    Args:
        dataset_path (str): Path to root folder (containing 'images' and 'labels')
        class_names (list): List of class names in order of their IDs
        n (int): Number of images to display
    """
    img_dir = Path(dataset_path) / "images"
    lbl_dir = Path(dataset_path) / "labels"
    
    # Get all image files
    img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg"))
    
    if len(img_files) < n:
        n = len(img_files)
        
    selected_imgs = random.sample(img_files, n)
    
    # Create a color map for different classes
    colors = plt.cm.get_cmap("tab10", len(class_names))
    
    plt.figure(figsize=(20, 10))
    
    for i, img_path in enumerate(selected_imgs):
        # Read image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        
        # Find corresponding label file
        label_path = lbl_dir / f"{img_path.stem}.txt"
        
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f.readlines():
                    cls_id, x_c, y_c, bw, bh = map(float, line.split())
                    
                    # Convert YOLO format to Pixel coordinates
                    # x_min = (x_center - width/2) * img_width
                    # y_min = (y_center - height/2) * img_height
                    x1 = int((x_c - bw / 2) * w)
                    y1 = int((y_c - bh / 2) * h)
                    x2 = int((x_c + bw / 2) * w)
                    y2 = int((y_c + bh / 2) * h)
                    
                    # Get color (scaled to 0-255)
                    color = [int(c * 255) for c in colors(int(cls_id))[:3]]
                    
                    # Draw Box and Label
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, class_names[int(cls_id)], (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Plotting
        plt.subplot(1, n, i + 1)
        plt.imshow(img)
        plt.title(img_path.name)
        plt.axis("off")
        
    plt.tight_layout()
    plt.show()
#experiments on using pytesseract for identifying the id: https://gemini.google.com/share/54f0575cafcb
def main():

    '''print("Start connection test")
    headers = {
        "Authorization": f"Token {API_KEY}"
    }
    #'Authorization: Bearer <token>'

    try:
        response = requests.get(URL+f'/api/projects/{PROJECT_ID}', headers=headers)
        if response.status_code == 200:
            print("Success! The token and path are correct.")
        elif response.status_code == 401:
            print("401: Still unauthorized. Double-check the token in the UI.")
        else:
            print(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Could not connect to the server: {e}")'''
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    
    out_file = os.path.join(OUTPUT_PATH, "json_results.json")

    if not SKIP_DOWNLOAD:
        # 1. Initialize the new Client
        ls = LabelStudio(
            base_url=URL,
            api_key=API_KEY
        )

        # 2. Define your filters
        # Note: The parameters might slightly differ in the new SDK 
        # but the API allows passing them as extra arguments.
        export_params = {
            "task_filter_options": {
                "annotated": "only",
                "skipped": "exclude"
            }
        }

        print("Creating export snapshot...")
        # 3. Use the new path: client.projects.exports.create()
        export_snapshot = ls.projects.exports.create(
            id=PROJECT_ID,
            **export_params
        )

        # In the new SDK, export_snapshot is an object. Get the ID.
        snapshot_id = export_snapshot.id

        print(f"Downloading export ID {snapshot_id} in YOLO format...")

        # 4. Download using the client directly
        # This returns a generator or bytes, so we save it manually
        response = ls.projects.exports.download(
            id=PROJECT_ID,
            export_pk=snapshot_id,
            export_type="JSON"
        )

        # Since the SDK returns a response object/generator:
        with open(out_file, "wb") as f:
            for chunk in response:
                f.write(chunk)
        print("Done! File saved")

    yolo_out_file = os.path.join(OUTPUT_PATH, "yolo_results")
    if not SKIP_CONVERT:
        print("Start conversion to yolo format")
        if os.path.exists(yolo_out_file):
            remove_folder(yolo_out_file)
        print("Removed folder, now recreating ...")
        #add a sleep of 5ms

        create_folder(yolo_out_file)
        convert_json_to_yolo_labels(out_file, yolo_out_file, MY_CLASSES)
        print("Label conversion complete. No images were processed.")

    if VISUALIZE:
        class_names = [name for name, _ in sorted(MY_CLASSES.items(), key=lambda item: item[1])]
        visualize_yolo_data(yolo_out_file, class_names, n=5)

    return 0

if __name__ == "__main__":
    main()