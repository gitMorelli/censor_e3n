#!/usr/bin/env python3
import json
import argparse
import logging
import json
import os
import io
from time import perf_counter
import pikepdf
import fitz
from PIL import Image
from PyPDF2 import PdfReader
import re
import math
import cv2
import pytesseract #for ocr
pytesseract.pytesseract.tesseract_cmd = r'//vms-e34n-databr/2025-handwriting\programs\tesseract\tesseract.exe'

from src.utils.json_parsing import get_attributes_by_page 
from src.utils.convert_utils import process_pdf_files
from src.utils.feature_extraction import preprocess_text_region, preprocess_page



#JSON_PATH= "//vms-e34n-databr/2025-handwriting\\vscode\censor_e3n\data\q5_tests\\annotazioni" 

def parse_args():
    """Handle command-line arguments."""
    parser = argparse.ArgumentParser(description="Test script")
    '''parser.add_argument(
        "-f", "--folder_path",
        default=TEMPLATES_PATH,
        help="Path to the PDF file to convert",
    )
    parser.add_argument(
        "-s", "--save_path",
        default=SAVE_PATH,
        help="Directory to save the converted PNG images",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )'''
    return parser.parse_args()




#experiments on using pytesseract for identifying the id: https://gemini.google.com/share/54f0575cafcb
def main():
    args = parse_args()
    load_path_file = "//vms-e34n-databr/2025-handwriting\\label_studio_jsons\\json_handwriting_digits_checkmarks.json"
    save_path_file = "//vms-e34n-databr/2025-handwriting\\label_studio_jsons\\json_handwriting_digits_checkmarks_to_predictions.json"

    # Load your exported Label Studio JSON
    with open(load_path_file, 'r') as f:
        data = json.load(f)

    for task in data:
        # 1. Take the existing annotations
        if 'annotations' in task and len(task['annotations']) > 0:
            predictions = []
            
            for ann in task['annotations']:
                # 2. Map annotation data to the prediction format
                pred = {
                    "result": ann['result'],  # This is the actual label data
                    "score": 0.95,            # Optional: helps sort by confidence
                    "model_version": "manual_correction_v1" 
                }
                predictions.append(pred)
            
            # 3. Assign to the predictions key
            task['predictions'] = predictions
            
            # 4. CRITICAL: Remove the annotations so the task becomes "Unlabeled" again
            task['annotations'] = []

    # Save the new file
    with open(save_path_file, 'w') as f:
        json.dump(data, f, indent=2)



    return 0

if __name__ == "__main__":
    main()
