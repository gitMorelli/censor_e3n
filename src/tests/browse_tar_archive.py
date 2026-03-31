import argparse
import pandas as pd
import os
from pathlib import Path
import fitz  # PyMuPDF
import numpy as np
import cv2
from time import perf_counter
import time
import tarfile
import matplotlib.pyplot as plt

from src.utils.file_utils import get_basename, create_folder, remove_folder, load_annotation_tree

def process_and_extract_tar(tar_path, output_folder):
    # 1. Ensure the output directory exists
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    #print(f"Opening: {tar_path}")
    
    list_images = []
    with tarfile.open(tar_path, 'r') as archive:
        # 2. Extract the entire contents to the user-defined folder
        archive.extractall(path=output_path)
        #print(f"Successfully extracted all files to: {output_path}")

        # 3. Iterate through members to check PNG channels using cv2
        for member in archive.getmembers():
            if member.name.lower().endswith('.png'):
                # Extract file-like object from tar
                file_obj = archive.extractfile(member)
                if file_obj is not None:
                    # Read bytes and convert to a numpy array for OpenCV
                    file_bytes = np.frombuffer(file_obj.read(), np.uint8)
                    
                    # Use cv2.IMREAD_UNCHANGED to see Alpha channels (4th channel)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

                    if img is not None:
                        # Get shape (height, width, channels)
                        # Note: Grayscale images will only have (H, W)
                        shape = img.shape
                        channels = shape[2] if len(shape) == 3 else 1
                        
                        #print(f"File: {member.name} | Channels: {channels} | Shape: {shape}")
                        list_images.append(img.copy())
                    else:
                        #print(f"Warning: Could not decode {member.name}")
                        pass
    return list_images


QUESTIONNAIRE="8"
def main():
    args = parse_args()
    #filename ="A0A0D6X7.tar"
    main_path = "//smb-recherche-s1.prod-powerscale.intra.igr.fr/E34N_HANDWRITING$\\censored_files\\"
    if QUESTIONNAIRE in ["13","9","8"]:
        tar_path = os.path.join(main_path,f"Q{QUESTIONNAIRE}\\images")
    elif QUESTIONNAIRE in ["12","11"]:
        tar_path = os.path.join(main_path,f"Q{QUESTIONNAIRE}\\archived_{QUESTIONNAIRE}")
    elif QUESTIONNAIRE=="10":
        tar_path = os.path.join(main_path,f"Q{QUESTIONNAIRE}\\{QUESTIONNAIRE}")
    ref_path = os.path.join(main_path,f"Q{QUESTIONNAIRE}\\run_data_q{QUESTIONNAIRE}\\csv_results_aggregated", "combined_success_ids.csv")
    out_path = os.path.join("//vms-e34n-databr/2025-handwriting\\data\\test_tar",f"Q{QUESTIONNAIRE}" )
    #eliminate the output folder if it exists
    if os.path.exists(out_path):
        remove_folder(out_path)
    create_folder(out_path)

    #load the csv file
    df = pd.read_csv(ref_path)
    cols = df.columns
    for col in cols:
        print(col)
    #print stats on warnings
    
    df = analysis(df)

    #rows that have Warning_censoring==No warning and Warning_ordering==First stage or Warning_ordering==Second stage
    filtered_df=df[(df["Warning_censoring"]=="No warning") & ((df["Warning_ordering"]=="First stage") | (df["Warning_ordering"]=="Second stage"))]
    #random pages
    extract_random_examples(filtered_df,tar_path,out_path,type_of_analysis='standard',n_samples=5)

    #analyze pages with ordering warnings
    #filtered_df=df[df["Warning_ordering"]=="OCR stage, pages with warning:   1/20"]
    filtered_df = df[df["Warning_ordering"].str.contains("OCR", na=False)]
    extract_random_examples(filtered_df,tar_path,out_path,type_of_analysis='ordering',n_samples=5)

    #analyze pages with censoring warnings
    #filtered_df=df[df["Warning_censoring"]=="pages censored with large boxes:   1/20"]
    filtered_df = df[df["Warning_censoring"].str.contains("large", na=False)]
    extract_random_examples(filtered_df,tar_path,out_path,type_of_analysis='censoring',n_samples=5)
    '''list_images = process_and_extract_tar(tar_path, out_path)

    # OpenCV uses BGR, Matplotlib uses RGB
    img_rgb = cv2.cvtColor(list_images[0], cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    plt.title(f"Shape: {img_rgb.shape}")
    plt.axis('off') # Hide the X/Y axes
    plt.show()'''
    #check: Q10 B8Y7L2N5 (first page is an extra page that should not be present); 
    # Q9 A8J0A6M4 (two pages discarded because 0 ocr confidence), B4X4S9G1 (one); 
    # Q11 B0R4C6V7 (not censored); Q9 A4S9H4B1 (not censored); -> solved

def extract_random_examples(filtered_df,source_path,out_path,type_of_analysis='standard',n_samples=5):
    #filter the df for lines with "Warning_censoring"=="pages censored with large boxes:   1/20"
    #get 5 random ids from the filtered df
    random_ids_censoring = filtered_df["e3n_id_hand"].sample(min(n_samples,len(filtered_df)), random_state=42).tolist()

    for id in random_ids_censoring:
        print(f"Processing ID: {id}")
        out_tar_path = os.path.join(out_path,type_of_analysis)

        source_tar_path = os.path.join(source_path,f"{id}.tar")
     
        process_and_extract_tar(source_tar_path, out_tar_path)
def analysis(df):
    #recall that you have duplicate lines (one per file per each id) and you want to keep only one line per id, for example the last one (the one with the largest time)
    final_df=df.copy()
    #keep a single line per unique id
    final_df = final_df.drop_duplicates(subset='e3n_id_hand', keep='last')
    #get the unique values and their count for the columns "Warning_ordering", "Warning_censoring", "status"
    for col in ["Warning_ordering", "Warning_censoring", "status"]:
        print(f"Unique values for {col}:")
        print(final_df[col].value_counts())
        print("\n")
    # get the 5 ids with larger value of the 'time' column
    top_5_time = final_df.nlargest(5, 'time')
    print("Top 5 IDs with largest time:")
    print(top_5_time[['e3n_id_hand', 'time']])
    # get the list of ids with 'status'=='timeout'
    timeout_ids = final_df[final_df['status'] == 'timeout']['e3n_id_hand'].tolist()
    print(f"IDs with status 'timeout': {timeout_ids[:min(10, len(timeout_ids))]}") # Print only the first 10 for brevity
    #return the average time and the number of ids with time>100 over the total
    avg_time = final_df['time'].mean()
    count_time_gt_100 = (final_df['time'] > 100).sum()
    print(f"Average time: {avg_time}")
    print(f"Number of IDs with time > 100: {count_time_gt_100} over {len(final_df)} total IDs")
    return final_df

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

if __name__ == "__main__":
    main()