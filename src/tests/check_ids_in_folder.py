#!/usr/bin/env python3
import argparse
import logging
import os
from time import perf_counter
from pathlib import Path
import sys
import shutil

import pandas as pd

from src.utils.convert_utils import pdf_to_images,save_as_is, process_pdf_files
from src.utils.file_utils import list_files_with_extension, get_basename, create_folder,list_subfolders,get_page_number


logging.basicConfig( 
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

METADATA_PATH = "//smb-recherche-s1.prod-powerscale.intra.igr.fr/E34N_HANDWRITING$\\ref_pdf_Qx"
OUTPUT_PATH = "//vms-e34n-databr/2025-handwriting\\winscp_logs"
PDF_PATH = "//smb-recherche-s1.prod-powerscale.intra.igr.fr/E34N_HANDWRITING$\\censored_files"

def parse_args():
    """Handle command-line arguments."""
    parser = argparse.ArgumentParser(description="Script to convert PDF template pages to PNG images.")
    parser.add_argument(
        "-m", "--metadata_path",
        default=METADATA_PATH,
        help="Path to the PDF file to convert",
    )
    parser.add_argument(
        "-o", "--output_path",
        default=OUTPUT_PATH,
        help="Path to the PDF file to convert",
    )
    parser.add_argument(
        "-f", "--pdf_path",
        default=PDF_PATH,
        help="Path to the PDF file to convert",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def remove_after_first_dot(text):
    """
    Removes all characters following the first '.' in a string.
    If no dot is found, returns the original string.
    """
    return text.split('.', 1)[0]

def preprocess_df(df_main,filename_col,id_col,used_col_name='Used',warning_ordering_col_name='Warning_ordering',warning_censoring_col_name='Warning_censoring'):
    df=df_main.copy()
    # 1. Drop lines with at least one missing values
    df = df.dropna()

    unique_ids_before = df[id_col].nunique()
    # 2. Remove file extensions from filenames (remove everything after the first point)
    df[filename_col] = df[filename_col].str.rsplit('.', n=1).str[0]

    # 3. Remove lines with filenames that are associated with more than one ID
    fname_id_counts = df.groupby(filename_col)[id_col].nunique()
    multi_id_filenames = fname_id_counts[fname_id_counts > 1].index.tolist()
    df = df[~df[filename_col].isin(multi_id_filenames)]
    print(f"Removed filenames because associated to multiple ids: {len(multi_id_filenames)}")
    
    length_before = len(df)
    df.drop_duplicates(inplace=True) #by default it keep the first occurrence
    length_after = len(df)
    print(f"Before eliminating duplicates the row length is {length_before} after it is {length_after} -> {length_before-length_after} rows were eliminated")

    unique_ids_after = df[id_col].nunique()
    print(f"Unique ids before: {unique_ids_before} after: {unique_ids_after} -> {unique_ids_before-unique_ids_after} unique ids were eliminated")

    df = df.sort_values(id_col).reset_index(drop=True)
    #add columns
    df[used_col_name] = False
    df[warning_ordering_col_name] =''
    df[warning_censoring_col_name] =''

    return df



def main():
    questionnaire = "10"
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)

    metadata_path = args.metadata_path 
    output_dir = args.output_path 
    pdf_path = args.pdf_path
    os.makedirs(output_dir, exist_ok=True) # Create folder if it doesn't exist

    #questionnaire_folder_path = os.path.join(pdf_path,f"Q{questionnaire}",f"archived_{questionnaire}")
    questionnaire_folder_path = os.path.join(pdf_path,f"Q{questionnaire}",f"images")
    #questionnaire_folder_path = os.path.join(pdf_path,f"Q{questionnaire}",f"{questionnaire}")
    metadata_file_path = os.path.join(metadata_path, f"ref_pdf_Q{questionnaire}.csv")
    report_file_path = os.path.join(output_dir, f"audit_report_Q{questionnaire}.txt")

    raw_metadata = pd.read_csv(metadata_file_path)
    raw_metadata = preprocess_df(raw_metadata,'object_name','e3n_id_hand',used_col_name='Used',warning_ordering_col_name='Warning_ordering',warning_censoring_col_name='Warning_censoring')
    metadata_ids = set(raw_metadata['e3n_id_hand'].to_list())

    if not os.path.exists(questionnaire_folder_path):
        print("wrong pathfile")
        return 0

    count=0
    with open(report_file_path, 'w') as f:
        for id in metadata_ids:
            if not os.path.exists(os.path.join(questionnaire_folder_path, f"{id}.tar")):
                f.write(f"{id} not found\n")
                count+=1
        f.write(f"total ids in metadata: {len(metadata_ids)}\n")
        f.write(f"\nTotal missing IDs: {count}\n")

    return 0

if __name__ == "__main__":
    main()