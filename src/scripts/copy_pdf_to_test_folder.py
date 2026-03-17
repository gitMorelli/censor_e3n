#!/usr/bin/env python3
import argparse
import logging
import os
from time import perf_counter
from pathlib import Path
import sys
import copy
import shutil

import pandas as pd
import numpy as np
import cv2
from pympler import asizeof
import psutil

from src.utils.file_utils import get_basename, create_folder, remove_folder

from src.utils.logging import FileWriter, initialize_logger

logging.basicConfig( 
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PDF_LOAD_PATH="//smb-recherche-s1.prod-powerscale.intra.igr.fr/E34N_HANDWRITING$\\Fichiers"#additional"#100263_template"
CSV_LOAD_PATH="//smb-recherche-s1.prod-powerscale.intra.igr.fr/E34N_HANDWRITING$\\ref_pdf_Qx"#additional"#100263_template"
SAVE_PATH="//vms-e34n-databr/2025-handwriting\\data\\test_parallelization"#additional"#100263_template" 

# other variables
QUESTIONNAIRE = "5"
N_ids = 100
ID_COL = 'e3n_id_hand'
FILENAME_COL = 'object_name'
#SAVE_ANNOTATED_TEMPLATES=True
#additional cols
USED_COL = 'Used'
WARNING_ORDERING_COL_NAME = 'Warning_ordering'
WARNING_CENSORING_COL_NAME = 'Warning_censoring'


def main():
    args = parse_args()

    ####### INITIALIZING PATHS #########
    pdf_load_path = args.pdf_load_path
    csv_load_path = args.csv_load_path
    save_path = args.save_path

    #folder for the csv table
    updated_csv_paths = os.path.join(save_path,"ref_pdf")
    #folder for the global logging
    log_path=os.path.join(save_path,'logs')
    #load path for the pdfs
    questionnairres_log_path=os.path.join(pdf_load_path, f"Q{QUESTIONNAIRE}")
    # folder in which i save the pdfs
    pdf_save_path=os.path.join(save_path,f"Q{QUESTIONNAIRE}")


    ####### CLEANING FOLDERS (only in debugging) ###########
    if args.delete_previous_results:
        if os.path.exists(updated_csv_paths):
            remove_folder(updated_csv_paths)
        if os.path.exists(log_path):
            remove_folder(log_path)
        if os.path.exists(pdf_save_path):
            remove_folder(pdf_save_path)

    ########## Initializing loggers ##################
    #file logger (global logger for the execution)
    create_folder(log_path, parents=True, exist_ok=True)
    file_logger=FileWriter(enabled=args.verbose,path=os.path.join(log_path,f"global_logger.txt"))

    ########## LOAD the DATFRAME with the subject ids and filenames ###############
    csv_modified_path = os.path.join(updated_csv_paths,f"updated_ref_pdf_Q{QUESTIONNAIRE}.csv")
    if os.path.exists(csv_modified_path)==False: #if the csv has not been preprocessed yet
        df = preprocess_df(os.path.join(csv_load_path,f"ref_pdf_Q{QUESTIONNAIRE}.csv"),FILENAME_COL, ID_COL,USED_COL,WARNING_ORDERING_COL_NAME,WARNING_CENSORING_COL_NAME)
        create_folder(updated_csv_paths, parents=True, exist_ok=True)
        df.to_csv(csv_modified_path)
    
    df = load_preprocessed_df(csv_modified_path,used_col_name=USED_COL,id_col_name=ID_COL) #load the preprocessed df
    file_logger.write(df.head(10).to_string()) #log the first 10 lines of the df to check it is correct

    count=0
    #for unique_id, group in filtered_df.groupby(ID_COL):
    for unique_id, group in df.groupby(ID_COL):
        count+=1
        if count>N_ids:
            break

        #### LOAD filenames for selected ID #####
        filenames = group[FILENAME_COL].tolist()
        # i sort the filenames by name (expected page ordering is absed on alphabetical ordering)
        filenames.sort()  
        #checks both for .pdf and for .tif.pdf
        pdf_paths = get_file_paths(filenames,questionnairres_log_path,file_logger) 
        file_logger.write(filenames)

        #copy each file to the test folder (keep the same name)
        for pdf_path in pdf_paths:
            if pdf_path is not None:
                filename = os.path.basename(pdf_path)
                save_path_pdf = os.path.join(pdf_save_path,filename)
                create_folder(os.path.dirname(save_path_pdf), parents=True, exist_ok=True)
                shutil.copy2(pdf_path, save_path_pdf)
                #file_logger.write(f"Copied {pdf_path} to {save_path_pdf}")
            else:
                file_logger.write(f"File path for {filenames} is None, skipping copying.")
        #update used column in the dataframe for the selected ID
        df.loc[df[ID_COL]==unique_id, USED_COL] = True
    #save the df with the updated used column
    df.to_csv(csv_modified_path, index=False)



def get_file_paths(filenames,pdf_load_path,logger):
    file_paths = []

    def get_filepath(path):
        if os.path.exists(path):
            return path
        else:
            #try adding .tif before the extension 
            # (eg if the path is /folder/doc_5_page_1.pdf it will try /folder/doc_5_page_1.tif.pdf)
            base, ext = os.path.splitext(path)
            new_path = base + '.tif' + ext
            if os.path.exists(new_path):
                return new_path
            else:
                return logger.write(f"Neither {path} nor {new_path} exist.")

    for filename in filenames:
        try:
            # check both for file.pdf and for file.tif.pdf
            file_path = get_filepath(os.path.join(pdf_load_path, filename+'.pdf'))
            file_paths.append(file_path)
        except FileNotFoundError as e:
            logger.warning(str(e))
            continue
    return file_paths

def load_preprocessed_df(file_path,id_subset = None, used_col_name='Used',id_col_name='e3n_id_hand'):
    df=pd.read_csv(file_path)
    #select only the lines with used=False
    df = df[df[used_col_name]==False] 
    #Select a subset of ids if provided (eg PD subjects)
    if id_subset is not None:
        df = df[df[id_col_name].isin(id_subset)]

    return df

def preprocess_df(source_path,filename_col,id_col,used_col_name='Used',warning_ordering_col_name='Warning_ordering',warning_censoring_col_name='Warning_censoring'):
    df=pd.read_csv(source_path)
    log_filename = 'log_'+get_basename(source_path,remove_extension=True).split('_')[-1]
    log_path = os.path.join(os.path.dirname(source_path),log_filename+'.txt')
    def log(message):
        # 'a' (append) creates the file if it doesn't exist
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(str(message) + "\n")

    # 1. Drop lines with at least one missing values
    df = df.dropna()

    # 2. Remove file extensions from filenames (remove everything after the first point)
    df[filename_col] = df[filename_col].str.rsplit('.', n=1).str[0]

    # 3. Remove lines with filenames that are associated with more than one ID
    fname_id_counts = df.groupby(filename_col)[id_col].nunique()
    multi_id_filenames = fname_id_counts[fname_id_counts > 1].index.tolist()
    df = df[~df[filename_col].isin(multi_id_filenames)]
    log(f"Removed these filenames because associated to multiple ids: {multi_id_filenames}")
    
    length_before = len(df)
    df.drop_duplicates(inplace=True) #by default it keep the first occurrence
    length_after = len(df)
    log(f"Before eliminating duplicates the row length is {length_before} after it is {length_after} -> {length_before-length_after} rows were eliminated")

    df = df.sort_values(id_col).reset_index(drop=True)
    #add columns
    df[used_col_name] = False
    df[warning_ordering_col_name] =''
    df[warning_censoring_col_name] =''

    return df

def parse_args():
    """Handle command-line arguments."""
    parser = argparse.ArgumentParser(description="Script to convert PDF template pages to PNG images.")
    parser.add_argument(
        "-l", "--pdf_load_path",
        default=PDF_LOAD_PATH,
        help="Path to the PDF file to convert",
    )
    parser.add_argument(
        "-c", "--csv_load_path",
        default=CSV_LOAD_PATH,
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
    )

    parser.add_argument(
        "--save_debug_times",
        action="store_true",
        help="Enlarge censor boxes before applying censorship",
    )

    parser.add_argument(
        "-d",
        "--delete_previous_results",
        action="store_true",
        help="Delete the results from the prvious directories to test the pipeline from scratch",
    )

    parser.add_argument(
        "-i",
        "--save_debug_images",
        action="store_true",
        help="",
    )
    return parser.parse_args()

if __name__ == "__main__":
    main()