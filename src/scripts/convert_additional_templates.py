#!/usr/bin/env python3
import argparse
import logging
import os
from src.utils.convert_utils import pdf_to_images,save_as_is,extract_images
from src.utils.file_utils import list_files_with_extension, get_basename, create_folder,list_subfolders,get_page_number
from time import perf_counter
from pathlib import Path

logging.basicConfig( 
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

TEMPLATES_PATH="//vms-e34n-databr/2025-handwriting\\data\\e3n_templates_pdf\\100263_template"#additional"#100263_template"
SAVE_PATH="//vms-e34n-databr/2025-handwriting\\data\\e3n_templates_png\\100263_template"#additional"#100263_template" 

def parse_args():
    """Handle command-line arguments."""
    parser = argparse.ArgumentParser(description="Script to convert PDF template pages to PNG images.")
    parser.add_argument(
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
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)

    templates_path = args.folder_path
    save_path = args.save_path
    pdf_paths = list_subfolders(templates_path)

    for pdf_path in pdf_paths: #iterate on the folders Q_1,Q_2,..
        folder_name = Path(pdf_path).name
        pdf_files = list_files_with_extension(pdf_path, ["pdf",'tif'], recursive=False)
        logger.info("Found %d PDF file(s) in %s", len(pdf_files), pdf_path)
        if not pdf_files:
            logger.warning("No PDF files found. Exiting.")
            return 0

        save_folder=save_path+'\\'+folder_name
        n_template=get_page_number(pdf_path) #get the number of the questionnaire
        process_pdf_files(n_template,pdf_files,save_folder)

    logger.info("Conversion finished")
    return 0


def process_pdf_files(n_quest,pdf_files,save_path):
    #num_files=len(pdf_files)
    group_1= []#1,2,3,5,6,7,9] #in this group the templates are saved as separate pdf files, each a tiff image
    group_2=  []#8,10,11,12,13] #in this group the templates are saved as single pdf with all the pages
    group_3 = [4]#4] #come 2 ma in ordine inverso
    if n_quest in group_1:
        #_t0 = perf_counter()
        for i, pdf_file in enumerate(pdf_files): #iterate on the pdf files in Q_i
            file_name = get_basename(pdf_file,remove_extension=True)
            create_folder(save_path, parents=True, exist_ok=True)
            out_path = os.path.join(save_path, file_name)
            images_data,doc=extract_images(pdf_file)
            save_as_is(doc,0,images_data,out_path) #i always have a single image
        #_t1 = perf_counter()
        #print(f"time={( _t1 - _t0 ):0.6f}s")
    elif n_quest in group_2:
        for i, pdf_file in enumerate(pdf_files): #iterate on the pdf files in Q_i
            sub_folder_name=get_basename(pdf_file,remove_extension=True)
            doc_path = os.path.join(save_path, sub_folder_name)
            images_data,doc=extract_images(pdf_file)
            create_folder(doc_path, parents=True, exist_ok=True)
            for j, image in enumerate(images_data):
                out_path = os.path.join(doc_path, f"page_{j+1}")
                save_as_is(doc,j,images_data,out_path) #i always have a single image
    elif n_quest in group_3:
        for i, pdf_file in enumerate(pdf_files): #iterate on the pdf files in Q_i
            sub_folder_name=get_basename(pdf_file,remove_extension=True)
            doc_path = os.path.join(save_path, sub_folder_name)
            images_data,doc=extract_images(pdf_file)
            create_folder(doc_path, parents=True, exist_ok=True)

            for j in range(len(images_data)):
                out_path = os.path.join(doc_path, f"page_{j+1}")
                save_as_is(doc,len(images_data)-j-1,images_data,out_path) #i always have a single image

    return 0

if __name__ == "__main__":
    main()