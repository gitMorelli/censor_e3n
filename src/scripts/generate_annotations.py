#!/usr/bin/env python3
import argparse
import logging
import os
import json

from src.utils.file_utils import list_subfolders,list_files_with_extension
from src.utils.file_utils import get_basename, create_folder, check_name_matching
#from src.utils.xml_parsing import load_xml, iter_boxes, add_attribute_to_boxes
#from src.utils.xml_parsing import save_xml, iter_images, set_box_attribute,get_box_coords
from src.utils.json_parsing import get_attributes_by_page, get_page_list, get_page_dimensions, get_box_coords_json
from src.utils.feature_extraction import crop_patch, preprocess_alignment_roi, preprocess_roi, preprocess_blank_roi,load_image
from src.utils.feature_extraction import extract_features_from_blank_roi, extract_features_from_roi, extract_features_from_text_region, preprocess_text_region
from src.utils.feature_extraction import extract_features_from_page, preprocess_page
from PIL import Image
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SOURCE = "//vms-e34n-databr/2025-handwriting\\vscode\\censor_e3n\\data\\q5_tests_ordering\\"
#"//vms-e34n-databr/2025-handwriting\\vscode\\censor_e3n\\data\\q5_tests\\" # "Z:\\vscode\\censor_e3n\\data\\q5_tests\\" #C:\\Users\\andre\\VsCode\\censoring project\\data\\rimes_tests\\
CROP_PATCH_PCTG = 0.02
OCR_PSM=6


def parse_args():
    """Handle command-line arguments."""
    parser = argparse.ArgumentParser(description="Script to convert PDF template pages to PNG images.")
    parser.add_argument(
        "-t", "--template_path",
        default=SOURCE+"templates",
        help="Path to the template files, saved as folders containing PNG images",
    )
    parser.add_argument(
        "-a", "--annotation_path",
        default=SOURCE+"annotazioni",
        help="Directory with the annotation files from cvat for each image",
    )
    parser.add_argument(
        "-s", "--save_path",
        default=SOURCE+"annotazioni\\precomputed_features",
        help="Directory where I save the final annotation files",
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

    template_path = args.template_path
    save_path = args.save_path
    annotation_path = args.annotation_path

    logger.info("Starting PDF -> PNG conversion")
    logger.debug("Input folder: %s", template_path)
    logger.debug("Output folder: %s", save_path)
    logger.debug("Annotation folder: %s", annotation_path)

    annotation_files = list_files_with_extension(annotation_path, "json", recursive=False)
    logger.info("Found %d annotation file(s) in %s", len(annotation_files), annotation_path)
    if not annotation_files:
        logger.warning("No annotation files found. Exiting.")
        return 0

    annotation_file_names = [get_basename(annotation_file, remove_extension=True) for annotation_file in annotation_files]
    template_folders = list_subfolders(template_path, recursive=False)
    template_folder_names = [get_basename(p, remove_extension=False) for p in template_folders]
    logger.debug("Template folder names: %s", template_folder_names)

    #check that names match
    if check_name_matching(annotation_file_names, template_folder_names, logger) == 1:
        logger.error("Mismatch between annotation files and template folders. Exiting.")
        return 1
    #check that they are sorted in the same way
    assert annotation_file_names == template_folder_names, "Annotation files and template folders are not in the same order."

    #i can access them by index since they are sorted in the same way
    for i, annotation_file in enumerate(annotation_files):
        logger.info("Processing file %d/%d: %s", i + 1, len(annotation_files), annotation_file)
        doc_path = template_folders[i]
        #pages = list_files_with_extension(doc_path, "png", recursive=False)
        #load the json file
        with open(annotation_file, 'r') as f: json_data = json.load(f)
        pages_in_annotation = get_page_list(json_data)
        #iterate on the images in the annotation file (page index)
        data_dict = {}
        for img_id in pages_in_annotation:
            data_dict[img_id]=[]
            img_name=f'page_{img_id}.png'
            img_size = get_page_dimensions(json_data,img_id)
            logger.debug("Processing image: id=%s, name=%s, size=%s", img_id, img_name, img_size)
            #find the corresponding png image in the template folder
            png_img_path = os.path.join(doc_path, img_name)
            if not os.path.exists(png_img_path):
                logger.error("PNG image not found for annotation image %s: expected at %s", img_name, png_img_path)
                continue
            #load image with cv2
            mode="cv2"
            img=load_image(png_img_path, mode=mode, verbose=False)
            bb_list=get_attributes_by_page(json_data, img_id)


            #iterate on the selected boxes and precompute features
            for box in bb_list:
                #get coordinates
                box_coords=get_box_coords_json(box,img_size)
                #check what kind of box it is
                #i extract and save features for roi, roi-blank, and align boxes
                pre_comp = None
                if box['sub_attribute']=='align':
                    pre_comp = {'full':preprocess_alignment_roi(img, box_coords, mode=mode, verbose=False)}
                elif box['sub_attribute']=='blank':
                    patch = preprocess_blank_roi(img, box_coords, mode=mode, verbose=False)
                    pre_comp = extract_features_from_blank_roi(patch, mode=mode, verbose=False,to_compute=['cc','n_black'])
                elif box['sub_attribute']=='standard' and box['label']=='roi':
                    patch = preprocess_roi(img, box_coords, mode=mode, verbose=False)
                    pre_comp = extract_features_from_roi(patch, mode=mode, 
                                                        verbose=False,to_compute=['crc32','dct_phash', 'ncc','edge_iou','profile'])
                elif box['sub_attribute']=='text':
                    patch = preprocess_text_region(img, box_coords, mode=mode, verbose=False)
                    pre_comp = extract_features_from_text_region(patch, mode=mode, 
                                                        verbose=True,psm=OCR_PSM)
                data_dict[img_id].append(pre_comp)
            
            #precompute features for the whole page
            preprocessed_img = preprocess_page(img)
            pre_comp = extract_features_from_page(preprocessed_img, mode=mode, verbose=False,to_compute=['page_phash'],border_crop_pct=CROP_PATCH_PCTG)
            data_dict[img_id].append(pre_comp) #i add as -1 element, recall when you perform the censoring

        #save data_dict as npy file
        save_folder = create_folder(save_path, parents=True, exist_ok=True)
        save_file_path = os.path.join(save_folder, f"{annotation_file_names[i]}.npy")
        np.save(save_file_path, data_dict)

    logger.info("Conversion finished")
    return 0


if __name__ == "__main__":
    main()