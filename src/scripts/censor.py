#!/usr/bin/env python3
import argparse
import logging
import os
import shutil
import json

import cv2
#from src.utils.convert_utils import pdf_to_png_images
from src.utils.file_utils import list_subfolders,list_files_with_extension,sort_files_by_page_number,get_page_number
from src.utils.file_utils import get_basename, create_folder, check_name_matching, remove_folder
#from src.utils.xml_parsing import load_xml, iter_boxes, add_attribute_to_boxes
#from src.utils.xml_parsing import save_xml, iter_images, set_box_attribute,get_box_coords
from src.utils.json_parsing import get_attributes_by_page, get_page_list, get_page_dimensions,get_box_coords_json, get_censor_type
from src.utils.feature_extraction import crop_patch, preprocess_alignment_roi, preprocess_roi, preprocess_blank_roi,load_image
from src.utils.feature_extraction import extract_features_from_blank_roi, extract_features_from_roi,censor_image
from src.utils.alignment_utils import page_vote,compute_transformation, compute_misalignment,apply_transformation,enlarge_crop_coords
from src.utils.alignment_utils import plot_rois_on_image_polygons,plot_rois_on_image,plot_both_rois_on_image,template_matching
from src.utils.logging import FileWriter
from PIL import Image
import numpy as np 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SOURCE = "//vms-e34n-databr/2025-handwriting\\vscode\\censor_e3n\\data\\q5_tests" #"Z:\\vscode\\censor_e3n\\data\\q5_tests" # "C:\\Users\\andre\\VsCode\\censoring project\\data\\rimes_tests"

# thresholds
MIN_TO_CHECK_TEMPLATE = 4
THRESHOLD_MATCHING = 0.7
SCALE_FACTOR_MATCHING = 2 
#global vars
mode = 'cv2'

def main():
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)

    skip_checking_1 = args.skip_checking_1
    skip_checking_2 = args.skip_checking_2
    skip_aligning = args.skip_aligning
    save_path = args.save_path
    annotation_path = args.annotation_path
    filled_path = args.filled_path
    enlarge_censor_boxes = args.enlarge_censor_boxes
    save_debug_images = args.save_debug_images
    save_debug_times = args.save_debug_times

    #remove files and folders to generate
    remove_folder(save_path)
    if save_debug_images : remove_folder(os.path.join(SOURCE,'debug'))
    if save_debug_times : remove_folder(os.path.join(SOURCE,'time_logs'))

    logger.debug("Output folder: %s", save_path)
    logger.debug("Annotation folder: %s", annotation_path)
    logger.debug("Filled folder: %s", filled_path)
    logger.debug("Skip checking %s, Skip aligning %s",skip_checking_1,skip_checking_2,skip_aligning)

    annotation_files = list_files_with_extension(annotation_path, "json", recursive=False)
    logger.info("Found %d annotation file(s) in %s", len(annotation_files), annotation_path)
    if not annotation_files:
        logger.warning("No annotation files found. Exiting.")
        return 0

    print(annotation_path)
    annotation_file_names = [get_basename(annotation_file, remove_extension=True) for annotation_file in annotation_files]
    filled_folders = list_subfolders(filled_path, recursive=False)
    filled_folder_names = [get_basename(p, remove_extension=False) for p in filled_folders]
    logger.debug("Filled folder names: %s", len(filled_folder_names))

    annotation_roots, npy_data = load_template_info(annotation_files,annotation_file_names,annotation_path)

    log_path=os.path.join(SOURCE,'time_logs')
    create_folder(log_path, parents=True, exist_ok=True)
    global_time_logger=FileWriter(save_debug_times,
                                    os.path.join(log_path,f"global_time_logger.txt"))
    warning_map=[[] for _ in range(len(filled_folders))]

    # i iterate on the filled_folders (study subjects)
    for j, filled_folder in enumerate(filled_folders): #subject level
        warning_map[j]=[[] for _ in range(len(annotation_files))]
        subj_id=filled_folder_names[j]
        #I load the documents for the ith subject
        documents = list_subfolders(filled_folder, recursive=False) # the document paths for the jth subject
        documents_folder_names = [get_basename(p, remove_extension=False) for p in documents]
        logger.debug("Document folder names for subject %s: %s", j, documents_folder_names)

        # I match them with the annotation file names (will be a more complex function, in this test the names are the same)
        #check that names match
        if check_name_matching(annotation_file_names, documents_folder_names, logger) == 1:
            logger.error(f"Mismatch between annotation files and document folders for subject {j}. Exiting.")
            return 1
        #check that they are sorted in the same way
        assert annotation_file_names == documents_folder_names, "Annotation files and documents folders are not in the same order."
        
        #i can access them by index since they are sorted in the same way
        for i, annotation_file in enumerate(annotation_files): #document level

            doc_path = documents[i] #the file path for the ith document of the jth subject
            doc_files = list_files_with_extension(doc_path, ['png','tif'], recursive=False)
            sorted_files=sort_files_by_page_number(doc_files)

            #load the json file
            root = annotation_roots[i]
            pages_in_annotation = get_page_list(root)
            npy_dict = npy_data[i]
            page_dictionary = [{} for p in pages_in_annotation]

            #iterate on the pages in a document
            for img_id in pages_in_annotation:
                warning_map[j][i].append([0,0])

                page_dictionary[img_id]['img_id']=img_id

                censor_type=get_censor_type(root,img_id) 
                page_dictionary[img_id]['type']=censor_type

                img_name=f'page_{img_id}.png'
                page_dictionary[img_id]['img_name']=img_name

                png_img_path = find_corresponding_file(sorted_files, img_name)
                page_dictionary[img_id]['img_path']=png_img_path

                page_dictionary[img_id]['img_size']=get_page_dimensions(root,img_id)

                page_dictionary[img_id]['template_matches']=0
                page_dictionary[img_id]['shifts']=[(0,0),(0,0)] #(shift_x,shift_y) for first qnd second region
                page_dictionary[img_id]['stored_template']=None # to store the features extracted from the region -> i don't recompute multiple times
                page_dictionary[img_id]['matched_page']=None #initially I have no matches
                page_dictionary[img_id]['align_boxes']=None
                page_dictionary[img_id]['pre_computed_align']=None
                
                if censor_type!='N':
                    img=load_image(png_img_path, mode=mode, verbose=False) #modify code to manage tiff and jpeg if needed
                    page_dictionary[img_id]['img']=img.copy()
                else:
                    page_dictionary[img_id]['img']=None
                
            no_censor_mask = [d.get('type') == 'N' for d in page_dictionary] #mask to index on the pages that we don't need to censor
            p_censor_mask = [d.get('type') == 'P' for d in page_dictionary] #mask to index on the pages that we can censor fast
            censor_mask = [d.get('type') == 'C' for d in page_dictionary] #mask to index on the pages that we need to censor

            #open extra pages if you don't have enough
            N_to_censor = len(p_censor_mask)+len(censor_mask) 
            to_add = MIN_TO_CHECK_TEMPLATE - N_to_censor
            for d in page_dictionary[no_censor_mask]:
                img_id=d['img_id']
                if to_add>0:
                    img=load_image(d['img_path'], mode=mode, verbose=False) #modify code to manage tiff and jpeg if needed
                    page_dictionary[img_id]['img']=img.copy()
                    to_add-=1
                else:
                    break
            
            #perform the check on exactly four pages
            loaded_mask = [d.get('img') != None for d in page_dictionary]
            for page in page_dictionary[loaded_mask][:MIN_TO_CHECK_TEMPLATE]:
                img_id=page['img_id']
                align_boxes, pre_computed_align = get_align_boxes(root,pre_computed,img_id) 
                page_dictionary[img_id]['align_boxes']=align_boxes
                page_dictionary[img_id]['pre_computed_align']=pre_computed_align

                shifts, centers, processed_rois = compute_misalignment(page_dictionary[img_id]['img'], align_boxes, page_dictionary[img_id]['img_size'], 
                                     pre_computed_template=pre_computed_align,scale_factor=SCALE_FACTOR_MATCHING) #recall this functions returns a shift for each good match
                #thus you expect len=2 for the shift variable, instead processed_rois returns all regions
                
                if len(shifts)==2:
                    page_dictionary[img_id]['shifts']=shifts
                    page_dictionary[img_id]['centers'] = centers 
                    page_dictionary[img_id]['template_matches']=1
                    page_dictionary[img_id]['stored_template'] = processed_rois #save the rois so i can re-use them without recomputing
            
            #count the matchign and decide if the test is passed
            matching_mask = [d.get('template_matches') == 1  for d in page_dictionary] #mask to find matching pages
            matching_pages=page_dictionary[matching_mask]
            failed_first_ordering_test=False
            if len(matching_pages)<4:
                failed_first_ordering_test=True
            
            if failed_first_ordering_test:
                for img_id in pages_in_annotation: # i precompute all align_boxes and store all pre_computed aligns
                    page = page_dictionary[img_id]
                    if page[img_id]['align_boxes']:
                        align_boxes=page[img_id]['align_boxes']
                        pre_computed_align=page[img_id]['pre_computed_align']
                    else:
                        align_boxes, pre_computed_align = get_align_boxes(root,pre_computed,img_id) 
                        page_dictionary[img_id]['align_boxes']=align_boxes
                        page_dictionary[img_id]['pre_computed_align']=pre_computed_align
                    
                    # i need to load all the images
                    img=load_image(png_img_path, mode=mode, verbose=False) #modify code to manage tiff and jpeg if needed
                    page_dictionary[img_id]['img']=img.copy()
                
                for img_id in pages_in_annotation:
                    page = page_dictionary[img_id]
                    align_boxes=page[img_id]['align_boxes']

                    for img_id_2 in pages_in_annotation:
                        if img_id_2==img_id and (img_id in pages_in_annotation[loaded_mask]): #skip these that are already checked
                            continue

                        page_2=page_dictionary[img_id_2]
                        pre_computed_align_2=page_2[img_id_2]['pre_computed_align']
                        
                        shifts, centers, processed_rois = compute_misalignment(page['img'], align_boxes, page['img_size'], 
                                     pre_computed_template=pre_computed_align_2,scale_factor=SCALE_FACTOR_MATCHING, pre_computed_rois=page['stored_template'])
                        #i pass both the image and the templates, if the templates has not been yet computed the stored_template entry is none and the function takes care of it
                        if page['stored_template']==None:
                            page_dictionary[img_id]['stored_template'] = processed_rois #so in next steps is not re-computed
                        
                        if len(shifts)==2: #if there is a match save the matching page number and increase the matcher count, also store the shift/centers for this msot recent match
                            page_dictionary[img_id]['shifts']=shifts
                            page_dictionary[img_id]['centers'] = centers 
                            page_dictionary[img_id]['template_matches']+=1
                            page_dictionary[img_id]['stored_template'] = processed_rois #save the rois so i can re-use them without recomputing
                            page_dictionary[img_id]['matched_page']=img_id_2

                problematic_mask = [d.get('template_matches') != 1  for d in page_dictionary] #mask to find the pages that have multiple matches or no matches
                problematic_pages = page_dictionary[problematic_mask]

                if len(problematic_pages)=0: # no problematic pages -> i can reorder based on the matching I found
                else:
                    pass

            else:
                for img_id in pages_in_annotation:
                    page_dictionary[img_id]['matched_page']=img_id #if the test is passed they are orthered correctly -> i match with corresponding index

            for img_id in pages_in_annotation:

                log_path=os.path.join(SOURCE,'time_logs', f"patient_{subj_id}", f"document_{i}")
                create_folder(log_path, parents=True, exist_ok=True)
                image_time_logger=FileWriter(save_debug_times,
                                             os.path.join(log_path,f"time_logger_page_{img_id}.txt"))
                img_name=f'page_{img_id}.png'
                img_size = get_page_dimensions(root,img_id)

                #get censor boxes
                censor_boxes,partial_coverage = get_censor_boxes(root,img_id)
                png_img_path = find_corresponding_file(sorted_files, img_name)
                #load image with cv2 and pre_computed rois
                
                if len(censor_boxes)<=0:
                    logger.debug("Skip image: id=%s, name=%s, size=%s, no censor regions", img_id, img_name, img_size)
                    image_time_logger.call_start('copy_image')
                    copy_image(png_img_path, save_path,subj_id,i,img_id)
                    image_time_logger.call_end('copy_image')
                else:
                    logger.debug("Processing image: id=%s, name=%s, size=%s", img_id, img_name, img_size)
                    #find the corresponding png image in the template folder
                    image_time_logger.call_start('load_image')
                    img=load_image(png_img_path, mode=mode, verbose=False) #modify code to manage tiff and jpeg if needed
                    image_time_logger.call_end('load_image')

                    pre_computed = npy_dict[img_id]
                    #logger.debug("Pre-computed data keys for image %s: %s", img_name, pre_computed)

                    #check if templates are aligned
                    roi_boxes, pre_computed_rois = get_roi_boxes(root,pre_computed,img_id)
                    decision_1=False
                    if not skip_checking_1:
                        decision_1 = page_vote(img, roi_boxes, min_votes=2, template_png=None, pre_computed_rois=pre_computed_rois,logger=image_time_logger)
                        #start logging of all nested function if active
                    if save_debug_images:
                        align_boxes, pre_computed_align = get_align_boxes(root,pre_computed,img_id)
                        debug_boxes=align_boxes+roi_boxes+censor_boxes
                        colors=["red" for i in range(len(align_boxes))]+["green" for i in range(len(roi_boxes))]+["blue" for i in range(len(censor_boxes))]
                        parent_path=os.path.join(SOURCE,'debug', f"patient_{subj_id}", f"document_{i}")#, f"censored_page_{n_page}.png")
                        create_folder(parent_path, parents=True, exist_ok=True)
                        save_debug_path=os.path.join(parent_path, f"{img_id}_original_w_boxes.png")
                        plot_rois_on_image(img, debug_boxes, save_debug_path,colors=colors)
                    
                    decision_2=False
                    if not skip_aligning and not decision_1:
                        image_time_logger.call_start('alignement_and_check',block=True)
                        image_time_logger.call_start('alignement_only')
                        align_boxes, pre_computed_align = get_align_boxes(root,pre_computed,img_id)

                        image_time_logger.call_start('compute_misalignement')
                        shifts, centers = compute_misalignment(img, align_boxes, img_size,scale_factor=2,pre_computed_rois=pre_computed_align)
                        image_time_logger.call_end('compute_misalignement')

                        image_time_logger.call_start('compute_transformation')
                        scale_factor, shift_x, shift_y, angle_degrees,reference = compute_transformation(shifts, centers)
                        image_time_logger.call_end('compute_transformation')
                        #new/template , new-template, new-template
                        if not skip_checking_2:
                            new_roi_boxes = []
                            new_align_boxes = []
                            for coord in roi_boxes: 
                                image_time_logger.call_start('apply_transformation_roi')
                                new_coord = apply_transformation(reference,coord, scale_factor, shift_x, shift_y, angle_degrees, inverse=False)
                                image_time_logger.call_end('apply_transformation_roi')
                                new_roi_boxes.append(new_coord)
                            if save_debug_images:
                                for coord in align_boxes:
                                    new_coord = apply_transformation(reference, coord, scale_factor, shift_x, shift_y, angle_degrees, inverse=False)
                                    new_align_boxes.append(new_coord)
                            
                            image_time_logger.call_end('alignement_only')
                            decision_2 = page_vote(img, new_roi_boxes, min_votes=2, template_png=None, 
                                                   pre_computed_rois=pre_computed_rois,logger=image_time_logger)  
                        image_time_logger.call_end('alignement_and_check',block=True)
                    
                    if decision_1:
                        decision_2 = True

                    image_time_logger.call_start('censoring',block=True)
                    # increase size of censor boxes
                    if enlarge_censor_boxes:
                        image_time_logger.call_start(f'enlarge_{len(censor_boxes)}_censor_regions')
                        new_censor_boxes = []
                        for coord in censor_boxes:
                            new_coord = enlarge_crop_coords(coord, scale_factor=1.2, img_shape=img_size)
                            new_censor_boxes.append(new_coord)
                        censor_boxes = new_censor_boxes
                        image_time_logger.call_end(f'enlarge_{len(censor_boxes)}_censor_regions')
                    #transform censor box regions if documents need aligning
                    if not skip_aligning and not decision_1:
                        new_censor_boxes = []
                        for coord in censor_boxes:
                            image_time_logger.call_start('apply_transformation_censor')
                            new_coord = apply_transformation(reference, coord, scale_factor, shift_x, shift_y, angle_degrees, inverse=False)
                            image_time_logger.call_end('apply_transformation_censor')
                            new_censor_boxes.append(new_coord)
                        censor_boxes=new_censor_boxes
                    
                    if save_debug_images and not decision_1:
                        debug_boxes=new_align_boxes+new_roi_boxes+new_censor_boxes
                        colors=["red" for i in range(len(align_boxes))]+["green" for i in range(len(roi_boxes))]+["blue" for i in range(len(censor_boxes))]
                        parent_path=os.path.join(SOURCE,'debug', f"patient_{subj_id}", f"document_{i}")#, f"censored_page_{n_page}.png")
                        create_folder(parent_path, parents=True, exist_ok=True)
                        save_debug_path=os.path.join(parent_path, f"{img_id}_aligned_w_boxes.png")
                        #plot_rois_on_image(img, debug_boxes, save_path,colors=colors)
                        plot_both_rois_on_image(img, roi_boxes+align_boxes, new_roi_boxes+new_align_boxes, os.path.join(parent_path, f"{img_id}_both_roi_boxes.png"),color_1="red", color_2="green")
                        plot_rois_on_image_polygons(img, debug_boxes, save_debug_path,colors)
                    
                    #save the censored image
                    warning=str(1-int(decision_1))+str(1-int(decision_2)) #true decision becomes 1 which becomes '0' in the warning
                    warning_map[j][i][-1] = [1-int(decision_1), 1-int(decision_2)]
                    save_censored_image(img, censor_boxes, save_path,subj_id,i,img_id,
                                        warning=warning,partial_coverage=partial_coverage,
                                        thickness_pct=0.2, spacing_mult=0.5,logger=image_time_logger)
                    image_time_logger.call_end('censoring',block=True)
                    image_time_logger.call_end('complete_process',block=True)
    #save warning_map as npy file
    warning_map_path=os.path.join(save_path, "warning_map.npy")
    np.save(warning_map_path, np.array(warning_map, dtype=object))
    '''warning_map = np.load(warning_map_path, allow_pickle=True)'''
    logger.debug("Warning map: %s", warning_map)
    logger.info("Conversion finished")
    global_time_logger.call_end('complete_process')
    return 0

# Assistance function ---------------------------------------------------------------------------------------------------------------------

#i can load all the pre-computed data at once to spare time; since i won't re-open the files every time (shouldn't be intensive on memory) 
def load_template_info(annotation_files,annotation_file_names,annotation_path):
    annotation_roots=[]
    path_npy=os.path.join(annotation_path,"precomputed_features")
    npy_files=list_files_with_extension(path_npy, "npy", recursive=False)
    npy_file_names = [get_basename(npy_file, remove_extension=True) for npy_file in npy_files]

    # I match them with the annotation file names (will be a more complex function, in this test the names are the same)
    #check that names match
    if check_name_matching(annotation_file_names, npy_file_names, logger) == 1:
        logger.error(f"Mismatch between annotation files and numpy files . Exiting.")
        return 1
    #check that they are sorted in the same way
    assert annotation_file_names == npy_file_names, "Annotation files and numpy files are not in the same order."

    #i load the data i will use (xml first)
    for i, annotation_file in enumerate(annotation_files):
        #_ ,root = load_xml(annotation_file)
        with open(annotation_file, 'r') as f: root = json.load(f)
        annotation_roots.append(root)
    #then numpy arrays
    npy_data=[]
    for i, npy_file in enumerate(npy_files):
        data_dict = np.load(npy_file, allow_pickle=True).item()
        npy_data.append(data_dict)
    return annotation_roots, npy_data

def get_roi_boxes(root,pre_computed,img_id):
    roi_boxes = []
    pre_computed_rois = []
    bb_list=get_attributes_by_page(root, img_id)
    img_size = get_page_dimensions(root,img_id)

    i=0
    for box in bb_list:
        box_coords=get_box_coords_json(box,img_size)
        if box['label'] == "roi" and box['sub_attribute']=='standard':
            roi_boxes.append(box_coords)
            pre_computed_rois.append(pre_computed[i])
        elif box['label'] == "roi" and box['sub_attribute']=="blank":
            blank_box=box_coords
            pre_computed_blank=pre_computed[i]
            #print("Found blank box")
        i+=1
    #print(i)
    roi_boxes.append(blank_box) # i put the blank box as the last one
    pre_computed_rois.append(pre_computed_blank)
    return roi_boxes, pre_computed_rois

def get_align_boxes(root,pre_computed,img_id):
    roi_boxes = []
    pre_computed_rois = []
    bb_list=get_attributes_by_page(root, img_id)
    img_size = get_page_dimensions(root,img_id)

    i=0
    for box in bb_list:
        box_coords=get_box_coords_json(box,img_size)
        if box['sub_attribute'] == "align":
            roi_boxes.append(box_coords)
            pre_computed_rois.append(pre_computed[i])
        i+=1
    return roi_boxes, pre_computed_rois

def get_censor_boxes(root,img_id):
    roi_boxes = []
    partial_coverage=[]
    bb_list=get_attributes_by_page(root, img_id)
    img_size = get_page_dimensions(root,img_id)

    i=0
    for box in bb_list:
        box_coords=get_box_coords_json(box,img_size)
        if box['label'] == "censor":
            roi_boxes.append(box_coords)
            if box['sub_attribute'] == "partial":
                partial_coverage.append(True)
            else:
                partial_coverage.append(False)
        i+=1
    return roi_boxes, partial_coverage

def find_corresponding_file(sorted_files, img_name):
    index=get_page_number(img_name)
    if index <= len(sorted_files):
        return sorted_files[index-1]
    return None

def save_censored_image(img, censor_boxes, save_path,n_p,n_doc,n_page,warning='00', verbose=False,partial_coverage=None,logger=None,**kwargs):
    parent_path=os.path.join(save_path, f"patient_{n_p}", f"document_{n_doc}")#, f"censored_page_{n_page}.png")
    create_folder(parent_path, parents=True, exist_ok=True)
    save_path=os.path.join(parent_path, f"censored_page_w{warning}_{n_page}.png")
    censored_img = censor_image(img, censor_boxes, verbose=verbose,partial_coverage=partial_coverage,logger=logger,**kwargs)
    logger and logger.call_start(f'writing_to_memory')
    cv2.imwrite(str(save_path), censored_img)
    logger and logger.call_end(f'writing_to_memory')

def copy_image(src_path, dest_folder,n_p,n_doc,n_page):
    """
    Copy an image file from src_path to the destination folder.
    
    Parameters:
        src_path (str): Full path to the source image file.
        dest_folder (str): Path to the destination folder.
    """
    # Build the destination file path
    dest_path = os.path.join(dest_folder, f"patient_{n_p}", f"document_{n_doc}")
    create_folder(dest_path, parents=True, exist_ok=True)
    save_path=os.path.join(dest_path, f"censored_page_w00_{n_page}.png")

    shutil.copy2(src_path, save_path)  # copy2 preserves metadata

# main blocks --------------------------------------------------------------------------------

# parsing
def parse_args():
    """Handle command-line arguments."""
    parser = argparse.ArgumentParser(description="Script to convert PDF template pages to PNG images.")
    parser.add_argument(
        "-a", "--annotation_path",
        default=SOURCE+"\\annotazioni",
        help="Directory with the annotation files from cvat for each image",
    )
    parser.add_argument(
        "-f", "--filled_path",
        #default=SOURCE+"\\filled\\rimes",
        default=SOURCE+"\\filled",
        help="Directory with the files to censor",
    )
    parser.add_argument(
        "-s", "--save_path",
        default=SOURCE+"\\censored",
        help="Directory where I save the final censored files",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--skip_checking_1",
        action="store_true",
        help="Skip checking for matching annotation and numpy files",
    )
    parser.add_argument(
        "--skip_checking_2",
        action="store_true",
        help="Skip checking for matching annotation and numpy files",
    )
    parser.add_argument(
        "--skip_aligning",
        action="store_true",
        help="Skip alignment for matching annotation and numpy files",
    )
    parser.add_argument(
        "--enlarge_censor_boxes",
        action="store_true",
        help="Enlarge censor boxes before applying censorship",
    )
    parser.add_argument(
        "--save_debug_images",
        action="store_true",
        help="Enlarge censor boxes before applying censorship",
    )

    parser.add_argument(
        "--save_debug_times",
        action="store_true",
        help="Enlarge censor boxes before applying censorship",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()