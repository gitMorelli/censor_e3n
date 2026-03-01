from pdf2image import convert_from_path
import fitz  # PyMuPDF
import os
import numpy as np
import cv2

#from src.utils.convert_utils import pdf_to_png_images
from src.utils.file_utils import list_subfolders,list_files_with_extension,sort_files_by_page_number,get_page_number, load_template_info
from src.utils.file_utils import get_basename, create_folder, check_name_matching, remove_folder, load_annotation_tree, load_subjects_tree

#from src.utils.xml_parsing import load_xml, iter_boxes, add_attribute_to_boxes
#from src.utils.xml_parsing import save_xml, iter_images, set_box_attribute,get_box_coords

from src.utils.json_parsing import get_attributes_by_page, get_page_list, get_page_dimensions,get_box_coords_json, get_censor_type
from src.utils.json_parsing import get_align_boxes, get_ocr_boxes, get_roi_boxes, get_censor_boxes

from src.utils.feature_extraction import crop_patch, preprocess_alignment_roi, preprocess_roi, preprocess_blank_roi,load_image
from src.utils.feature_extraction import extract_features_from_blank_roi, extract_features_from_roi,censor_image
from src.utils.feature_extraction import extract_features_from_page, preprocess_page, extract_features_from_text_region, preprocess_text_region

from src.utils.alignment_utils import page_vote,compute_transformation, compute_misalignment,apply_transformation,enlarge_crop_coords
from src.utils.alignment_utils import plot_rois_on_image_polygons,plot_rois_on_image,plot_both_rois_on_image,template_matching

from src.utils.logging import FileWriter, initialize_logger

from src.utils.matching_utils import update_phash_matches, match_pages_phash, check_matching_correspondence, pre_load_images_to_censor, pre_load_image_properties
from src.utils.matching_utils import compare_pages_same_section, match_pages_text, initialize_sorting_dictionaries, pre_load_selected_templates, perform_template_matching
from src.utils.matching_utils import perform_phash_matching, perform_ocr_matching, ordering_scheme_base

from src.utils.censor_utils import save_as_is_no_censoring, save_original_w_boxes, get_transformation_to_match_to_template, apply_transformation_to_boxes
from src.utils.censor_utils import enlarge_censor_regions, save_pre_post_boxes, save_censored_image, generate_warning_string, censor_page_base


POPPLER_PATH = "\\vms-e34n-databr\\2025-handwriting\\programs\\Release-25.11.0-0\\poppler-25.11.0\\Library\\bin" #"Z:\\programs\\Release-25.11.0-0\\poppler-25.11.0\\Library\\bin" #"C:\\Program Files\\poppler-25.11.0\\Library\\bin"

def pdf_to_images(pdf_path):
    """Convert PDF pages to PNG images.

    Args:
        args: Command-line arguments containing the PDF file path and poppler path.

    Returns:
        List of PNG images converted from the PDF.
    """
    images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
    return images

def extract_images(pdf_path): 
    ''' given a pdf file it returns the info needed to extract the image for each image'''
    doc = fitz.open(pdf_path)
    images_data=[]
    for page_index in range(len(doc)):
        page = doc[page_index]
        image_data = page.get_images(full=True)  
        images_data.append(image_data)
    return images_data,doc

def save_as_is(doc,i,images_data,out_path, return_image = False):
    # get list of images for this specific page
    page_images = images_data[i]

    # take first image on this page
    img = page_images[0]
    
    xref = img[0]
    base_image = doc.extract_image(xref)
    image_bytes = base_image["image"]
    ext = base_image["ext"]
    if return_image:
        # Convert bytes to a numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        # Decode the array into an OpenCV image (BGR format)
        cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return cv_img
    else:
        out_path = out_path+f".{ext}"
        with open(out_path, "wb") as f:
            f.write(image_bytes)
        return 0


def process_pdf_files(n_quest,pdf_files,save_path, save=True, groups = None):
    ''' if save is false it returns the list of images instead of saving the pngs to memory
    If groups is provided it overrides the specification of the groups in the function'''
    #num_files=len(pdf_files)
    if groups:
        group_1= groups[0]
        group_2=  groups[1]
        group_3 = groups[2]
    else:
        group_0= ["1","2","3","6","7","9"] #in this group the templates are saved as separate pdf files
        group_1= ["5"] #in this group the templates are saved as separate pdf files, each a tiff image and the order should be reversed
        group_2=  ["8","10","11","12","13"] #in this group the templates are saved as single pdf with all the pages
        group_3 = ["4"]
    images_list = []
    if n_quest in group_0:
        #_t0 = perf_counter()
        for i, pdf_file in enumerate(pdf_files): #iterate on the pdf files in Q_i
            images_data,doc=extract_images(pdf_file)
            if save:
                file_name = get_basename(pdf_file,remove_extension=True)
                create_folder(save_path, parents=True, exist_ok=True)
                out_path = os.path.join(save_path, file_name)
                save_as_is(doc,0,images_data,out_path) #i always have a single image
            else:
                images_list.append( save_as_is(doc,0,images_data,None,return_image=True) )
    elif n_quest in group_1:
        #_t0 = perf_counter()
        for i, pdf_file in enumerate(reversed(pdf_files)): #iterate on the pdf files in Q_i
            images_data,doc=extract_images(pdf_file)
            if save:
                file_name = get_basename(pdf_file,remove_extension=True)
                create_folder(save_path, parents=True, exist_ok=True)
                out_path = os.path.join(save_path, file_name)
                save_as_is(doc,0,images_data,out_path) #i always have a single image
            else:
                images_list.append( save_as_is(doc,0,images_data,None,return_image=True) ) 
        #_t1 = perf_counter()
        #print(f"time={( _t1 - _t0 ):0.6f}s")
    elif n_quest in group_3:
        for i, pdf_file in enumerate(pdf_files): #iterate on the pdf files in Q_i
            images_data,doc=extract_images(pdf_file)
            if save:
                sub_folder_name=get_basename(pdf_file,remove_extension=True)
                doc_path = os.path.join(save_path, sub_folder_name)
                create_folder(doc_path, parents=True, exist_ok=True)
            for j, image in enumerate(images_data):
                if save:
                    out_path = os.path.join(doc_path, f"page_{j+1}")
                    save_as_is(doc,len(images_data)-j-1,images_data,out_path) #i always have a single image
                else:
                    images_list.append( save_as_is(doc,len(images_data)-j-1,images_data,None,return_image=True) ) 
    else: #if it is not in the outher groups is in group_2 or we don't know. If we don't know we extract everything without reordering
        for i, pdf_file in enumerate(pdf_files): #iterate on the pdf files in Q_i
            images_data,doc=extract_images(pdf_file)
            if save:
                file_name = get_basename(pdf_file,remove_extension=True)
                #sub_folder_name=get_basename(pdf_file,remove_extension=True)
                #doc_path = os.path.join(save_path, sub_folder_name)
                doc_path = save_path
                create_folder(doc_path, parents=True, exist_ok=True)
            for j, image in enumerate(images_data):
                if save:
                    out_path = os.path.join(doc_path, f"{file_name}_page_{j+1}")
                    save_as_is(doc,j,images_data,out_path) #i always have a single image
                else:
                    images_list.append( save_as_is(doc,j,images_data,None,return_image=True) ) 

    return images_list

