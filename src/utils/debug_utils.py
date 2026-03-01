
import shutil
import os

import cv2

#from src.utils.convert_utils import pdf_to_png_images
from src.utils.file_utils import list_subfolders,list_files_with_extension,sort_files_by_page_number,get_page_number, load_template_info
from src.utils.file_utils import get_basename, create_folder, check_name_matching, remove_folder, load_annotation_tree, load_subjects_tree

#from src.utils.xml_parsing import load_xml, iter_boxes, add_attribute_to_boxes
#from src.utils.xml_parsing import save_xml, iter_images, set_box_attribute,get_box_coords

from src.utils.json_parsing import get_attributes_by_page, get_page_list, get_page_dimensions,get_box_coords_json, get_censor_type
from src.utils.json_parsing import get_align_boxes, get_ocr_boxes, get_roi_boxes, get_censor_boxes, get_censor_close_boxes

from src.utils.feature_extraction import crop_patch, preprocess_alignment_roi, preprocess_roi, preprocess_blank_roi,load_image
from src.utils.feature_extraction import extract_features_from_blank_roi, extract_features_from_roi,censor_image
from src.utils.feature_extraction import extract_features_from_page, preprocess_page, extract_features_from_text_region, preprocess_text_region

from src.utils.alignment_utils import page_vote,compute_transformation, compute_misalignment,apply_transformation,enlarge_crop_coords
from src.utils.alignment_utils import plot_rois_on_image_polygons,plot_rois_on_image,plot_both_rois_on_image,template_matching, plot_rois_on_image_stackable

from src.utils.logging import FileWriter, initialize_logger

from src.utils.matching_utils import update_phash_matches, match_pages_phash, check_matching_correspondence, pre_load_images_to_censor, pre_load_image_properties
from src.utils.matching_utils import compare_pages_same_section, match_pages_text, initialize_sorting_dictionaries, pre_load_selected_templates, perform_template_matching
from src.utils.matching_utils import perform_phash_matching, perform_ocr_matching, ordering_scheme_base

from src.utils.censor_utils import save_as_is_no_censoring, save_original_w_boxes, get_transformation_to_match_to_template, apply_transformation_to_boxes


def visualize_templates_w_annotations(annotation_files,annotation_roots,npy_data,source,align=True,censor=False,roi=False, mode='cv2'): 
    for i, annotation_file in enumerate(annotation_files): #document level
        #load the json file
        root = annotation_roots[i]
        pages_in_annotation = get_page_list(root)
        npy_dict = npy_data[i]
        annotation_filename=get_basename(annotation_file, remove_extension=True)

        for img_id in pages_in_annotation:
            pre_computed=npy_dict[img_id]
            align_boxes, pre_computed_align = get_align_boxes(root,pre_computed,img_id)
            debug_boxes=align_boxes
            colors=["red" for i in range(len(align_boxes))]
            parent_path=os.path.join(source,'debug', "templates", f"{annotation_filename}")#, f"censored_page_{n_page}.png")
            create_folder(parent_path, parents=True, exist_ok=True)
            save_debug_path=os.path.join(parent_path, f"{img_id}_template_w_align.png")

            img_path=os.path.join(source,'templates', f"{annotation_filename}",f"page_{img_id}.png")
            img = load_image(img_path, mode=mode, verbose=False)
            plot_rois_on_image(img, debug_boxes, save_debug_path,colors=colors)


def save_w_boxes(save_path,subj_id,doc_ind,matched_id,img,root,pre_computed,logger,which_boxes=['align','transformed'], transformation=None):
    parent_path=os.path.join(save_path, f"patient_{subj_id}", f"document_{doc_ind}")#, f"censored_page_{n_page}.png")
    create_folder(parent_path, parents=True, exist_ok=True)
    save_debug_path=os.path.join(parent_path, f"{matched_id}_original_w_boxes.png")
    
    debug_boxes=[]
    box_colors = []
    box_types = []
    colors = {"align":"red", "roi":"green", "censor":"blue", "censor_close":"cyan",
              "new_align":"orange", "new_roi":"lightgreen", "new_censor":"lightblue", "new_censor_close":"lightcyan"}

    if "align" in which_boxes:
        boxes, pre_computed_align = get_align_boxes(root,pre_computed,matched_id)
        debug_boxes+=boxes
        box_colors+=[colors["align"] for i in range(len(boxes))]
        box_types+=["align" for i in range(len(boxes))]
    if "roi" in which_boxes:
        boxes, pre_computed_align = get_roi_boxes(root,pre_computed,matched_id)
        debug_boxes+=boxes
        box_colors+=[colors["roi"] for i in range(len(boxes))]
        box_types+=["roi" for i in range(len(boxes))]
    if "censor" in which_boxes:
        boxes, pre_computed_align = get_censor_boxes(root,pre_computed,matched_id)
        debug_boxes+=boxes
        box_colors+=[colors["censor"] for i in range(len(boxes))]
        box_types+=["censor" for i in range(len(boxes))]
    if "censor_close" in which_boxes:
        boxes, pre_computed_align = get_censor_close_boxes(root,pre_computed,matched_id)
        debug_boxes+=boxes
        box_colors+=[colors["censor_close"] for i in range(len(boxes))]
        box_types+=["censor_close" for i in range(len(boxes))]
    new_image=plot_rois_on_image_stackable(img, debug_boxes, colors=box_colors)

    if (transformation is not None) and ('transformed' in which_boxes): 
        new_boxes = apply_transformation_to_boxes(debug_boxes, logger, transformation['reference'],transformation['scale_factor'], 
                                                            transformation['shift_x'], transformation['shift_y'], 
                                                            transformation['angle_degrees'],name='align')
        box_colors=[]
        for box_type in box_types:
            box_colors.append(colors[f"new_{box_type}"] )
        plot_rois_on_image_polygons(new_image, new_boxes, save_debug_path,colors=box_colors)