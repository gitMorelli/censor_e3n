#!/usr/bin/env python3
import argparse
import logging
import os
from time import perf_counter
from pathlib import Path

import pandas as pd
import numpy as np
import cv2

from src.utils.convert_utils import process_pdf_files
#from src.utils.convert_utils import pdf_to_png_images
from src.utils.file_utils import list_files_with_extension, load_template_info
from src.utils.file_utils import get_basename, create_folder, remove_folder, load_annotation_tree

from src.utils.json_parsing import get_page_list, get_roi_boxes
from src.utils.json_parsing import get_censor_boxes, get_censor_close_boxes

from src.utils.feature_extraction import extract_features_from_page, preprocess_page, extract_features_from_text_region, preprocess_text_region, preprocess_blank_roi, censor_image_with_boundary

from src.utils.matching_utils import pre_load_image_properties, initialize_page_dictionary, initialize_template_dictionary
from src.utils.matching_utils import initialize_sorting_dictionaries, pre_load_selected_templates, perform_template_matching
from src.utils.matching_utils import perform_phash_matching, perform_ocr_matching, ordering_scheme_base, perform_orb_matching

from src.utils.alignment_utils import compute_misalignment, roi_blank_decision, adjust_boundary_boxes, orb_matching

from src.utils.censor_utils import map_to_smallest_containing, save_as_is_no_censoring, get_transformation_from_dictionaries, apply_transformation_to_boxes, save_censored_image

from src.utils.debug_utils import visualize_templates_w_annotations, save_w_boxes

from src.utils.logging import FileWriter, initialize_logger

logging.basicConfig( 
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PDF_LOAD_PATH="//smb-recherche-s1.prod-powerscale.intra.igr.fr/E34N_HANDWRITING$\\Fichiers"#additional"#100263_template"
CSV_LOAD_PATH="//smb-recherche-s1.prod-powerscale.intra.igr.fr/E34N_HANDWRITING$\\ref_pdf_Qx"#additional"#100263_template"
TEMPLATES_PATH="//vms-e34n-databr/2025-handwriting\\data\\annotations\current_template"#additional"#100263_template"
SAVE_PATH="//vms-e34n-databr/2025-handwriting\\data\\test_censoring_pipeline"#additional"#100263_template" 


QUESTIONNAIRE = "5"
N_ALIGN_REGIONS = 2 #minimum number of align boxes needed for matching
SCALE_FACTOR_MATCHING = 2 
GAP_THRESHOLD_PHASH = 5
MAX_DIST_PHASH = 18
TEXT_SIMILARITY_METRIC = 'similarity_jaccard_tokens'
FILENAME_COL = 'object_name'
ID_COL = 'e3n_id_hand'
USED_COL = 'Used'
ORB_GOOD_MATCH = 50
MATCHING_THRESHOLD = 0.7
N_BLACK_THRESH=0.1
BLANK_REGION_TESTING_THRESHOLD = 1
EPSILON_EDGE_MATCHING = 2.0
THICKNESS_PCT = 0.2
SPACING_MULT = 0.5
ORB_NFEATURES = 2000
ORB_top_n_matches = 50
ORB_match_threshold = 10
SAVE_ANNOTATED_TEMPLATES=True


def main():
    args = parse_args()

    templates_path = args.templates_path
    pdf_load_path = args.pdf_load_path
    csv_load_path = args.csv_load_path
    save_path = args.save_path
    save_debug_times=args.save_debug_times
    save_debug_images=args.save_debug_images

    updated_csv_paths = os.path.join(save_path,"ref_pdf")
    log_path=os.path.join(save_path,'logs')
    debug_images_path=os.path.join(save_path,'debug_images')
    debug_images_templates_path=os.path.join(save_path,'debug_images_templates')
    debug_images_original_path=os.path.join(debug_images_path,'original')
    debug_images_w_boxes_path=os.path.join(debug_images_path,'original_w_boxes')
    questionnairres_log_path=os.path.join(pdf_load_path, f"Q{QUESTIONNAIRE}")
    time_logs_path = os.path.join(log_path,'time_logs')
    results_path = os.path.join(save_path,'results')

    #i clean the folders from previous results
    if args.delete_previous_results:
        if os.path.exists(updated_csv_paths):
            remove_folder(updated_csv_paths)
        if os.path.exists(log_path):
            remove_folder(log_path)
        if os.path.exists(debug_images_path):
            remove_folder(debug_images_path)
        if os.path.exists(results_path):
            remove_folder(results_path)
    
    if args.verbose:
        #console logger
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
        #file logger
        create_folder(log_path, parents=True, exist_ok=True)
        file_logger=FileWriter(enabled=True,path=os.path.join(log_path,f"global_logger.txt"))
        # create folder to save debug images
        create_folder(debug_images_path, parents=True, exist_ok=True)

    csv_modified_path = os.path.join(updated_csv_paths,f"updated_ref_pdf_Q{QUESTIONNAIRE}.csv")
    if os.path.exists(csv_modified_path)==False: #if the csv has not been preprocessed yet
        df = preprocess_df(os.path.join(csv_load_path,f"ref_pdf_Q{QUESTIONNAIRE}.csv"),FILENAME_COL, ID_COL,USED_COL)
        create_folder(updated_csv_paths, parents=True, exist_ok=True)
        df.to_csv(csv_modified_path)
    
    df = load_preprocessed_df(csv_modified_path,used_col_name=USED_COL,id_col_name=ID_COL) #load the preprocessed df

    file_logger.write(df.head(10).to_string()) #log the first 10 lines of the df to check it is correct

    #load the annotation files (full paths and names)
    annotation_file_names, annotation_files = load_annotation_tree(file_logger, templates_path)
    #i select only the template of interest (eg for Q5 only doc_5.json)
    selected_templates = select_specific_annotation_file(QUESTIONNAIRE)

    file_logger.write(selected_templates)

    # I open the jsons for the selected templates and save them in a list, i also open the corresponding pre_computed data
    # #this list is a single element for QX X>1 and two elements for X=1 
    annotation_roots, npy_data = load_template_info(file_logger,annotation_files,annotation_file_names,
                                                    templates_path, selected_files=selected_templates)
    
    pages_in_annotation = get_page_list(annotation_roots[0]) #is the same for both templates in Q1 case 
    #so i can just take it from the first one

    file_logger.write(pages_in_annotation)

    # Group by the 'id' column and iterate over each group
    for unique_id, group in df.groupby(ID_COL):

        filenames = group[FILENAME_COL].tolist()
        # i sort the filenames by name (expected page ordering is absed on alphabetical ordering)
        filenames.sort()  # eg ordered as k,l,m,n
        #checks both for .pdf and for .tif.pdf
        pdf_paths = get_file_paths(filenames,questionnairres_log_path,file_logger) 

        file_logger.write(filenames)
        #file_logger.write(pdf_paths)

        #i extract the images and order them based on the expeected ordering
        #in some cases pdf_paths is a single multipage pdfs in others are multiple one page pdfs files
        list_of_images = process_pdf_files(QUESTIONNAIRE,pdf_paths,None,save=False)

        #TEST, remove after, disorder the pages
        new_order=[3,2,1,0]
        list_of_images = [list_of_images[i] for i in new_order] 

        #DEBUG
        if save_debug_images:
            save_list_of_images(list_of_images, debug_images_original_path, unique_id, args.verbose)

        # I want to find the best match for the 
        # load dictionary to store warning messages on pages
        test_log = {'doc_level_warning':None,
                    'Choosen template': QUESTIONNAIRE,
                    'Confidence on template choice': None,
                    'n_pages': len(list_of_images),
                    'n_expected_pages': len(pages_in_annotation),
                    'report_orb': None ,
                    'report_ocr': None, 
                    'report_phash': None}
        for p in pages_in_annotation:
            test_log[p]={'failed_test_1': False, 'template_1': None, 'confidences_template_1': None,
                        'failed_test_2': False, 'orb': None, 'template_2': None, 'confidences_template_2': None,
                        'OCR_WARNING': None, 'OCR': None,
                        'page_level_warning': None,
                        'alingement_report': None, 
                        'roi_and_blank_before' : None, 'roi_and_blank_after' : None}

        
        #i select the annotation root and npy data corresponding to the correct template 
        # #(in Q1 case i have two templates, in the other cases only one so it is straightforward)
        report,selected_template_index,selected_confidence, root , npy_dict = select_template(pages_in_annotation,QUESTIONNAIRE,
        annotation_roots,npy_data, list_of_images, file_logger) 

        if report:
            test_log['Choosen template'] = selected_templates[selected_template_index]
            test_log['Confidence on template choice'] = selected_confidence
            test_log['report_phash'] = report
        
        file_logger.write(selected_templates[selected_template_index])
        file_logger.write(report)
        
        #initialize the dictionaries i will use to store info on the sorting process
        #i should avoid to re-initialize if q1 (but i spare a negligible amount of time)
        page_dictionary,template_dictionary = initialize_sorting_dictionaries(list_of_images, root, input_from_file=False)
        #i will consider all template pages from the beginning and all images of course
        templates_to_consider = pages_in_annotation[:]
        pages_to_consider = [i+1 for i in range(len(list_of_images))]

        #pre_load_template_info
        template_dictionary = pre_load_selected_templates(templates_to_consider,npy_dict, 
                                                          root, template_dictionary)
        
        #check with pages were not matched
        test_passed,page_dictionary, template_dictionary, test_log, problematic_pages, problematic_templates = perform_first_stage_check(pages_to_consider, templates_to_consider, 
                                                                                                                                         page_dictionary, template_dictionary, test_log,
                                                                                                                                         file_logger)
            
        file_logger.write(f"problematic pages {problematic_pages}")
        file_logger.write(f"test passed: {test_passed}")

        
        if not test_passed: 
            #sort with orb matching and check if the association is correct via template matching
            #pre_load orb keypoints for images
            page_dictionary = pre_load_image_properties(problematic_pages,page_dictionary,
                                                        template_dictionary,properties=['orb'])
            test_passed,page_dictionary, template_dictionary, test_log, problematic_pages, problematic_templates = perform_second_stage_check(problematic_pages, problematic_templates, 
                                                                                                                                         page_dictionary, template_dictionary, test_log)
            
            file_logger.write(f"problematic pages step 2: {problematic_pages}")
            file_logger.write(f"test passed 2: {test_passed}")

            debug_print_associations(pages_to_consider,page_dictionary,file_logger)
        
            if not test_passed:
                #sort with ocr matching 
                template_dictionary, page_dictionary, report = perform_ocr_matching(problematic_pages,problematic_templates, 
                                                        page_dictionary, template_dictionary,text_similarity_metric=TEXT_SIMILARITY_METRIC, compute_report=True)
                #update the test log with the ocr results
                for img_id in problematic_pages:
                    test_log[img_id]['OCR'] = page_dictionary[img_id]['match_ocr']
                test_log['report_ocr'] = report

                debug_print_associations(pages_to_consider,page_dictionary,file_logger)
        
        # now we censor the pages
        for img_id in pages_to_consider:

            page = page_dictionary[img_id]
            matched_id = page['matched_page']
            template = template_dictionary[matched_id]

            #create time logger for the current page
            patient_time_log_path=os.path.join(time_logs_path, f"patient_{unique_id}", QUESTIONNAIRE)
            create_folder(patient_time_log_path, parents=True, exist_ok=True)
            image_time_logger=FileWriter(save_debug_times,
                                            os.path.join(patient_time_log_path,f"time_logger_page_{matched_id}.txt"))

            if matched_id == None:
                test_log[img_id]['page_level_warning'] = "Page not matched with any template page, it will be ignored"
                continue

            img_size = page['img_size']
            template_size = template_dictionary[matched_id]['template_size']
            img=page['img'] #all pages are already loaded

            if template['type']=='N':
                file_logger.write(f"Page {img_id} considered as N, no censoring applied, saved as is")
                save_as_is_no_censoring(file_logger,image_time_logger,img_id,page_dictionary,dest_folder=results_path,
                                        n_p=unique_id,n_doc=QUESTIONNAIRE,n_page=matched_id)
            
            #I check the extra roi and the blank region and save the results
            #extra roi
            pre_computed = npy_dict[matched_id]
            roi_boxes, pre_computed_rois = get_roi_boxes(root,pre_computed,matched_id) #first one is the extra roi, second one is the blank
            #prev_values = check_blank_and_extra(roi_boxes, pre_computed_rois, page, img_size) #uncomment if you want to use the roi for template matching and the blank for checking
            
            #i get the parameters needed for the alignement of the page
            scale_factor, shift_x, shift_y, angle_degrees,reference = get_transformation_from_dictionaries(page, template, image_time_logger, scale_factor=SCALE_FACTOR_MATCHING)
            alignement_report = {'scale_factor': scale_factor, 'shift_x': shift_x, 'shift_y': shift_y, 'angle_degrees': angle_degrees}

            #DEBUG
            transformation = {'reference': reference, 'scale_factor': scale_factor, 'shift_x': shift_x, 'shift_y': shift_y, 'angle_degrees': angle_degrees}
            file_logger.write(f"Alignement parameters for page {matched_id}: {alignement_report}") 
            if save_debug_images:
                save_w_boxes(debug_images_w_boxes_path,unique_id,QUESTIONNAIRE,matched_id,img,root,
                             pre_computed,image_time_logger,which_boxes=['align','transformed'], transformation=transformation)
            
            continue

            # I check that the extra roi i consider is closer and that the blank region is void/voider
            new_roi_boxes = apply_transformation_to_boxes(roi_boxes, image_time_logger, reference, scale_factor, 
                                                                shift_x, shift_y, angle_degrees,name='roi') 
            #new_values = check_blank_and_extra(new_roi_boxes, pre_computed_rois, page, img_size) #uncomment if you want to use the roi for template matching and the blank for checking


            # I check alignement on the extra roi with orb matching
            orb_shift_x, orb_shift_y, orb_scale, orb_angle = orb_matching(img,new_roi_boxes[0],pre_computed_rois[0], top_n_matches=ORB_top_n_matches, 
                                                                          orb_nfeatures=ORB_NFEATURES , match_threshold=ORB_match_threshold, scale_factor=SCALE_FACTOR_MATCHING)
            orb_match=True
            if orb_shift_x is None:
                orb_match=False 


            # i save the results in the test log
            #test_log[img_id]['roi_and_blank_before'] = prev_values
            #test_log[img_id]['roi_and_blank_after'] = new_values
            test_log[img_id]['alingement_report'] = alignement_report
            
            #i compute the warning string to attach to the filename
            warning_string = ""

            # I preprocess the censor regions to extend their dimensions to page limits
            censor_boxes,partial_coverage = get_censor_boxes(root,matched_id) #we need to refer to the correct id of the template
            censor_close_boxes,_ = get_censor_close_boxes(root,matched_id)
            censor_boxes = adjust_boundary_boxes(censor_boxes, template['template_size'], img_size , epsilon=EPSILON_EDGE_MATCHING) 
            #should i apply the scale factor  to the page dimension or rescale everything 
            #together later?

            if orb_match:
                #i associate each close box with the container box and create an ordered list of the containers that match the censoring boxes
                map_to_container = map_to_smallest_containing(censor_close_boxes,censor_boxes)
                boundary_boxes = []
                for close_box in censor_close_boxes:
                    container_box = map_to_container[close_box]
                    boundary_boxes.append(container_box)
                #i rescale the dimensions of the censor-close boxes
                censor_close_boxes = apply_transformation_to_boxes(censor_close_boxes, image_time_logger, reference, scale_factor, 
                                                                    shift_x, shift_y, angle_degrees,name='censor_close')

                # I enlarge close-censor regions based on alignement results
                censor_close_boxes = adjust_close_censor(censor_close_boxes,orb_shift_x, orb_shift_y, orb_scale, orb_angle)

                # apply censoring considering the boundary boxes and the close censor boxes
                parent_path=os.path.join(save_path, f"patient_{unique_id}", f"document_{QUESTIONNAIRE}")#, f"censored_page_{n_page}.png")
                create_folder(parent_path, parents=True, exist_ok=True)
                save_path=os.path.join(parent_path, f"censored_page_w{warning_string}_{matched_id}.png")
                censored_img = censor_image_with_boundary(img, censor_close_boxes, boundary_boxes, 
                                                        partial_coverage=partial_coverage,logger=image_time_logger,
                                                        thickness_pct=THICKNESS_PCT, spacing_mult=SPACING_MULT)
                image_time_logger and image_time_logger.call_start(f'writing_to_memory')
                cv2.imwrite(str(save_path), censored_img)
                image_time_logger and image_time_logger.call_end(f'writing_to_memory')
            else:
                # i censor considering the large regions
                save_censored_image(img, censor_boxes, save_path,unique_id,QUESTIONNAIRE,matched_id,
                                    warning=warning_string,partial_coverage=partial_coverage,
                                    thickness_pct=THICKNESS_PCT, spacing_mult=SPACING_MULT,logger=image_time_logger)
        return 0    

    logger.info("Conversion finished")
    return 0

def debug_print_associations(pages_to_consider,page_dictionary,logger):
    for img_id in pages_to_consider:
        page = page_dictionary[img_id]
        matched_id = page['matched_page']
        logger.write(f"Page {img_id} is matched with template page {matched_id}")
    logger.write("-"*50 + "\n")
    return 

def initialize_result_df(pdf_names,template_type):
    if template_type == "Q10":
        #add code to modify the list
        pass
    df = pd.DataFrame(pdf_names, columns=['pdf_name'])
    # 3. Add two empty columns
    # Using None or np.nan is standard for "empty" data
    df['saved_file_name'] = pdf_names
    df['id'] = None
    df['matched_template_page'] = None
    return df

def preprocess_df(source_path,filename_col,id_col,used_col_name='Used'):
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
    df[used_col_name] = False

    return df

def load_preprocessed_df(file_path,id_subset = None, used_col_name='Used',id_col_name='e3n_id_hand'):
    df=pd.read_csv(file_path)
    #select only the lines with used=False
    df = df[df[used_col_name]==False] 
    #Select a subset of ids if provided (eg PD subjects)
    if id_subset is not None:
        df = df[df[id_col_name].isin(id_subset)]

    return df


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

def select_specific_annotation_file(questionnaire):
    #i will select only one annotation file from the library
    if questionnaire in [f"{i}" for i in range(2,14)]:
        selected_templates = [f"q_{questionnaire}"]
    elif questionnaire == "1":
        selected_templates = ["q_1","q_1v2"]
    return selected_templates

def select_template(pages_in_annotation,questionnaire,annotation_roots,npy_data, list_of_images, logger):
    if questionnaire == "1":
        #i have two templates for Q1 so i will select the one that has better matches with the images
        #i will do this by performing a simple phash matching between the images 
        # and the templates and selecting the template with more matches
        templates_to_consider = pages_in_annotation[:]
        pages_to_consider = [i+1 for i in range(len(list_of_images))]

        #i load image information
        page_dictionary = initialize_page_dictionary(list_of_images,input_from_file=False)

        #set the cost to infinity
        max_cost = float('inf')
        selected_confidence = 0

        for i in range(len(annotation_roots)):
            root, npy_dict = annotation_roots[i], npy_data[i]

            #initialize the template dictionaries i will use to store info on the sorting process
            template_dictionary = initialize_template_dictionary(root)

            #pre_load_template_info
            template_dictionary = pre_load_selected_templates(templates_to_consider,npy_dict, 
                                                              root, template_dictionary)
            
            #pre_load phash for images, i can do only once but i have to do after i have the template
            #dictionary because it contains info on page pre_processing for phash
            if i==0:
                page_dictionary = pre_load_image_properties(pages_to_consider,page_dictionary,
                                                            template_dictionary,properties=['phash'])
            
            #perform phash matching
            page_dictionary, report = perform_phash_matching(page_dictionary,template_dictionary, templates_to_consider, templates_to_consider, 
                            gap_threshold=GAP_THRESHOLD_PHASH,max_dist=MAX_DIST_PHASH, compute_report=True)
            total_cost = report['total_cost']

            if total_cost < max_cost:
                selected_template_index = i
                max_cost = total_cost
                selected_confidence = report["is_confident"]
        logger.write(f"Selected template: {i} with total cost {max_cost}")
        
        return report,selected_template_index,selected_confidence,annotation_roots[selected_template_index],npy_data[selected_template_index]
            
    else:
        return None,0, None, annotation_roots[0],npy_data[0]

def perform_first_stage_check(pages_to_consider, templates_to_consider, page_dictionary, template_dictionary, test_log, logger):
    #i test if the pages are already in place
    pairs_to_consider = []
    #i need to prepare the list of pairs to check considering that the extracted pages can be less or more than the pages in the template, 
    min_length = min(len(pages_to_consider),len(templates_to_consider))
    range_to_check = range(min_length)
    for i in range_to_check: 
        img_id = pages_to_consider[i]
        matched_id = templates_to_consider[i]
        pairs_to_consider.append([img_id,matched_id])

    #perform template_matching and update the matching keys in the dictionaries
    page_dictionary,template_dictionary = perform_template_matching(pairs_to_consider,page_dictionary,template_dictionary, 
                                n_align_regions=N_ALIGN_REGIONS,scale_factor=SCALE_FACTOR_MATCHING,matching_threshold=MATCHING_THRESHOLD, compute_report=True)
    
    problematic_pages = pages_to_consider[:]
    problematic_templates = templates_to_consider[:]
    n_matches=0
    for i in range_to_check:
        img_id = pages_to_consider[i]
        t_id = templates_to_consider[i]
        matched_id_template = page_dictionary[img_id]['matched_page']
        logger.write(f"Checking page {img_id} against template {t_id}: matched with template {matched_id_template} with confidence {page_dictionary[img_id]['confidence_template']}")
        test_log[img_id]['confidences_template_1'] = page_dictionary[img_id]['confidence_template'] 
        if matched_id_template: #there was match with the expected page -> remove it from the problematic list
            n_matches+=1
            problematic_pages.remove(img_id)
            problematic_templates.remove(t_id)
            test_log[img_id]['template_1'] = matched_id_template
        else:
            test_log[img_id]['failed_test_1'] = True 
    test_passed=False
    if n_matches==min_length:
        test_passed=True
    
    return test_passed,page_dictionary, template_dictionary, test_log, problematic_pages, problematic_templates

#probably i can make more modular by making the "check_with_template_matching" a function that can be used after phash or after orb or others ...
def perform_second_stage_check(pages_to_consider, templates_to_consider, page_dictionary, template_dictionary, test_log):

    page_dictionary, report = perform_orb_matching(page_dictionary,template_dictionary, pages_to_consider, templates_to_consider, 
                            gap_threshold=GAP_THRESHOLD_PHASH,max_dist=MAX_DIST_PHASH, orb_good_match=ORB_GOOD_MATCH,compute_report=True)
    test_log['report_orb'] = report

    #i test if the pages are already in place
    pairs_to_consider = []
    for img_id in pages_to_consider: 
        orb_matched_id = page_dictionary[img_id]['match_orb']
        #print(orb_matched_id)
        pairs_to_consider.append([img_id,orb_matched_id])

    #perform template_matching and update the matching keys in the dictionaries
    page_dictionary,template_dictionary = perform_template_matching(pairs_to_consider,page_dictionary,template_dictionary, 
                                n_align_regions=N_ALIGN_REGIONS,scale_factor=SCALE_FACTOR_MATCHING,matching_threshold=MATCHING_THRESHOLD, compute_report=True)
    
    problematic_pages = pages_to_consider[:]
    problematic_templates = templates_to_consider[:]
    n_matches=0
    for i in range(len(pairs_to_consider)):
        img_id = pairs_to_consider[i][0]
        orb_match = pairs_to_consider[i][1]
        matched_id_template = page_dictionary[img_id]['matched_page']
        test_log[img_id]['orb'] = orb_match
        test_log[img_id]['confidences_template_2'] = page_dictionary[img_id]['confidence_template']
        if matched_id_template == orb_match: #there was match with the expected page -> remove it from the problematic list
            n_matches+=1
            problematic_pages.remove(img_id)
            problematic_templates.remove(orb_match)
            test_log[img_id]['template_2'] = matched_id_template
        else:
            test_log[img_id]['failed_test_2'] = True 
    test_passed=False
    if n_matches==len(pairs_to_consider):
        test_passed=True
    
    return test_passed,page_dictionary, template_dictionary, test_log, problematic_pages, problematic_templates

def check_blank_and_extra(roi_boxes, pre_computed_rois, page, img_size):
    shifts, centers,_,confidences= compute_misalignment(page['img'], roi_boxes[:1], img_size, pre_computed_rois[:1], scale_factor=SCALE_FACTOR_MATCHING,
                            matching_threshold=MATCHING_THRESHOLD, pre_computed_rois=None,return_confidences=True)
    #blank region
    f_roi = preprocess_blank_roi(page['img'], roi_boxes[-1])
    decision, black_diff_to_template,cc_difference_to_template = roi_blank_decision(f_roi,pre_computed_roi=pre_computed_rois[-1], return_features = True ,
                                                                                    n_black_thresh=N_BLACK_THRESH,threshold_test=BLANK_REGION_TESTING_THRESHOLD)
    prev_values = {'shifts': shifts, 'centers': centers, 'confidences': confidences, 
                    'black_diff_to_template': black_diff_to_template, 'cc_difference_to_template': cc_difference_to_template}
    return prev_values
        
def adjust_close_censor(new_censor_close_boxes,orb_shift_x, orb_shift_y, orb_scale, orb_angle):
    def transform_box(pts, scale, shift_x, shift_y):
        #i suppose the box is saved as a polygon: points in clockwise order starting from the top left corner (pt0, pt1, pt2, pt3)
        # 1. Scaling around the center
        center = np.mean(pts, axis=0)

        if scale>1:
            pts = center + (pts - center) * scale
        
        # 2. Define Local Axes (Direction vectors)
        # Unit vector along the 'width' (from pt0 to pt1)
        v_w = pts[1] - pts[0]
        dist_w = np.linalg.norm(v_w)
        u_w = v_w / dist_w
        
        # Unit vector along the 'height' (from pt0 to pt3)
        v_h = pts[3] - pts[0]
        dist_h = np.linalg.norm(v_h)
        u_h = v_h / dist_h

        # 3. Apply Shifts
        # Shift X: Affects 'right' (pts 1,2) or 'left' (pts 0,3)
        if shift_x > 0:
            pts[1] += u_w * shift_x
            pts[2] += u_w * shift_x
        elif shift_x < 0:
            pts[0] += u_w * shift_x # shift_x is negative, so it moves 'backwards'
            pts[3] += u_w * shift_x

        # Shift Y: Affects 'bottom' (pts 2,3) or 'top' (pts 0,1)
        # Note: Logic depends on if your Y-axis grows 'down' (images) or 'up' (math)
        if shift_y > 0:
            pts[2] += u_h * shift_y
            pts[3] += u_h * shift_y
        elif shift_y < 0:
            pts[0] += u_h * shift_y
            pts[1] += u_h * shift_y

        return pts
    for i in range(len(new_censor_close_boxes)):
        new_censor_close_boxes[i] = transform_box(new_censor_close_boxes[i], orb_scale, orb_shift_x, orb_shift_y)
    
    return new_censor_close_boxes

#### Debug ####
def save_list_of_images(list_of_images, debug_images_original, unique_id, verbose):
    if not verbose:
        return 0
    else:
        for i in range(len(list_of_images)):
            #save the images in the debug folder to check they are correctly extracted and ordered
            debug_img_path = os.path.join(debug_images_original ,f"patient_{unique_id}_page_{i+1}.png")
            create_folder(os.path.dirname(debug_img_path), parents=True, exist_ok=True)
            cv2.imwrite(debug_img_path, list_of_images[i])

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
        "-t", "--templates_path",
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