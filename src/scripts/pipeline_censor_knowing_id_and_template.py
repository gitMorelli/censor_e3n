#!/usr/bin/env python3
import argparse
import logging
import os
from time import perf_counter
from pathlib import Path

import pandas as pd

from src.utils.convert_utils import process_pdf_files
#from src.utils.convert_utils import pdf_to_png_images
from src.utils.file_utils import list_files_with_extension, load_template_info
from src.utils.file_utils import get_basename, create_folder, remove_folder, load_annotation_tree

from src.utils.json_parsing import get_page_list
from src.utils.json_parsing import get_censor_boxes, get_censor_close_boxes

from src.utils.feature_extraction import extract_features_from_page, preprocess_page, extract_features_from_text_region, preprocess_text_region

from src.utils.matching_utils import pre_load_image_properties, initialize_page_dictionary, initialize_template_dictionary
from src.utils.matching_utils import initialize_sorting_dictionaries, pre_load_selected_templates, perform_template_matching
from src.utils.matching_utils import perform_phash_matching, perform_ocr_matching, ordering_scheme_base, perform_orb_matching

from src.utils.censor_utils import map_to_smallest_containing

from src.utils.debug_utils import visualize_templates_w_annotations


logging.basicConfig( 
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PDF_LOAD_PATH="//smb-recherche-s1.prod-powerscale.intra.igr.fr/E34N_HANDWRITING$\\Fichiers"#additional"#100263_template"
CSV_LOAD_PATH="//smb-recherche-s1.prod-powerscale.intra.igr.fr/E34N_HANDWRITING$\\ref_pdf_Qx"#additional"#100263_template"
TEMPLATES_PATH="//vms-e34n-databr/2025-handwriting\\data\\e3n_templates_png\\current_template"#additional"#100263_template"
SAVE_PATH="//vms-e34n-databr/2025-handwriting\\data\\test_censoring_pipeline"#additional"#100263_template" 

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
    return parser.parse_args()

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

def main():
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)

    templates_path = args.templates_path
    pdf_load_path = args.pdf_load_path
    csv_load_path = args.csv_load_path
    save_path = args.save_path

    csv_modified_path = os.path.join(save_path,"ref_pdf",f"updated_ref_pdf_Q{QUESTIONNAIRE}.csv")
    if os.path.exists(csv_modified_path)==False: #if the csv has not been preprocessed yet
        df = preprocess_df(csv_load_path,FILENAME_COL, ID_COL,USED_COL)
        df.to_csv(csv_modified_path)
    
    load_preprocessed_df(csv_modified_path,used_col_name=USED_COL,id_col_name=ID_COL) #load the preprocessed df

    #load the annotation files (full paths and names)
    annotation_file_names, annotation_files = load_annotation_tree(logger, templates_path)
    #i select only the template of interest (eg for Q5 only doc_5.json)
    selected_templates = select_specific_annotation_file(QUESTIONNAIRE)

    # I open the jsons for the selected templates and save them in a list, i also open the corresponding pre_computed data
    # #this list is a single element for QX X>1 and two elements for X=1 
    annotation_roots, npy_data = load_template_info(logger,annotation_files,annotation_file_names,
                                                    templates_path, selected_files=selected_templates)
    
    pages_in_annotation = get_page_list(annotation_roots[0]) #is the same for both templates in Q1 case 
    #so i can just take it from the first one

    # Group by the 'id' column and iterate over each group
    for unique_id, group in df.groupby(ID_COL):

        filenames = group[FILENAME_COL].tolist()
        # i sort the filenames by name (expected page ordering is absed on alphabetical ordering)
        filenames.sort() 
        #checks both for .pdf and for .tif.pdf
        pdf_paths = get_file_paths(filenames,pdf_load_path,logger) 

        #i extract the images and order them based on the expeected ordering
        #in some cases pdf_paths is a single multipage pdfs in others are multiple one page pdfs files
        list_of_images = process_pdf_files(QUESTIONNAIRE,pdf_paths,None,save=False)

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
                        'OCR_WARNING': None, 'OCR': None}

        #i select the annotation root and npy data corresponding to the correct template 
        # #(in Q1 case i have two templates, in the other cases only one so it is straightforward)
        report,selected_template_index,selected_confidence, root , npy_dict = select_template(QUESTIONNAIRE,annotation_roots,npy_data, list_of_images, logger) 

        if selected_template_index:
            test_log['Choosen template'] = selected_templates[selected_template_index]
            test_log['Confidence on template choice'] = selected_confidence
            test_log['report_phash'] = report
        
        #initialize the dictionaries i will use to store info on the sorting process
        page_dictionary,template_dictionary = initialize_sorting_dictionaries(list_of_images, root, input_from_file=False)
        #i will consider all template pages from the beginning and all images of course
        templates_to_consider = pages_in_annotation[:]
        pages_to_consider = [i+1 for i in range(len(list_of_images))]

        #pre_load_template_info
        template_dictionary = pre_load_selected_templates(templates_to_consider,npy_dict, 
                                                          root, template_dictionary)
        
        #check with pages were not matched
        test_passed,page_dictionary, template_dictionary, test_log, problematic_pages, problematic_templates = perform_first_stage_check(pages_to_consider, templates_to_consider, 
                                                                                                                                         page_dictionary, template_dictionary, test_log)
            
        if not test_passed: 
            #sort with orb matching and check if the association is correct via template matching
            #pre_load orb keypoints for images
            page_dictionary = pre_load_image_properties(problematic_pages,page_dictionary,
                                                        template_dictionary,properties=['orb'])
            test_passed,page_dictionary, template_dictionary, test_log, problematic_pages, problematic_templates = perform_second_stage_check(problematic_pages, problematic_templates, 
                                                                                                                                         page_dictionary, template_dictionary, test_log)
            
            if not test_passed:
                #sort with ocr matching 
                template_dictionary, page_dictionary, report = perform_ocr_matching(problematic_pages,problematic_templates, 
                                                        page_dictionary, template_dictionary,text_similarity_metric=TEXT_SIMILARITY_METRIC, compute_report=True)
                #update the test log with the ocr results
                for img_id in problematic_pages:
                    test_log[img_id]['OCR'] = page_dictionary[img_id]['match_ocr']
                test_log['report_ocr'] = report
        
        # now we censor the pages


    logger.info("Conversion finished")
    return 0

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
                raise FileNotFoundError(f"Neither {path} nor {new_path} exist.")

    for filename in filenames:
        try:
            # check both for file.pdf and for file.tif.pdf
            file_path = get_filepath(os.path.join(pdf_load_path, filename+'.pdf'))
            file_paths.append(file_path)
        except FileNotFoundError as e:
            logger.warning(str(e))
            continue

def select_specific_annotation_file(questionnaire):
    #i will select only one annotation file from the library
    if questionnaire in [f"Q{i}" for i in range(2,14)]:
        selected_templates = [f"q_{questionnaire.split('Q')[1]}"]
    elif QUESTIONNAIRE == "Q1":
        selected_templates = ["q_1","q_1v2"]
    return selected_templates

def select_template(pages_in_annotation,questionnaire,annotation_roots,npy_data, list_of_images, logger):
    if questionnaire == "Q1":
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
            page_dictionary, confident, report = perform_phash_matching(page_dictionary,template_dictionary, templates_to_consider, templates_to_consider, 
                            gap_threshold=GAP_THRESHOLD_PHASH,max_dist=MAX_DIST_PHASH, compute_report=True)
            total_cost = report['total_cost']

            if total_cost < max_cost:
                selected_template_index = i
                max_cost = total_cost
                selected_confidence = confident
        logger.info(f"Selected template: {annotation_roots[selected_template_index]['filename']} with total cost {max_cost}")
        
        return report,selected_template_index,selected_confidence,annotation_roots[selected_template_index],npy_data[selected_template_index]
            
            
    else:
        return None,None, None, annotation_roots[0],npy_data[0]

def perform_first_stage_check(pages_to_consider, templates_to_consider, page_dictionary, template_dictionary, test_log):
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
        orb_matched_id = pages_to_consider[img_id]['match_orb']
        pairs_to_consider.append([img_id,orb_matched_id])

    #perform template_matching and update the matching keys in the dictionaries
    page_dictionary,template_dictionary = perform_template_matching(pairs_to_consider,page_dictionary,template_dictionary, 
                                n_align_regions=N_ALIGN_REGIONS,scale_factor=SCALE_FACTOR_MATCHING,matching_threshold=MATCHING_THRESHOLD, compute_report=True)
    
    problematic_pages = pages_to_consider[:]
    problematic_templates = templates_to_consider[:]
    n_matches=0
    for i in range(len(pairs_to_consider)):
        img_id = pairs_to_consider[0]
        orb_match = pairs_to_consider[1]
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


if __name__ == "__main__":
    main()