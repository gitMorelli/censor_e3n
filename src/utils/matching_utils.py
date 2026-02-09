import cv2
import numpy as np
from scipy.fftpack import dct
from scipy.optimize import linear_sum_assignment
import re
import unicodedata
from difflib import SequenceMatcher

from src.utils.file_utils import list_subfolders,list_files_with_extension,sort_files_by_page_number,get_page_number, load_template_info
from src.utils.file_utils import get_basename, create_folder, check_name_matching, remove_folder, load_annotation_tree, load_subjects_tree
#from src.utils.xml_parsing import load_xml, iter_boxes, add_attribute_to_boxes
#from src.utils.xml_parsing import save_xml, iter_images, set_box_attribute,get_box_coords

from src.utils.json_parsing import get_attributes_by_page, get_page_list, get_page_dimensions,get_box_coords_json, get_censor_type
from src.utils.json_parsing import get_align_boxes


from src.utils.feature_extraction import crop_patch, preprocess_alignment_roi, preprocess_roi, preprocess_blank_roi,load_image
from src.utils.feature_extraction import extract_features_from_blank_roi, extract_features_from_roi,censor_image
from src.utils.feature_extraction import extract_features_from_page, preprocess_page, extract_features_from_text_region, preprocess_text_region
from src.utils.alignment_utils import page_vote,compute_transformation, compute_misalignment,apply_transformation,enlarge_crop_coords
from src.utils.alignment_utils import plot_rois_on_image_polygons,plot_rois_on_image,plot_both_rois_on_image,template_matching
from src.utils.logging import FileWriter, initialize_logger
from src.utils.matching_utils import update_phash_matches, match_pages_phash, check_matching_correspondence, compare_pages_same_section, match_pages_text


def check_matching_correspondence(page_dict, pages_list):
    non_corresponding_subset=[]
    for img_id in pages_list:
        if page_dict[img_id]['match_phash']!=page_dict[img_id]['matched_page']:
            non_corresponding_subset.append(img_id)
    return non_corresponding_subset


def hamming_distance(hash1: np.ndarray, hash2: np.ndarray) -> int:
    return int(np.count_nonzero(hash1 ^ hash2))

def match_pages_phash(page_dict, template_dict, pages_list, templates_to_consider, gap_threshold=5, max_dist=18):
    #assume phash are already computed

    hashes_pages = []
    hashes_templates = []

    #length of pages and templates may be different if templates only has the ones to censor
    for img_id in pages_list:
        hashes_pages.append(page_dict[img_id]['page_phash'])
    for t_id in templates_to_consider:
        hashes_templates.append(template_dict[t_id]['page_phash'])

    # Cost matrix: Hamming distances
    cost = np.zeros((len(hashes_pages), len(hashes_templates)), dtype=np.int32)
    for i in range(len(hashes_pages)):
        for j in range(len(hashes_templates)):
            cost[i, j] = hamming_distance(hashes_pages[i], hashes_templates[j])

    # Hungarian assignment (minimize total cost)
    assignement = linear_sum_assignment(cost)
    row_ind, col_ind = assignement

    '''if the dimension of the two lists is different -> Every single template will be assigned to a page.
    The function will pick the subset of pages that results in the lowest total Hamming distance.
    The "leftover" pages will be ignored. Because linear_sum_assignment only returns a match for the smaller dimension, 
    the row_ind and col_ind arrays will only have a length equal to the number of templates'''
    
    matches = []
    for i, j in zip(row_ind, col_ind):
        matches.append({
            "page_index": pages_list[i],
            "template_index": templates_to_consider[j],
            "hamming": int(cost[i, j]),
        })
    
    #confident,report = assignment_confidence_phash(cost, assignement, gap_threshold=gap_threshold, max_dist=max_dist)

    matches_sorted = sorted(matches, key=lambda x: x["page_index"])

    return matches_sorted, cost #, confident, report

def update_phash_matches(matches_sorted,page_dict):
    for match in matches_sorted:
        img_id = match['page_index']
        page_dict[img_id]['match_phash']=match['template_index']
    return page_dict


def hungarian_min_cost(cost: np.ndarray):
    r, c = linear_sum_assignment(cost)
    return list(zip(r.tolist(), c.tolist()))


def assignment_confidence_phash(cost: np.ndarray, assignment, gap_threshold: int, max_dist: int):
    """
    Returns (is_confident, report_dict).
    Confidence logic:
      - per-row gap test: (2nd best - best) >= gap_threshold
      - matched distance <= max_dist
    """
    n = cost.shape[0]
    row_ind, col_ind = assignment
    row_to_col = {r: c for r, c in zip(row_ind, col_ind)}

    per_row = []
    confident = True

    for r in range(n):
        row = cost[r].astype(int)
        best = int(row.min())
        # second best: take smallest two
        sorted_vals = np.partition(row, 1)[:2]
        second = int(sorted_vals[1]) if len(sorted_vals) > 1 else 10**9
        gap = second - best

        chosen_c = row_to_col[r]
        chosen = int(cost[r, chosen_c])

        row_ok = (gap >= gap_threshold) and (chosen <= max_dist)
        if not row_ok:
            confident = False

        per_row.append({
            "row": r,
            "best": best,
            "second": second,
            "gap": gap,
            "chosen": chosen,
            "ok": row_ok
        })

    total = int(sum(cost[r, c] for r, c in zip(row_ind, col_ind)))
    avg = float(total) / float(n)

    report = {
        "total_cost": total,
        "avg_cost": avg,
        "gap_threshold": gap_threshold,
        "max_dist": max_dist,
        "per_row": per_row,
    }
    return confident, report

############     ################
############ OCR ################
############     ################


# -----------------------------
# Text normalization
# -----------------------------

def normalize_text(s: str) -> str:
    """
    Normalize in a way that tends to help comparisons:
    - Unicode normalize
    - lowercase
    - collapse whitespace
    - remove “weird” punctuation except basic ones (tweak to your needs)
    """
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()

    # keep letters/numbers/basic punctuation; drop the rest
    s = re.sub(r"[^\w\s\.\,\:\;\-\(\)\/%]", " ", s, flags=re.UNICODE)

    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


# -----------------------------
# Similarity measures
# -----------------------------

def sequence_similarity(a: str, b: str) -> float:
    """
    Character-level similarity in [0,1].
    """
    return SequenceMatcher(None, a, b).ratio()


def jaccard_similarity_tokens(a: str, b: str) -> float:
    """
    Token-level Jaccard similarity in [0,1].
    """
    ta = set(a.split())
    tb = set(b.split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def containment_similarity(a: str, b: str) -> float:
    """
    How much of the shorter text's tokens are contained in the longer text.
    Useful when one side has extra boilerplate.
    """
    ta = a.split()
    tb = b.split()
    sa, sb = set(ta), set(tb)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    small, big = (sa, sb) if len(sa) <= len(sb) else (sb, sa)
    return len(small & big) / max(1, len(small))


def compare_pages_same_section(raw1, raw2):
    n1 = normalize_text(raw1)
    n2 = normalize_text(raw2)

    return {
        "raw_text_1": raw1,
        "raw_text_2": raw2,
        "normalized_text_1": n1,
        "normalized_text_2": n2,
        "similarity_sequence": sequence_similarity(n1, n2),
        "similarity_jaccard_tokens": jaccard_similarity_tokens(n1, n2),
        "similarity_containment": containment_similarity(n1, n2),
    }

def match_pages_text(pages_list,templates_to_consider,similarity):
    #assume phash are already computed
    cost = similarity*(-1)+1
    # Hungarian assignment (minimize total cost)
    assignement = linear_sum_assignment(cost)
    row_ind, col_ind = assignement

    '''if the dimension of the two lists is different -> Every single template will be assigned to a page.
    The function will pick the subset of pages that results in the lowest total Hamming distance.
    The "leftover" pages will be ignored. Because linear_sum_assignment only returns a match for the smaller dimension, 
    the row_ind and col_ind arrays will only have a length equal to the number of templates'''
    
    matches = []
    for i, j in zip(row_ind, col_ind):
        matches.append({
            "page_index": pages_list[i],
            "template_index": templates_to_consider[j],
            "similarity": int(cost[i, j]),
        })
    
    # confident,report = assignment_confidence_text(cost, assignement, gap_threshold=gap_threshold, max_dist=max_dist)
    # decide how to evaluate the confidence

    matches_sorted = sorted(matches, key=lambda x: x["page_index"])

    return matches_sorted, cost #, confident, report

def assignment_confidence_text(cost: np.ndarray, assignment, gap_threshold: int, max_dist: int):
    """
    Returns (is_confident, report_dict).
    Confidence logic:
      - per-row gap test: (2nd best - best) >= gap_threshold
      - matched distance <= max_dist
    """
    n = cost.shape[0]
    row_to_col = {r: c for r, c in assignment}

    per_row = []
    confident = True

    for r in range(n):
        row = cost[r].astype(int)
        best = int(row.min())
        # second best: take smallest two
        sorted_vals = np.partition(row, 1)[:2]
        second = int(sorted_vals[1]) if len(sorted_vals) > 1 else 10**9
        gap = second - best

        chosen_c = row_to_col[r]
        chosen = int(cost[r, chosen_c])

        row_ok = (gap >= gap_threshold) and (chosen <= max_dist)
        if not row_ok:
            confident = False

        per_row.append({
            "row": r,
            "best": best,
            "second": second,
            "gap": gap,
            "chosen": chosen,
            "ok": row_ok
        })

    total = int(sum(cost[r, c] for r, c in assignment))
    avg = float(total) / float(n)

    report = {
        "total_cost": total,
        "avg_cost": avg,
        "gap_threshold": gap_threshold,
        "max_dist": max_dist,
        "per_row": per_row,
    }
    return confident, report

def initialize_sorting_dictionaries(sorted_files, root,mode='cv2'):
    ''' this function takes one json annotation file called root (for one template) 
    takes a list of file_paths and returns the page_dictionary, and template_dictionary (each initialized to the starting values)'''
    def find_corresponding_file(sorted_files, img_name):
        index=get_page_number(img_name)
        if index <= len(sorted_files):
            return sorted_files[index-1]
        return None

    pages_in_annotation = get_page_list(root)
    page_dictionary = {}
    template_dictionary = {}
    for p in pages_in_annotation:
        page_dictionary[p]={}
        template_dictionary[p]={}
    #iterate on the pages in a document and initialize their parameters
    for img_id in pages_in_annotation:
        
        page_dictionary[img_id]['img_id']=img_id
        img_name=f'page_{img_id}.png'
        page_dictionary[img_id]['img_name']=img_name 
        png_img_path = find_corresponding_file(sorted_files, img_name)
        page_dictionary[img_id]['img_path']=png_img_path
        page_dictionary[img_id]['img_size']=get_page_dimensions(root,img_id)
        page_dictionary[img_id]['template_matches']=0 #how many time this page was matched with a template
        page_dictionary[img_id]['shifts']=None #(shift_x,shift_y) for first qnd second region
        page_dictionary[img_id]['centers']=None #(shift_x,shift_y) for first qnd second region
        page_dictionary[img_id]['stored_template']=None # to store the features extracted from the align regions 
        #for the page (will be overwritten each time i compare with diff template)
        page_dictionary[img_id]['matched_page']=None #initially I assume the page is matched to the same index template
        page_dictionary[img_id]['page_phash']=None
        page_dictionary[img_id]['match_phash']=None
        page_dictionary[img_id]['text']=None

        censor_type=get_censor_type(root,img_id) 
        template_dictionary[img_id]['type']=censor_type
        template_dictionary[img_id]['align_boxes']=None #the coordinates of the align boxes for a template
        template_dictionary[img_id]['pre_computed_align']=None #the pre computed values for the align region in the template
        template_dictionary[img_id]['matched_to_this']=0
        template_dictionary[img_id]['page_phash']=None
        template_dictionary[img_id]['final_match']=None
        #template_dictionary[img_id]['text']=None
        
    #print(len(templates_to_consider))
    #perform the check on all the pages to censor or partially censor
    # i perform both the template matching and the phash check
    return page_dictionary,template_dictionary

def pre_load_images_to_censor(template_dictionary,page_dictionary, mode='csv'):
    ''' takes the initialized dictionaries and return the updated dictionaries (with pre-loaded values) and the 
    list of template ids'''
    templates_to_consider=[]
    pages_in_annotation = list(page_dictionary.keys())
    for img_id in pages_in_annotation:
        censor_type=template_dictionary[img_id]['type']
        png_img_path=page_dictionary[img_id]['img_path']
        #i load in memory only the pages that needs censoring or partial censoring at the beginning
        if censor_type!='N':
            img=load_image(png_img_path, mode=mode, verbose=False) #modify code to manage tiff and jpeg if needed
            page_dictionary[img_id]['img']=img.copy()
            templates_to_consider.append(img_id)
        else:
            page_dictionary[img_id]['img']=None
    return page_dictionary,template_dictionary, templates_to_consider

def pre_load_selected_templates(templates_to_consider,npy_dict, root, template_dictionary):
    for t_id in templates_to_consider:
        pre_computed = npy_dict[t_id]
        align_boxes, pre_computed_align = get_align_boxes(root,pre_computed,t_id) 
        template_dictionary[t_id]['align_boxes']=align_boxes
        template_dictionary[t_id]['pre_computed_align']=pre_computed_align
        template_dictionary[t_id]['page_phash']=pre_computed[-1]['page_phash']
        template_dictionary[t_id]['border_crop_pct']=pre_computed[-1]['border_crop_pct']
    return template_dictionary

def pre_load_image_properties(pages_to_consider,page_dictionary,template_dictionary,properties=[],mode='csv'):
    '''given a list of images and the properties to pre-compute it updates the appropriate 
    keys in the dictionary '''
    for img_id in pages_to_consider:
        if 'img' in properties:
            if page_dictionary[img_id]['img'] is None:
                img=load_image(page_dictionary[img_id]['img_path'], mode=mode, verbose=False) #modify code to manage tiff and jpeg if needed
                page_dictionary[img_id]['img']=img.copy()
        if 'phash' in properties: #should follow im loading because it requires the image to be in the dictionary already
            if page_dictionary[img_id]['page_phash'] is None:
                preprocessed_img = preprocess_page(page_dictionary[img_id]['img'])
                CROP_PATCH_PCTG = template_dictionary[img_id]['border_crop_pct'] #i can get this parameter from any page template really
                pre_comp = extract_features_from_page(preprocessed_img, mode=mode, verbose=False,to_compute=['page_phash'],border_crop_pct=CROP_PATCH_PCTG)
                page_dictionary[img_id]['page_phash']=pre_comp['page_phash'] #.copy() copy should not be needed if i reinitialize pre_comp in the loop
    return page_dictionary

def perform_template_matching(pairs_to_consider,page_dictionary,template_dictionary, n_align_regions,scale_factor):
    '''expects a list or a list of pairs, if only a list is provided it means we consider the list of pairs i,i j,j k,k ..
    the first element is the id of a page and the second is the id of the template'''
    if pairs_to_consider and isinstance(pairs_to_consider[0], (int, float)):
        #the input is a list of numbers
        new_pairs_to_consider=[]
        for i in pairs_to_consider:
            new_pairs_to_consider.append([i,i])
        pairs_to_consider = new_pairs_to_consider[:]
    
    for pair in pairs_to_consider:
        img_id,t_id = pair 
        align_boxes = template_dictionary[t_id]['align_boxes']
        pre_computed_align = template_dictionary[t_id]['pre_computed_align']
        shifts, centers, processed_rois = compute_misalignment(page_dictionary[img_id]['img'], align_boxes, page_dictionary[img_id]['img_size'], 
                                pre_computed_template=pre_computed_align,scale_factor=scale_factor) #recall this functions returns a shift for each good match
        #thus you expect len=2 for the shift variable, instead processed_rois returns all regions
        
        if len(shifts)==n_align_regions:
            page_dictionary[img_id]['shifts'] = shifts
            page_dictionary[img_id]['centers'] = centers 
            page_dictionary[img_id]['template_matches']=1
            page_dictionary[img_id]['stored_template'] = processed_rois #save the rois so i can re-use them without recomputing
            page_dictionary[img_id]['matched_page']=img_id
            template_dictionary[img_id]['matched_to_this']+=1
    return page_dictionary,template_dictionary

def perform_phash_matching(page_dictionary,template_dictionary, pages_list, templates_to_consider, 
                            gap_threshold,max_dist):
    matches_sorted, cost = match_pages_phash(page_dictionary,template_dictionary, pages_list, templates_to_consider, 
                            gap_threshold=gap_threshold,max_dist=max_dist) #As of now i don't consider the confidence of the matching, but I may in future versions
    page_dictionary = update_phash_matches(matches_sorted,page_dictionary)
    return page_dictionary