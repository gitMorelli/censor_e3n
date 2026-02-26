from src.utils.feature_extraction import preprocess_roi, preprocess_blank_roi,preprocess_alignment_roi, extract_features_from_roi
from src.utils.feature_extraction import profile_ncc, projection_profiles, edge_iou, ncc, dct_phash,count_black_pixels,count_connected_components
from src.utils.feature_extraction import phash_hamming_distance, binary_crc32, convert_to_grayscale
from src.utils.file_utils import deserialize_keypoints
import cv2  
import numpy as np
import math
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from src.utils.logging import FileWriter

######### Compute misalignment using ALIGN boxes #########
def enlarge_crop_coords(coords, scale_factor=1.2, img_shape=None):
    x_min, y_min, x_max, y_max = coords
    width = x_max - x_min
    height = y_max - y_min
    center_x = x_min + width / 2
    center_y = y_min + height / 2
    new_width = width * scale_factor
    new_height = height * scale_factor
    new_x_min = int(max(center_x - new_width / 2, 0))
    new_y_min = int(max(center_y - new_height / 2, 0))
    new_x_max = int(min(center_x + new_width / 2, img_shape[0] - 1)) if img_shape else int(center_x + new_width / 2)
    new_y_max = int(min(center_y + new_height / 2, img_shape[1] - 1)) if img_shape else int(center_y + new_height / 2)
    return (new_x_min, new_y_min, new_x_max, new_y_max)
def get_center(coords):
    x_min, y_min, x_max, y_max = coords
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    return center_x, center_y
def template_matching(f_roi, t_roi, coord, mode="cv2",threshold=0.7,shift_wr_center=(0,0)):
    w, h = t_roi.shape[1], t_roi.shape[0]
    

    # Template matching
    res = cv2.matchTemplate(f_roi, t_roi, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res) # positions are relative to top left of f_roi

    if max_val < threshold:
        # If no good match is found, return the original center
        center_x = coord[0] + (coord[2] - coord[0]) // 2
        center_y = coord[1] + (coord[3] - coord[1]) // 2
        return False,max_val,center_x, center_y
    # For TM_SQDIFF methods, the best match is min_loc; otherwise, max_loc
    top_left = max_loc
    center_x = top_left[0] + w // 2 
    center_y = top_left[1] + h // 2
    expected_x = (f_roi.shape[1] // 2) - shift_wr_center[0]
    expected_y = (f_roi.shape[0] // 2) - shift_wr_center[1] 
    shift_x = center_x - expected_x
    shift_y = center_y - expected_y
    return True,max_val,shift_x, shift_y


def compute_misalignment(filled_png, rois, img_shape, pre_computed_template, scale_factor=2,matching_threshold=0.7, pre_computed_rois=None,return_confidences=False):
    if pre_computed_rois:
        pre_computed=True
    else:
        pre_computed=False
    mode = "cv2"

    shifts = []
    centers = []
    processed_rois=[]
    confidences=[]
    for i,coord in enumerate(rois):
        center_x, center_y = get_center(coord)
        new_coord = enlarge_crop_coords(coord, scale_factor=scale_factor, img_shape=img_shape)
        new_center_x, new_center_y = get_center(new_coord)
        shift_wr_center = (new_center_x - center_x, new_center_y - center_y) #if the rescaled patch is not cropped 
        #we expect to find the template at w/2,h/2 in the referece frame of the enlarged patch; If it is cropped we expect to find it at -shift_wr_center
        t_roi = pre_computed_template[i]['full']
        if not pre_computed:
            f_roi = preprocess_alignment_roi(filled_png, new_coord, mode=mode, verbose=False)
        else:
            f_roi = pre_computed_rois[i]
        is_matched,max_val,shift_x, shift_y = template_matching(f_roi, t_roi, coord, mode=mode,shift_wr_center=shift_wr_center,threshold=matching_threshold)
        confidences.append(max_val)
        if is_matched: #only include regions for which you have a match
            shifts.append((shift_x, shift_y))
            centers.append((center_x, center_y))
        processed_rois.append(f_roi)
    if return_confidences:
        return shifts, centers,processed_rois,confidences
    return shifts, centers,processed_rois


def orb_matching(img,box,template_properties, top_n_matches=50, orb_nfeatures=2000, match_threshold=10, scale_factor=2):

    img_shape = (img.shape[1], img.shape[0]) # (width, height)  
    box = enlarge_crop_coords(box, scale_factor=scale_factor, img_shape=img_shape)
    # preprocess image
    preprocessed_patch = preprocess_roi(img, box, target_size=None)
    #compute orb features for the patch
    pre_comp = extract_features_from_roi(preprocessed_patch,to_compute=['orb'],orb_nfeatures=orb_nfeatures)
    kps_image , des_image = pre_comp['orb_kp'] , pre_comp['orb_des'] 

    kps_template, des_template = deserialize_keypoints(template_properties['orb_kp']), template_properties['orb_des']
    orb_kp=deserialize_keypoints(pre_comp[-1]['orb_kp'])

    # 2. Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des_image, des_template), key=lambda x: x.distance)

    # Use only the top 50 matches for stability
    good_matches = matches[:top_n_matches]

    if len(good_matches) > match_threshold:
        # Extract coordinates of matched points
        image_pts = np.float32([kps_image[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        template_pts = np.float32([kps_template[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2) 

        # 3. Find the Homography Matrix
        M, mask = cv2.findHomography(template_pts, image_pts, cv2.RANSAC, 5.0) #the template is transformed to the image
        #-> the scale, rotation and shift will be the parameters that bring from the template to the image

        # 4. Extract Info from Matrix M
        # M = [ [a, b, tx],
        #       [c, d, ty],
        #       [0, 0, 1 ] ]
        
        # Shift (Translation)
        shift_x = M[0, 2]
        shift_y = M[1, 2]

        # Scale
        # Derived from the change in length of the basis vectors
        scale_x = math.sqrt(M[0, 0]**2 + M[1, 0]**2)
        scale_y = math.sqrt(M[0, 1]**2 + M[1, 1]**2)
        avg_scale = (scale_x + scale_y) / 2

        # Rotation Angle (in degrees)
        angle = math.atan2(M[1, 0], M[0, 0]) * (180 / math.pi)

        return shift_x, shift_y, avg_scale, angle
    else:
        return None, None, None  # Not enough matches to compute transformation

def compute_distance(c1,c2):
    return np.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2)

def compute_transformation(shifts, centers):
    if len(shifts) < 2:
        return 1.0,0,0,0,(0,0)  # Not enough data to compute transformation
    max_dist = 0
    for i in range(len(shifts)):
        for j in range(i + 1, len(shifts)):
            c1 = centers[i]
            c2 = centers[j]
            dist = compute_distance(c1, c2)
            if dist > max_dist:
                max_dist = dist
                idx1, idx2 = i, j
    center_1=centers[idx1]
    center_2=centers[idx2]
    shift_1=shifts[idx1]
    shift_2=shifts[idx2]
    # Compute scale factor
    original_distance = compute_distance(center_1, center_2)
    shifted_center_1 = (center_1[0] + shift_1[0], center_1[1] + shift_1[1])
    shifted_center_2 = (center_2[0] + shift_2[0], center_2[1] + shift_2[1])
    shifted_distance = compute_distance(shifted_center_1, shifted_center_2)
    scale_factor = shifted_distance / original_distance if original_distance != 0 else 1.0
    # Compute shift
    shift_x = shifts[0][0]
    shift_y = shifts[0][1]
    # compute rotation angle
    delta_y = shifted_center_2[1] - shifted_center_1[1]
    delta_x = shifted_center_2[0] - shifted_center_1[0]
    shift_angle = np.arctan(delta_x / delta_y) - np.arctan((center_2[0] - center_1[0]) / (center_2[1] - center_1[1]))
    shift_angle_degrees = np.degrees(shift_angle)
    return scale_factor, shift_x, shift_y, shift_angle_degrees,center_1

def rotate_points_about_pivot(points, px, py, alpha_deg):
    a = np.deg2rad(alpha_deg)
    R = np.array([[np.cos(a), -np.sin(a)],
                  [np.sin(a),  np.cos(a)]], dtype=float)
    P = np.atleast_2d(points).astype(float)   # (N,2)
    rotated = (P - [px, py]) @ R.T + [px, py]
    return rotated

def apply_transformation(reference,coords, scale_factor, shift_x, shift_y, angle_degrees, inverse=False):
    #rotation is around reference box (upper left)
    if inverse:
        scale_factor = 1 / scale_factor
        shift_x = -shift_x
        shift_y = -shift_y
        angle_degrees = -angle_degrees
    angle_degrees = -angle_degrees
    x_min, y_min, x_max, y_max = coords
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    new_width = width * scale_factor
    new_height = height * scale_factor
    angle_radians = np.radians(angle_degrees)
    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)
    new_center_x = center_x + shift_x
    new_center_y = center_y + shift_y
    reference_x_min, reference_y_min = reference
    # Compute coordinates. Rotate the whole box around the reference point (sides may not be axis-aligned anymore)
    corners = np.array([
        [new_center_x - new_width/2, new_center_y - new_height/2],
        [new_center_x + new_width/2, new_center_y - new_height/2],
        [new_center_x + new_width/2, new_center_y + new_height/2],
        [new_center_x - new_width/2, new_center_y + new_height/2],
    ])
    corners_rot = rotate_points_about_pivot(corners, reference_x_min, reference_y_min, angle_degrees)

    return corners_rot #recall that now the box is not axis-aligned anymore corners_rot is a list of 4 (x,y) points

def plot_rois_on_image(image, rois, save_path, colors=None):

    h, w = image.shape[:2]

    # Create figure with size matching the image pixel dimensions
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)

    if colors is None:
        colors = ['red'] * len(rois)

    ax = plt.axes()
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Remove axes for clean output
    ax.axis('off')

    for i, coord in enumerate(rois):
        x_min, y_min, x_max, y_max = coord
        rect = plt.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=0.5,
            edgecolor=colors[i],
            facecolor='none'
        )
        ax.add_patch(rect)

    # Save without extra borders
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def plot_rois_on_image_polygons(image, rois, save_path, colors=None):

    h, w = image.shape[:2]

    # Maintain original image resolution
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = plt.axes()

    if colors is None:
        colors = ['red'] * len(rois)

    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.axis('off')

    for i, coord in enumerate(rois):
        pts = np.array(coord, dtype=float)  # expected shape: (4, 2)
        poly = Polygon(
            pts,
            closed=True,
            linewidth=0.5,
            edgecolor=colors[i],
            facecolor='none'
        )
        ax.add_patch(poly)

        # If labeling is needed, uncomment below
        # center_x, center_y = pts[:, 0].mean(), pts[:, 1].mean()
        # ax.text(center_x, center_y, str(i), color=colors[i], fontsize=12,
        #        ha='center', va='center')

    # Save as original resolution
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def plot_both_rois_on_image(image, rois_1, rois_2, save_path,
                            color_1="red", color_2="green"):
    """
    Draw rectangular ROIs (rois_1) and polygon ROIs (rois_2)
    while preserving the original image resolution.
    """

    h, w = image.shape[:2]

    # Create a figure matching the image's resolution
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = plt.axes()

    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.axis("off")

    # --- Draw rectangular ROIs ---
    for coord in rois_1:
        x_min, y_min, x_max, y_max = coord
        rect = plt.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=0.5,
            edgecolor=color_1,
            facecolor='none'
        )
        ax.add_patch(rect)

    # --- Draw polygon ROIs ---
    for coord in rois_2:
        pts = np.array(coord, dtype=float)  # expected shape (4,2)
        poly = Polygon(
            pts,
            closed=True,
            linewidth=0.5,
            edgecolor=color_2,
            facecolor='none'
        )
        ax.add_patch(poly)

    # Save image with original resolution
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def adjust_boundary_boxes(boxes, img_size_1, img_size_2, epsilon=2.0):
    """
    Adjusts box edges that coincide with img_size_1 boundaries 
    to match img_size_2 boundaries, without scaling internal coordinates.
    """
    w1, h1 = img_size_1
    w2, h2 = img_size_2
    
    # Convert to numpy array for vectorized operations
    boxes = np.array(boxes, dtype=float)
    
    # X-coordinates: Xtl (index 0) and Xbr (index 2)
    for i in [0, 2]:
        # If it was at/near the left edge (0), keep it at 0
        boxes[boxes[:, i] <= epsilon, i] = 0
        
        # If it was at/near the old right edge (w1), move it to the new right edge (w2)
        boxes[boxes[:, i] >= (w1 - epsilon), i] = w2

    # Y-coordinates: Ytl (index 1) and Ybr (index 3)
    for i in [1, 3]:
        # If it was at/near the top edge (0), keep it at 0
        boxes[boxes[:, i] <= epsilon, i] = 0
        
        # If it was at/near the old bottom edge (h1), move it to the new bottom edge (h2)
        boxes[boxes[:, i] >= (h1 - epsilon), i] = h2
        
    return boxes.tolist()
######### CHECK FOR ALIGNMENT using ROIs #########
# -----------------------------
# Decision logic per ROI
# -----------------------------
def roi_decision(f_roi,phash_hamm_thresh=8,
                 ncc_thresh=0.92,
                 edge_iou_thresh=0.75,
                 proj_ncc_thresh=0.90,t_roi=None,pre_computed_roi=None,
                 to_compute=['crc32','dct_phash', 'ncc','edge_iou','profile'],treshold_test=2,logger=None):
    """
    Returns True if the two ROIs are considered aligned/similar enough.
    Order of checks:
      0) Binary checksum (fast pass)
      1) DCT pHash (cv2-only)
      2) NCC
      3) Edge IoU
      4) Projection profiles (H & V)
    """
    tot_tests = len(to_compute)
    ok_tests = 0
    if pre_computed_roi:
        pre_computed=True
    else:
        pre_computed=False
    # --- 0) Binary checksum (Otsu binarize + CRC32)
    logger and logger.call_start(f'crc32')
    if 'crc32' in to_compute:
        if pre_computed:
            t_crc = pre_computed_roi['crc32']
        else:
            t_crc = binary_crc32(t_roi)
        f_crc = binary_crc32(f_roi)
        if t_crc == f_crc:
            ok_tests += 1
    logger and logger.call_end(f'crc32')

    # --- 1) DCT-based pHash (cv2.dct)
    logger and logger.call_start(f'dct')
    if 'dct_phash' in to_compute:
        if pre_computed:
            h1 = pre_computed_roi['dct_phash']#check that was precomputed with the same params
        else:
            h1 = dct_phash(t_roi, hash_size=8, dct_size=32) 
        h2 = dct_phash(f_roi, hash_size=8, dct_size=32)
        hdist = phash_hamming_distance(h1, h2)
        if hdist <= phash_hamm_thresh:
            ok_tests += 1
    logger and logger.call_end(f'dct')

    # --- 2) NCC on intensities
    logger and logger.call_start(f'ncc')
    if 'ncc' in to_compute:
        if pre_computed:
            t_roi = pre_computed_roi['full']
        ncc_value,_,_ = ncc(t_roi, f_roi)
        if ncc_value >= ncc_thresh:
            ok_tests += 1
    logger and logger.call_end(f'ncc')

    # --- 3) Edge IoU
    logger and logger.call_start(f'edge_iou')
    if 'edge_iou' in to_compute:
        if pre_computed:
            t_roi = pre_computed_roi['full']
        if edge_iou(t_roi, f_roi) >= edge_iou_thresh:
            ok_tests += 1
    logger and logger.call_end(f'edge_iou')

    # --- 4) Projection profiles (horizontal & vertical)
    logger and logger.call_start(f'profiles')
    if 'profile' in to_compute:
        if pre_computed:
            th = pre_computed_roi['profile_h']
            tv = pre_computed_roi['profile_v']
        else:
            th = projection_profiles(t_roi, axis=1)  # horizontal
            tv = projection_profiles(t_roi, axis=0)  # vertical
        fh = projection_profiles(f_roi, axis=1)
        fv = projection_profiles(f_roi, axis=0) 
        # Require both directions to be a strong match (or relax to either if desired)
        if profile_ncc(th, fh) >= proj_ncc_thresh and profile_ncc(tv, fv) >= proj_ncc_thresh:
            ok_tests += 1
    logger and logger.call_end(f'profiles')
    if ok_tests >= treshold_test:
        return True
    return False

def roi_blank_decision(f_roi, n_black_thresh=0.1, return_features = False,
                       t_roi=None,pre_computed_roi=None,to_compute=['cc','n_black'],threshold_test=1):
    ok_tests = 0
    if pre_computed_roi:
        pre_computed=True
    else:
        pre_computed=False
    
    if 'n_black' in to_compute:
        if pre_computed:
            t_n_black = pre_computed_roi['n_black']
        else:
            t_n_black = count_black_pixels(t_roi)
        f_n_black = count_black_pixels(f_roi)
        
        black_diff_to_template = (f_n_black-t_n_black)/(t_n_black+1e-10)

        if np.abs(black_diff_to_template) <= n_black_thresh:
            ok_tests += 1
    else:
        black_diff_to_template = None
    if 'cc' in to_compute:
        if pre_computed:
            t_cc = pre_computed_roi['cc']
        else:
            t_cc = count_connected_components(t_roi)
        f_cc = count_connected_components(f_roi)
        cc_difference_to_template= f_cc-t_cc
        if cc_difference_to_template == 0:
            ok_tests += 1
    else:
        cc_difference_to_template=None
    tests_ok=False
    if ok_tests >= threshold_test:
        tests_ok=True
    if return_features:
        return tests_ok,black_diff_to_template,cc_difference_to_template
    else:
        return tests_ok
# -----------------------------
# Page-level voting
# -----------------------------
def page_vote(filled_png, rois, min_votes=3,
              template_png=None,pre_computed_rois=None,logger=None):
    
    logger and logger.call_start('page_vote',block=True)

    votes = 0
    total = 0
    mode="cv2"
    if pre_computed_rois:
        pre_computed = True
    else:
        pre_computed = False
    #rois is the list of coordinates of the regions
    for i,coord in enumerate(rois[:-1]):
        # scale ROI coords
        decision=False

        logger and logger.call_start(f'preprocess_roi_{i}')
        f_roi = preprocess_roi(filled_png, coord, mode=mode, verbose=False)
        logger and logger.call_end(f'preprocess_roi_{i}')

        if not pre_computed:
            t_roi = preprocess_roi(template_png, coord, mode=mode, verbose=False)
            decision = roi_decision(f_roi, t_roi=t_roi)
        else:
            pre_comp_roi = pre_computed_rois[i]
            logger and logger.call_start(f'roi_decision_{i}')
            decision = roi_decision(f_roi, pre_computed_roi=pre_comp_roi,logger=logger)
            logger and logger.call_end(f'roi_decision_{i}')
        total += 1
        if decision:
            votes += 1
        # early exit: impossible to reach min_votes
        if votes + (len(rois)-total) < min_votes:
            logger and logger.call_end('page_vote',block=True)
            return False
    coord = rois[-1]  # blank ROI is the last one
    logger and logger.call_start(f'blank_preprocess')
    f_roi = preprocess_blank_roi(filled_png, coord, mode=mode, verbose=False)
    logger and logger.call_end(f'blank_preprocess')
    if not pre_computed:
        t_roi = preprocess_blank_roi(template_png, coord, mode=mode, verbose=False)
        decision = roi_blank_decision(f_roi, t_roi=t_roi)
    else:
        pre_comp_roi = pre_computed_rois[-1]
        logger and logger.call_start(f'blank_decision')
        decision = roi_blank_decision(f_roi, pre_computed_roi=pre_comp_roi)
        logger and logger.call_end(f'blank_decision')
    if decision:
        votes += 1
    
    logger and logger.call_end('page_vote',block=True)

    return votes >= min_votes