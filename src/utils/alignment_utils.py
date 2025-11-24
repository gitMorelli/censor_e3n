from src.utils.feature_extraction import preprocess_roi, preprocess_blank_roi,preprocess_alignment_roi
from src.utils.feature_extraction import profile_ncc, projection_profiles, edge_iou, ncc, dct_phash,count_black_pixels,count_connected_components
from src.utils.feature_extraction import phash_hamming_distance, binary_crc32, convert_to_grayscale
import cv2  
import numpy as np
from matplotlib.patches import Polygon

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
        return False,center_x, center_y
    # For TM_SQDIFF methods, the best match is min_loc; otherwise, max_loc
    top_left = max_loc
    center_x = top_left[0] + w // 2 
    center_y = top_left[1] + h // 2
    expected_x = (f_roi.shape[1] // 2) - shift_wr_center[0]
    expected_y = (f_roi.shape[0] // 2) - shift_wr_center[1]
    shift_x = center_x - expected_x
    shift_y = center_y - expected_y
    return True,shift_x, shift_y
def compute_misalignment(filled_png, rois, img_shape, scale_factor=2,template_png=None, pre_computed_rois=None):
    if pre_computed_rois:
        pre_computed=True
    else:
        pre_computed=False
    mode = "cv2"

    shifts = []
    centers = []
    for i,coord in enumerate(rois):
        if not pre_computed:
            t_roi = preprocess_alignment_roi(template_png, coord, mode=mode, verbose=False)
        else:
            t_roi = pre_computed_rois[i]['full']
        center_x, center_y = get_center(coord)
        new_coord = enlarge_crop_coords(coord, scale_factor=scale_factor, img_shape=img_shape)
        new_center_x, new_center_y = get_center(new_coord)
        shift_wr_center = (new_center_x - center_x, new_center_y - center_y) #if the rescaled patch is not cropped 
        #we expect to find the template at w/2,h/2 in the referece frame of the enlarged patch; If it is cropped we expect to find it at -shift_wr_center
        f_roi = preprocess_alignment_roi(filled_png, new_coord, mode=mode, verbose=False)
        is_matched,shift_x, shift_y = template_matching(f_roi, t_roi, coord, mode=mode,shift_wr_center=shift_wr_center)
        if is_matched: #only include regions for which you have a match
            shifts.append((shift_x, shift_y))
            centers.append((center_x, center_y))
    return shifts, centers

def compute_distance(c1,c2):
    return np.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2)

def compute_transformation(shifts, centers):
    if len(shifts) < 2:
        return 1.0,0,0,0  # Not enough data to compute transformation
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

def plot_rois_on_image(image, rois, save_path,color="red"):
    import matplotlib.pyplot as plt
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for i, coord in enumerate(rois):
        x_min, y_min, x_max, y_max = coord
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=0.5, edgecolor=color, facecolor='none')
        plt.gca().add_patch(rect)
        plt.text(center_x, center_y, str(i), color='blue', fontsize=12, ha='center', va='center')
    plt.savefig(save_path)
    plt.close()

def plot_rois_on_image_polygons(image, rois, save_path,color="red"):
    import matplotlib.pyplot as plt
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax = plt.gca()
    for i, coord in enumerate(rois):
        # coord is expected to be an iterable of four (x,y) points
        pts = np.array(coord, dtype=float)  # shape (4,2)
        poly = Polygon(pts, closed=True, linewidth=0.5, edgecolor=color, facecolor='none')
        ax.add_patch(poly)
        # centroid for labeling
        center_x = pts[:, 0].mean()
        center_y = pts[:, 1].mean()
        plt.text(center_x, center_y, str(i), color='blue', fontsize=12, ha='center', va='center')
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_both_rois_on_image(image, rois_1, rois_2, save_path,color_1="red", color_2="green"):
    import matplotlib.pyplot as plt
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for i, coord in enumerate(rois_1):
        x_min, y_min, x_max, y_max = coord
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=0.5, edgecolor=color_1, facecolor='none')
        plt.gca().add_patch(rect)
        plt.text(center_x, center_y, str(i), color='blue', fontsize=12, ha='center', va='center')
    for i, coord in enumerate(rois_2):
        x_min, y_min, x_max, y_max = coord
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=0.5, edgecolor=color_2, facecolor='none')
        plt.gca().add_patch(rect)
        plt.text(center_x, center_y, str(i), color='blue', fontsize=12, ha='center', va='center')
    plt.savefig(save_path)
    plt.close()
######### CHECK FOR ALIGNMENT using ROIs #########
# -----------------------------
# Decision logic per ROI
# -----------------------------
def roi_decision(f_roi,phash_hamm_thresh=8,
                 ncc_thresh=0.92,
                 edge_iou_thresh=0.75,
                 proj_ncc_thresh=0.90,t_roi=None,pre_computed_roi=None,
                 to_compute=['crc32','dct_phash', 'ncc','edge_iou','profile'],treshold_test=2):
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
    if 'crc32' in to_compute:
        if pre_computed:
            t_crc = pre_computed_roi['crc32']
        else:
            t_crc = binary_crc32(t_roi)
        f_crc = binary_crc32(f_roi)
        if t_crc == f_crc:
            ok_tests += 1

    # --- 1) DCT-based pHash (cv2.dct)
    if 'dct_phash' in to_compute:
        if pre_computed:
            h1 = pre_computed_roi['dct_phash']#check that was precomputed with the same params
        else:
            h1 = dct_phash(t_roi, hash_size=8, dct_size=32) 
        h2 = dct_phash(f_roi, hash_size=8, dct_size=32)
        hdist = phash_hamming_distance(h1, h2)
        if hdist <= phash_hamm_thresh:
            ok_tests += 1

    # --- 2) NCC on intensities
    if 'ncc' in to_compute:
        if pre_computed:
            t_roi = pre_computed_roi['full']
        ncc_value,_,_ = ncc(t_roi, f_roi)
        if ncc_value >= ncc_thresh:
            ok_tests += 1

    # --- 3) Edge IoU
    if 'edge_iou' in to_compute:
        if pre_computed:
            t_roi = pre_computed_roi['full']
        if edge_iou(t_roi, f_roi) >= edge_iou_thresh:
            ok_tests += 1

    # --- 4) Projection profiles (horizontal & vertical)
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
    if ok_tests >= treshold_test:
        return True
    return False

def roi_blank_decision(f_roi, n_black_thresh=0.1,
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
        
        if np.abs(f_n_black-t_n_black)/t_n_black <= n_black_thresh:
            ok_tests += 1
    if 'cc' in to_compute:
        if pre_computed:
            t_cc = pre_computed_roi['cc']
        else:
            t_cc = count_connected_components(t_roi)
        f_cc = count_connected_components(f_roi)
        if f_cc == t_cc:
            ok_tests += 1
    if ok_tests >= threshold_test:
        return True
    return False
# -----------------------------
# Page-level voting
# -----------------------------
def page_vote(filled_png, rois, min_votes=3,
              template_png=None,pre_computed_rois=None):
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
        f_roi = preprocess_roi(filled_png, coord, mode=mode, verbose=False)
        if not pre_computed:
            t_roi = preprocess_roi(template_png, coord, mode=mode, verbose=False)
            decision = roi_decision(f_roi, t_roi=t_roi)
        else:
            pre_comp_roi = pre_computed_rois[i]
            decision = roi_decision(f_roi, pre_computed_roi=pre_comp_roi)
        total += 1
        if decision:
            votes += 1
        # early exit: impossible to reach min_votes
        if votes + (len(rois)-total) < min_votes:
            return False
    coord = rois[-1]  # blank ROI is the last one
    f_roi = preprocess_blank_roi(filled_png, coord, mode=mode, verbose=False)
    if not pre_computed:
        t_roi = preprocess_blank_roi(template_png, coord, mode=mode, verbose=False)
        decision = roi_blank_decision(f_roi, t_roi=t_roi)
    else:
        pre_comp_roi = pre_computed_rois[-1]
        decision = roi_blank_decision(f_roi, pre_computed_roi=pre_comp_roi)
    if decision:
        votes += 1

    return votes >= min_votes