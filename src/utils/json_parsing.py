
import json
import re

def extract_page_number(image_path):
    """
    Extracts the page number X from a filename ending in page_X.png.
    """
    # Regex to find 'page_' followed by digits and then '.png'
    match = re.search(r'page_(\d+)\.png$', image_path)
    if match:
        return int(match.group(1))
    return None

def normalize_attribute_list(data, key):
    """
    Helper to ensure attributes like 'roi_type' or 'censor_type' are always lists.
    In the JSON, single items appear as strings, multiple items as lists.
    """
    val = data.get(key, [])
    if isinstance(val, str):
        return [val]
    return val

def parse_document_annotations(json_content):
    """
    Parses the document JSON structure and extracts regions with their
    geometry, labels, and correlated sub-attributes.
    """
    parsed_data = []

    # Iterate through each page in the list
    for page_entry in json_content:
        # 1) Get Page Number
        image_path = page_entry.get('image', '')
        page_num = extract_page_number(image_path)
        
        # Prepare sub-attribute lists (normalize strings to lists for consistent indexing)
        roi_types = normalize_attribute_list(page_entry, 'roi_type')
        censor_types = normalize_attribute_list(page_entry, 'censor_type')
        
        # Counters to track which sub-attribute we are on for this page
        roi_counter = 0
        censor_counter = 0
        
        # 2) Iterate on bounding box regions
        labels = page_entry.get('label', [])
        
        for region in labels:
            # 3) Extract Geometry
            geometry = {
                'x': region.get('x'),
                'y': region.get('y'),
                'width': region.get('width'),
                'height': region.get('height')
            }
            
            # 4) Extract rectanglelabels
            # It is a list in the JSON (e.g., ["roi"] or ["censor"])
            rect_labels = region.get('rectanglelabels', [])
            primary_label = rect_labels[0] if rect_labels else None
            
            # 5) Extract sub_attribute based on label type and order
            sub_attribute = None
            
            if primary_label == 'roi':
                if roi_counter < len(roi_types):
                    sub_attribute = roi_types[roi_counter]
                    roi_counter += 1
            elif primary_label == 'censor':
                if censor_counter < len(censor_types):
                    sub_attribute = censor_types[censor_counter]
                    censor_counter += 1
            
            # Compile the extracted info
            parsed_data.append({
                'page_number': page_num,
                'geometry': geometry,
                'label': primary_label,
                'sub_attribute': sub_attribute,
                'original_width': region.get('original_width'),
                'original_height': region.get('original_height')
            })
            
    return parsed_data

# --- Example Usage with the provided JSON data ---

# Load the JSON data (assuming raw_json_string contains the content from your file)
# In a real scenario: with open('doc_5.json', 'r') as f: data = json.load(f)

# Run the parser
# results = parse_document_annotations(raw_json_input)

# Print results to verify
'''for item in results:
    print(f"Page: {item['page_number']} | Label: {item['label']} | Sub-Attr: {item['sub_attribute']}")
    print(f"  Geometry: {item['geometry']}")
    print("-" * 30)'''