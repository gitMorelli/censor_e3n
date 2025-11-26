from pdf2image import convert_from_path
import fitz  # PyMuPDF
import os
POPPLER_PATH = "Z:\\programs\\Release-25.11.0-0\\poppler-25.11.0\\Library\\bin" #"C:\\Program Files\\poppler-25.11.0\\Library\\bin"

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

def save_as_is(doc,i,images_data,out_path):
    # get list of images for this specific page
    page_images = images_data[i]

    # take first image on this page
    img = page_images[0]
    
    xref = img[0]
    base_image = doc.extract_image(xref)
    image_bytes = base_image["image"]
    ext = base_image["ext"]
    out_path = out_path+f".{ext}"
    with open(out_path, "wb") as f:
        f.write(image_bytes)
    return 0
