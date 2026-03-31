import argparse
import pandas as pd
import os
from pathlib import Path
import fitz  # PyMuPDF
import numpy as np
import cv2
from time import perf_counter
import time
from PIL import Image
import tifffile
import imagecodecs

from src.utils.file_utils import get_basename, create_folder, remove_folder, load_annotation_tree

QUESTIONNAIRE="13"
def main():
    args = parse_args()
    main_path = "//vms-e34n-databr/2025-handwriting\\data\\experiment_pdf_extraction"
    file_path = os.path.join(main_path, "E3NQ13-0130-20230530-0.pdf")
    out_path = os.path.join(main_path, "extracted_images")
    lim_pages=3
    compression_levels = [3,4,5,6,7,8, 9]  # 0: no compression, 9: maximum compression
    if os.path.exists(out_path):
        remove_folder(out_path)
    #put a delay
    time.sleep(2)
    create_folder(out_path)

    images_list = []
    with fitz.open(file_path) as doc:
        n_pages = len(doc)
        for i in range(n_pages): 
            page = doc[i]
            # get list of images for this specific page
            page_images = page.get_images(full=True)  

            # take first image on this page
            img = page_images[0]
            
            xref = img[0]
            smask = img[1]  # xref of its /SMask
            bpc_originale = img[4] 
    
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]
            size_before = len(image_bytes) # Size in bytes 
            #print(1)
    
            # Convert bytes to a numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            # Decode the array into an OpenCV image (BGR format)
            #cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            cv_img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            bit_depth_cv = cv_img.dtype.itemsize * 8
            images_list.append(cv_img)
            # 3. Metrics AFTER imdecode
            shape_after = cv_img.shape
            # .nbytes returns the total bytes consumed by the elements of the array
            size_after = cv_img.nbytes 
            
            if i==0:
                print(f"--- Pagina {i} ---")
                print(f"Bit Depth Originale (PDF): {bpc_originale} bit")
                print('smaks value:', smask)
                print(f"Bit Depth OpenCV: {bit_depth_cv} bit (Dtype: {cv_img.dtype})")
                print('smaks filter:', base_image["smask"])
                print('colorspace:',base_image["colorspace"])
                print(f"--- Image Metrics (xref: {xref}) ---")
                print(f"Size Before (Compressed): {size_before / 1024:.2f} KB")
                print(f"Size After (In RAM):      {size_after / 1024:.2f} KB")
                print(f"Inflation Factor:         {size_after / size_before:.2f}x")
                print(f"Array Shape (H, W, C):    {shape_after}")
                print(f"Data Type:                {cv_img.dtype}")
            
            save_path = os.path.join(out_path,'from_bytes',f"page_{i}.{ext}")
            create_folder(os.path.dirname(save_path))
            with open(save_path, "wb") as f:
                f.write(image_bytes)
    
    t_0 = perf_counter()
    #create a new pdf document using fitz and insert the extracted images as pages
    new_doc = fitz.open()  # Create a new, empty PDF
    for cv_img in images_list:
        # 1. Convert OpenCV BGR(A) to PNG bytes
        # We use .png to keep it lossless, or .jpg for smaller size
        success, encoded_image = cv2.imencode(".png", cv_img)
        if not success:
            continue
        img_bytes = encoded_image.tobytes()

        # 2. Create a new page matching the image dimensions
        # height, width = cv_img.shape[:2]
        page = new_doc.new_page(width=cv_img.shape[1], height=cv_img.shape[0])

        # 3. Insert the image to fill the page
        page.insert_image(page.rect, stream=img_bytes)
    save_path = os.path.join(out_path,f"new_images_document.pdf")
    new_doc.save(save_path)
    new_doc.close()
    t_1 = perf_counter()
    print(f"Time taken to save new PDF with images: {t_1 - t_0:.4f} seconds")

    t_0 = perf_counter()
    #create anew pdf with pil
    pil_images = []
    for img in images_list:
        # Convert BGR (OpenCV) → RGB (PIL)
        pil_img = Image.fromarray(img).convert("L")

        pil_images.append(pil_img)
    # Save as PDF
    save_path = os.path.join(out_path,f"PIL_document.pdf")
    pil_images[0].save(
        save_path,
        save_all=True,
        append_images=pil_images[1:]
    )
    t_1 = perf_counter()
    print(f"Time taken to save new PDF with PIL: {t_1 - t_0:.4f} seconds")

    log_times = []
    for c in compression_levels:
        t_0 = perf_counter()
        for i in range(min(n_pages,lim_pages)):
            cv_img = images_list[i]
            #save the decoded image as well
            save_path = os.path.join(out_path,f"page_{i}_decoded_level_{c}.{ext}")
            cv2.imwrite(save_path, cv_img,[cv2.IMWRITE_PNG_COMPRESSION, c])
        t_1 = perf_counter()
        log_times.append((c, t_1 - t_0))
    # Print summary of times with respect to standard
    print("\nSummary of Compression Times:")
    for c, t in log_times:
        print(f"Compression level {c}: Time taken = {t:.4f} seconds")
        print(f"Ratio to standard (level 3): {t/log_times[0][1]:.2f}x")
    
    #chhosen level
    t_0 = perf_counter()
    for i in range(len(images_list)):
        cv_img = images_list[i]
        create_folder(os.path.join(out_path,"extracted_pages"))
        save_path = os.path.join(out_path,"extracted_pages",f"page_{i}_decoded_level_6.{ext}")
        cv2.imwrite(save_path, cv_img,[cv2.IMWRITE_PNG_COMPRESSION, 6])
    t_1 = perf_counter()
    print(f"Time taken to save extracted pages with chosen compression: {t_1 - t_0:.4f} seconds")
    
    #standard level
    t_0 = perf_counter()
    for i in range(len(images_list)):
        cv_img = images_list[i]
        create_folder(os.path.join(out_path,"extracted_pages_standard"))
        save_path = os.path.join(out_path,"extracted_pages_standard",f"page_{i}_decoded_level_3.{ext}")
        cv2.imwrite(save_path, cv_img)
    t_1 = perf_counter()
    print(f"Time taken to save extracted pages with standard compression: {t_1 - t_0:.4f} seconds")
    
    #1bit cv2
    for i in range(len(images_list)):
        cv_img = images_list[i]
        _, binary_img = cv2.threshold(cv_img, 127, 255, cv2.THRESH_BINARY)
        create_folder(os.path.join(out_path,"tiff_cv2_1bit"))
        save_path = os.path.join(out_path,"tiff_cv2_1bit",f"page_{i}_decoded_level.{ext}")
        cv2.imwrite(save_path, binary_img, [cv2.IMWRITE_TIFF_COMPRESSION, 4])


    #1bit
    t_0 = perf_counter()
    # Supponiamo che 'images_list' sia la tua lista di array OpenCV (8-bit)
    for i, cv_img in enumerate(images_list):
        
        # 1. Converti l'array OpenCV in un oggetto PIL
        # Se cv_img è grayscale (2D), PIL lo riconosce come 'L' (8-bit)
        pil_img = Image.fromarray(cv_img)
        
        # 2. Riconverti in 1-bit (BI-TONALE)
        # Questo è il passaggio chiave per passare da 8 bit a 1 bit per pixel
        # '1' = bilevel (bianco e nero puro, senza grigi)
        pil_img_1bit = pil_img.convert('1')
        
        # 3. Salva come PNG
        # optimize=True: forza l'encoder a cercare la migliore compressione possibile
        filename = os.path.join(out_path,'1bit',f"1bit_{i:03d}.png")
        create_folder(os.path.dirname(filename))
        pil_img_1bit.save(filename, format="PNG", optimize=True)
    t_1 = perf_counter()
    print(f"Time taken to save 1-bit PNGs: {t_1 - t_0:.4f} seconds")

    t_0 = perf_counter()
    # 1bit_tiff
    for i, cv_img in enumerate(images_list):
        
        # 1. Pulizia con OpenCV (Opzionale ma consigliata)
        # Assicuriamoci che l'immagine sia "netta" (solo 0 o 255) per evitare rumore
        _, binary_cv = cv2.threshold(cv_img, 127, 255, cv2.THRESH_BINARY)
        
        # 2. Converti in PIL e poi a 1-bit
        pil_img = Image.fromarray(binary_cv).convert('1')
        
        # 3. Salva in TIFF con compressione CCITT Group 4
        # Questa compressione funziona SOLO su immagini a 1-bit
        filename = os.path.join(out_path,'1bit_tiff',f"1bit_{i:03d}.png")
        create_folder(os.path.dirname(filename))
        pil_img.save(filename, compression="tiff_ccitt")
    t_1 = perf_counter()
    print(f"Time taken to save 1-bit TIFFs: {t_1 - t_0:.4f} seconds")

    #1bit tiff library
    t_0 = perf_counter()
    # 1bit_tiff
    for i, cv_img in enumerate(images_list):
        
        # 1. Binarizzazione con OpenCV
        # Otteniamo un array con soli 0 e 255
        _, binary_img = cv2.threshold(cv_img, 127, 255, cv2.THRESH_BINARY)
        bitonal_img = binary_img > 0
        # 2. Salvataggio con tifffile
        # compression=4 corrisponde a CCITT Group 4 (T.6)
        # Nota: tifffile richiede che l'array sia uint8 o booleano
        filename = os.path.join(out_path,'tifflibrary',f"1bit_{i:03d}.png")
        create_folder(os.path.dirname(filename))
        tifffile.imwrite(
            filename, 
            binary_img, 
            photometric='minisblack', 
            compression='adobe_deflate', # Algoritmo ZIP-based
            compressionargs={'level': 9} # Massimo livello di compressione (1-9)
            #compression='ccitt_t6', # Questo è il Group 4 dei fax
            #planarconfig='separate'
        )
    t_1 = perf_counter()
    print(f"Time taken to save 1-bit TIFFs with tiffflibrary: {t_1 - t_0:.4f} seconds")




    

def parse_args():
    """Handle command-line arguments."""
    parser = argparse.ArgumentParser(description="Test script")
    '''parser.add_argument(
        "-f", "--folder_path",
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
    )'''
    return parser.parse_args()

if __name__ == "__main__":
    main()