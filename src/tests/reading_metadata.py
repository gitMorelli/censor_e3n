#!/usr/bin/env python3
import argparse
import logging
import json
import os
import io
from src.utils.json_parsing import get_attributes_by_page
from time import perf_counter
import pikepdf
import fitz
from PIL import Image
from PyPDF2 import PdfReader



#JSON_PATH= "//vms-e34n-databr/2025-handwriting\\vscode\censor_e3n\data\q5_tests\\annotazioni" 

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


def main():
    args = parse_args()
    path_file = "//vms-e34n-databr/2025-handwriting\\data\\test_read_shared_files\\Q5\\A9Y0H8E8\\ISP00JLX_ISP01RGX.tif.pdf"
    path_file = "//smb-recherche-s1.prod-powerscale.intra.igr.fr/E34N_HANDWRITING$\\Fichiers\\Q5\\ISP00JLX_ISP01RGX.tif.pdf"

    print("trying with PyPDF2")
    # Carica il file
    reader = PdfReader(path_file)

    # Estrae i metadati standard
    meta = reader.metadata

    print(f"Autore: {meta.author}")
    print(f"Creatore: {meta.creator}")
    print(f"Produttore: {meta.producer}")
    print(f"Titolo: {meta.title}")

    with pikepdf.open(path_file) as pdf:
        info = pdf.docinfo
        print("Metadati Classici:")
        for key, value in info.items():
            print(f"{key}: {value}")
        print("Metadati formato moderno")
        meta = pdf.open_metadata()
        # Mostra i metadati in formato dizionario
        print("1")
        print(meta)
        for key, value in meta.items():
            print(f"{key}: {value}")
    
    print("provo a estrarre metadati dall'immagine tif")
    pdf_file = fitz.open(path_file)

    # Solitamente questi PDF hanno una sola pagina con una sola immagine
    for page_index in range(len(pdf_file)):
        page = pdf_file[page_index]
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_file.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Carichiamo i byte dell'immagine in Pillow
            image = Image.open(io.BytesIO(image_bytes))
            
            print(f"--- Metadati estratti dall'immagine nella pagina {page_index+1} ---")
            # Proviamo a leggere i tag (se presenti)
            if hasattr(image, 'tag_v2'):
                for tag, value in image.tag_v2.items():
                    print(f"Tag {tag}: {value}")
            else:
                print("L'immagine incorporata non contiene metadati interni.")

        pdf_file.close()

    return 0

if __name__ == "__main__":
    main()