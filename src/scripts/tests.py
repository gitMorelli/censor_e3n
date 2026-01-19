#!/usr/bin/env python3
import argparse
import logging
import json
import os
from src.utils.json_parsing import get_attributes_by_page
from time import perf_counter


JSON_PATH= "//vms-e34n-databr/2025-handwriting\\vscode\censor_e3n\data\q5_tests\\annotazioni" 

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
    with open(JSON_PATH+'\\doc_5.json', 'r') as f: data = json.load(f)

    attributes_page_3 = get_attributes_by_page(data, 3) # Assumendo che 'data' sia definito

    print(f"Trovate {len(attributes_page_3)} regioni nella pagina 3:")
    for attr in attributes_page_3:
        print(attr)

    return 0

if __name__ == "__main__":
    main()