#!/usr/bin/env python3
import argparse
import logging
import json
import os
from src.utils.json_parsing import parse_document_annotations
from time import perf_counter


JSON_PATH= "Z:\\vscode\censor_e3n\data\q5_tests\\annotazioni" 

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

    results = parse_document_annotations(data)

    # Print results to verify
    for item in results:
        print(f"Page: {item['page_number']} | Label: {item['label']} | Sub-Attr: {item['sub_attribute']}")
        print(f"  Geometry: {item['geometry']}")
        print("-" * 30)

    return 0

if __name__ == "__main__":
    main()