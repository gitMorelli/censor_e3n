#!/usr/bin/env python3
import argparse
import logging
import os
from src.utils.convert_utils import pdf_to_png_images
from src.utils.file_utils import list_files_with_extension, get_basename, create_folder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Handle command-line arguments."""
    parser = argparse.ArgumentParser(description="Script to convert PDF template pages to PNG images.")
    parser.add_argument(
        "-f", "--folder_path",
        default="C:\\Users\\andre\\PhD\\Datasets\\e3n\\e3n templates",
        help="Path to the PDF file to convert",
    )
    parser.add_argument(
        "-s", "--save_path",
        default="C:\\Users\\andre\\PhD\\Datasets\\e3n\\e3n templates\\png_output",
        help="Directory to save the converted PNG images",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)

    pdf_path = args.folder_path
    save_path = args.save_path

    logger.info("Starting PDF -> PNG conversion")
    logger.debug("Input folder: %s", pdf_path)
    logger.debug("Output folder: %s", save_path)

    pdf_files = list_files_with_extension(pdf_path, "pdf", recursive=False)
    logger.info("Found %d PDF file(s) in %s", len(pdf_files), pdf_path)
    if not pdf_files:
        logger.warning("No PDF files found. Exiting.")
        return 0

    pdf_names = [get_basename(pdf_file, remove_extension=True) for pdf_file in pdf_files]
    for i, pdf_file in enumerate(pdf_files):
        logger.info("Processing file %d/%d: %s", i + 1, len(pdf_files), pdf_file)
        try:
            images = pdf_to_png_images(pdf_file)
        except Exception:
            logger.exception("Failed to convert PDF to images: %s", pdf_file)
            continue

        for j, image in enumerate(images):
            doc_path = os.path.join(save_path, pdf_names[i])
            try:
                create_folder(doc_path, parents=True, exist_ok=True)
                out_path = os.path.join(doc_path, f"page_{j + 1}.png")
                logger.debug("Saving image to %s", out_path)
                image.save(out_path, "PNG")
                logger.info("Saved %s", out_path)
            except Exception:
                logger.exception("Failed to save image for %s page %d", pdf_file, j + 1)

    logger.info("Conversion finished")
    return 0


if __name__ == "__main__":
    main()