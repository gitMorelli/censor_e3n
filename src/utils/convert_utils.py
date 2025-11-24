from pdf2image import convert_from_path
POPPLER_PATH = "C:\\Program Files\\poppler-25.11.0\\Library\\bin"

def pdf_to_png_images(pdf_path):
    """Convert PDF pages to PNG images.

    Args:
        args: Command-line arguments containing the PDF file path and poppler path.

    Returns:
        List of PNG images converted from the PDF.
    """
    images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
    return images