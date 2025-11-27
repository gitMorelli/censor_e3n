from pathlib import Path
from typing import List, Union
from PIL import Image
import os
import shutil

def list_files_with_extension(
    folder: Union[str, Path],
    extension: Union[str, List[str], None] = None,
    recursive: bool = False,
) -> List[Path]:
    """
    Return file paths in `folder` that match `extension`.

    Args:
        folder: Directory to search (path string or Path).
        extension: File extension to match (with or without leading dot, e.g. "txt" or ".txt"),
                   a list of such extensions, or None/"*" / "all" to include all files.
        recursive: If True, search subdirectories recursively.

    Returns:
        List[Path]: Sorted list of matching file paths.

    Raises:
        ValueError: If `folder` does not exist or is not a directory.
    """
    folder_path = Path(folder)
    if not folder_path.exists() or not folder_path.is_dir():
        raise ValueError(f"Folder does not exist or is not a directory: {folder}")

    # treat None, "*", or "all" as match-all
    if extension is None or (isinstance(extension, str) and extension in ("*", "all")):
        iterator = folder_path.rglob("*") if recursive else folder_path.iterdir()
        return sorted([p for p in iterator if p.is_file()])

    # handle list/tuple of extensions
    if isinstance(extension, (list, tuple)):
        exts = [(e if e.startswith(".") else f".{e}") for e in extension if isinstance(e, str)]
        results = set()
        for ext in exts:
            pattern = f"*{ext}"
            iterator = folder_path.rglob(pattern) if recursive else folder_path.glob(pattern)
            for p in iterator:
                if p.is_file():
                    results.add(p)
        return sorted(results)

    # single extension string
    ext = extension if extension.startswith(".") else f".{extension}"
    pattern = f"*{ext}"
    iterator = folder_path.rglob(pattern) if recursive else folder_path.glob(pattern)
    return sorted([p for p in iterator if p.is_file()])

def get_basename(file_path: Union[str, Path], remove_extension: bool = False) -> str:
    """
    Return the base name (final path component) of `file_path`.

    Args:
        file_path: Path or path string to the file.
        remove_extension: If True, return the stem (name without suffix); otherwise return the full name.

    Returns:
        str: Base name or stem of the file path.
    """
    p = Path(file_path)
    return p.stem if remove_extension else p.name

def create_folder(folder: Union[str, Path], parents: bool = True, exist_ok: bool = True) -> Path:
    """
    Create the directory `folder` if it does not exist and return its Path.

    Args:
        folder: Directory path (string or Path).
        parents: If True, create parent directories as needed.
        exist_ok: Passed to Path.mkdir(); if False and the directory already exists an error is raised.

    Returns:
        Path: Path object for the created (or existing) directory.

    Raises:
        ValueError: If a file exists at `folder`.
        OSError: If the directory cannot be created.
    """
    p = Path(folder)
    if p.exists():
        if p.is_file():
            raise ValueError(f"Path exists and is a file: {folder}")
        return p
    p.mkdir(parents=parents, exist_ok=exist_ok)
    return p

def save_to_png_safe(input_path, output_path):
    with Image.open(input_path) as im:
        icc = im.info.get("icc_profile")
        im.save(
            output_path,
            format="PNG",
            optimize=False,
            icc_profile=icc,
        )
    return 0

def list_subfolders(folder: Union[str, Path], recursive: bool = False, include_hidden: bool = False) -> List[Path]:
    """
    Return directories contained in `folder`.

    Args:
        folder: Directory to list (path string or Path).
        recursive: If True, include subdirectories at all depths.
        include_hidden: If False, exclude entries whose name starts with a dot.

    Returns:
        List[Path]: Sorted list of subdirectory Paths.

    Raises:
        ValueError: If `folder` does not exist or is not a directory.
    """
    folder_path = Path(folder)
    if not folder_path.exists() or not folder_path.is_dir():
        raise ValueError(f"Folder does not exist or is not a directory: {folder}")

    iterator = folder_path.rglob("*") if recursive else folder_path.iterdir()
    subdirs = [
        p for p in iterator
        if p.is_dir() and (include_hidden or not p.name.startswith("."))
    ]
    return sorted(subdirs)

def check_name_matching(annotation_file_names, template_folder_names, logger):
    ann_set = set(annotation_file_names)
    tpl_set = set(template_folder_names)

    if ann_set != tpl_set:
        missing = sorted(list(ann_set - tpl_set))
        extra = sorted(list(tpl_set - ann_set))
        logger.error("Annotation files and template folders do not match.")
        if missing:
            logger.error("Annotations without corresponding template folders: %s", missing)
        if extra:
            logger.error("Template folders without corresponding annotations: %s", extra)
        return 1

    logger.info("Annotation filenames and template folder names match (%d items).", len(ann_set))

def get_page_number(file_name: str) -> int:
    """
    Extract the page number from a filename formatted as 'page_XX.png'.

    Args:
        file_name: Filename string.

    Returns:
        int: Extracted page number.

    Raises:
        ValueError: If the filename does not match the expected format.
    """
    base_name = Path(file_name).stem
    try:
        page_number = int(base_name.split("_")[-1])
    except (IndexError, ValueError):
        raise ValueError(f"Filename does not contain a valid page number: {file_name}")
    return page_number

def sort_files_by_page_number(file_paths: List[Union[str, Path]]) -> List[Path]:
    """
    Sort a list of file paths based on the page number extracted from their filenames.

    Args:
        file_paths: List of file paths (strings or Paths)."""
    return sorted(file_paths, key=lambda p: get_page_number(str(p)))
    
def remove_folder(folder_path):
    """
    Delete a folder and all of its contents.

    Parameters:
        folder_path (str): Path to the folder to delete.
    """
    if os.path.exists(folder_path):
        try:
            shutil.rmtree(folder_path)
            print(f"Folder removed: {folder_path}")
        except Exception as e:
            print(f"Error removing folder: {e}")
    else:
        print("Folder does not exist.")