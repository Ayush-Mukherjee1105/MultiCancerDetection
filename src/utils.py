# src/utils.py
import os
import glob
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple

def find_images(base_dir: str) -> List[str]:
    """
    Recursively find image files under base_dir.
    Returns a list of absolute file paths.
    """
    exts = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.bmp')
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(base_dir, '**', e), recursive=True))
    # Normalize paths
    files = [os.path.normpath(f) for f in files]
    return files

def images_by_class(image_paths: List[str]) -> Dict[str, List[str]]:
    """
    Group image file paths by their parent directory name (used as class label).
    Example:
       /data/LC/benign/img1.jpg -> 'benign'
    """
    d = defaultdict(list)
    for p in image_paths:
        try:
            cls = Path(p).parent.name
        except Exception:
            cls = 'unknown'
        d[cls].append(p)
    return dict(d)
