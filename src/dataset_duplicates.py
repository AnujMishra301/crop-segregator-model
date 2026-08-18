"""
Duplicate Image Detector Module
Identifies exact duplicates (MD5 hash) and perceptual near-duplicates (aHash)
across image directories to prevent dataset redundancy.
"""

import os
import hashlib
from PIL import Image
import numpy as np

def compute_md5(filepath):
    """Computes exact MD5 checksum of file bytes."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def compute_average_hash(image_path, hash_size=8):
    """Computes 64-bit Average Perceptual Hash (aHash) for near-duplicate image detection."""
    try:
        with Image.open(image_path) as img:
            img = img.convert('L').resize((hash_size, hash_size), Image.Resampling.BILINEAR)
            pixels = np.array(img.getdata(), dtype=float)
            avg = pixels.mean()
            diff = pixels > avg
            return ''.join(['1' if b else '0' for b in diff])
    except Exception:
        return None

def find_duplicates(image_dir):
    """Finds exact and near-duplicate images in image_dir.
    Returns dictionary of duplicate groups.
    """
    if not os.path.exists(image_dir):
        print(f"Directory '{image_dir}' does not exist.")
        return {}, {}

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    print(f"Checking {len(image_files)} images for exact and perceptual duplicates...")

    exact_hashes = {}
    exact_duplicates = []

    perceptual_hashes = {}
    perceptual_duplicates = []

    for fname in image_files:
        fpath = os.path.join(image_dir, fname)
        
        # MD5 Exact Check
        md5_val = compute_md5(fpath)
        if md5_val in exact_hashes:
            exact_duplicates.append((fname, exact_hashes[md5_val]))
        else:
            exact_hashes[md5_val] = fname

        # Perceptual Hash Check
        ahash_val = compute_average_hash(fpath)
        if ahash_val is not None:
            if ahash_val in perceptual_hashes:
                perceptual_duplicates.append((fname, perceptual_hashes[ahash_val]))
            else:
                perceptual_hashes[ahash_val] = fname

    print(f"Duplicate Check Complete: Found {len(exact_duplicates)} exact duplicates, {len(perceptual_duplicates)} perceptual duplicates.")
    return exact_duplicates, perceptual_duplicates

if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "dataset/raw/images"
    find_duplicates(target)
