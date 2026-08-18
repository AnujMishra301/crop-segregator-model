"""
Corrupted Image Detector Module
Inspects all images in a specified dataset directory for file corruption,
invalid headers, zero file sizes, or unreadable byte streams.
"""

import os
from PIL import Image

def check_corrupt_images(image_dir):
    """Scans image_dir for corrupt, unreadable, or empty images.
    Returns: (valid_images_list, corrupt_images_list)
    """
    valid_images = []
    corrupt_images = []

    if not os.path.exists(image_dir):
        print(f"Directory '{image_dir}' does not exist.")
        return valid_images, corrupt_images

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]
    print(f"Scanning {len(image_files)} image files in '{image_dir}' for corruption...")

    for fname in image_files:
        fpath = os.path.join(image_dir, fname)
        
        # Check zero-byte file
        if os.path.getsize(fpath) == 0:
            corrupt_images.append((fname, "Zero byte file size"))
            continue
            
        try:
            with Image.open(fpath) as img:
                img.verify() # Verify image headers and structure
            
            # Reopen to check pixel payload integrity
            with Image.open(fpath) as img:
                img.load()
                w, h = img.size
                if w <= 0 or h <= 0:
                    corrupt_images.append((fname, f"Invalid dimensions: {w}x{h}"))
                else:
                    valid_images.append(fpath)
                    
        except Exception as e:
            corrupt_images.append((fname, str(e)))

    print(f"Corrupt Scan Complete: {len(valid_images)} valid, {len(corrupt_images)} corrupt.")
    return valid_images, corrupt_images

if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "dataset/raw/images"
    check_corrupt_images(target)
