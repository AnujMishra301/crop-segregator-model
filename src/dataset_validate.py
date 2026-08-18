"""
Annotation Validator Module
Inspects YOLO label files (.txt) for formatting compliance, valid class IDs [0..3],
out-of-bounds bounding boxes, zero-area boxes, and missing corresponding image files.
"""

import os
from PIL import Image

VALID_CLASS_IDS = {0, 1, 2, 3}

def validate_annotations(image_dir, label_dir):
    """Validates images and corresponding YOLO label files.
    Returns: (valid_pairs, invalid_annotations, orphan_images, orphan_labels)
    """
    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        print("Image or Label directory does not exist.")
        return [], [], [], []

    img_files = {os.path.splitext(f)[0]: f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))}
    lbl_files = {os.path.splitext(f)[0]: f for f in os.listdir(label_dir) if f.lower().endswith('.txt')}

    all_bases = set(img_files.keys()).union(set(lbl_files.keys()))
    
    valid_pairs = []
    invalid_annotations = []
    orphan_images = []
    orphan_labels = []

    for base in sorted(all_bases):
        if base not in img_files:
            orphan_labels.append(lbl_files[base])
            continue
        if base not in lbl_files:
            orphan_images.append(img_files[base])
            continue

        img_path = os.path.join(image_dir, img_files[base])
        lbl_path = os.path.join(label_dir, lbl_files[base])

        # Inspect label contents
        is_valid = True
        error_msg = ""
        box_count = 0

        with open(lbl_path, 'r') as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                is_valid = False
                error_msg = f"Line {line_num}: Expected 5 elements (cls xc yc w h), got {len(parts)}"
                break
            try:
                cls_id = int(parts[0])
                xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                is_valid = False
                error_msg = f"Line {line_num}: Invalid float numbers in '{line}'"
                break

            if cls_id not in VALID_CLASS_IDS:
                is_valid = False
                error_msg = f"Line {line_num}: Invalid class_id {cls_id} (must be 0, 1, 2, or 3)"
                break

            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
                is_valid = False
                error_msg = f"Line {line_num}: Center coords out of bounds ({xc:.3f}, {yc:.3f})"
                break

            if w <= 0.0 or h <= 0.0 or w > 1.0 or h > 1.0:
                is_valid = False
                error_msg = f"Line {line_num}: Invalid box dimensions w={w:.3f}, h={h:.3f}"
                break

            box_count += 1

        if is_valid:
            valid_pairs.append((img_files[base], lbl_files[base], box_count))
        else:
            invalid_annotations.append((lbl_files[base], error_msg))

    print(f"Annotation Validation Complete:")
    print(f"  Valid Image-Label Pairs: {len(valid_pairs)}")
    print(f"  Invalid Label Files:     {len(invalid_annotations)}")
    print(f"  Orphan Images (No label):{len(orphan_images)}")
    print(f"  Orphan Labels (No image):{len(orphan_labels)}")

    return valid_pairs, invalid_annotations, orphan_images, orphan_labels

if __name__ == "__main__":
    import sys
    img_d = sys.argv[1] if len(sys.argv) > 1 else "dataset/raw/images"
    lbl_d = sys.argv[2] if len(sys.argv) > 2 else "dataset/raw/labels"
    validate_annotations(img_d, lbl_d)
