"""
AgriDataValue Interactive Annotation Correction Utility

YOLO classes:
    0 = Crop
    1 = Weed

Controls:
    LEFT CLICK       Select nearest bounding box
    RIGHT CLICK      Delete selected bounding box
    C               Change selected class Crop <-> Weed
    A                Start drawing a new bounding box
    ENTER            Finish new bounding box
    ESC              Cancel drawing
    U                Undo last change
    S                Save current annotation
    N                Save and go to next image
    P                Save and go to previous image
    Q                Quit

Safety:
    - Creates a backup of each label before first modification.
    - Writes normalized YOLO coordinates.
"""

import os
import shutil
import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = "dataset_agridatavalue"
EXTRACTED_DIR = os.path.join(DATASET_DIR, "extracted")
QA_DIR = os.path.join(DATASET_DIR, "qa", "initial_annotations")

BACKUP_DIR = os.path.join(DATASET_DIR, "qa", "annotation_backups")

SPLITS = ["train", "valid", "test"]

CLASS_NAMES = {
    0: "Crop",
    1: "Weed"
}

CLASS_COLORS = {
    0: (0, 255, 0),      # Green
    1: (0, 0, 255)       # Red
}


# ============================================================
# GLOBAL STATE
# ============================================================

boxes = []
selected_index = -1

drawing = False
draw_start = None
draw_end = None

history = []


# ============================================================
# FILE DISCOVERY
# ============================================================

def find_split_files(split):
    """
    Find matching image/label pairs recursively.
    """

    split_dir = os.path.join(EXTRACTED_DIR, split)

    image_map = {}
    label_map = {}

    for root, _, files in os.walk(split_dir):

        for filename in files:

            base, ext = os.path.splitext(filename)

            full_path = os.path.join(root, filename)

            if ext.lower() in [".jpg", ".jpeg", ".png"]:
                image_map[base] = full_path

            elif ext.lower() == ".txt" and filename.lower() != "classes.txt":
                label_map[base] = full_path

    common = sorted(set(image_map) & set(label_map))

    return [
        {
            "base": base,
            "image": image_map[base],
            "label": label_map[base],
            "split": split
        }
        for base in common
    ]


def build_dataset_index():
    """
    Build complete list of image/label pairs.
    """

    dataset = []

    for split in SPLITS:

        pairs = find_split_files(split)

        print(
            f"[INDEX] {split}: "
            f"{len(pairs)} image/label pairs"
        )

        dataset.extend(pairs)

    return dataset


# ============================================================
# YOLO LABEL HANDLING
# ============================================================

def load_labels(label_path):

    loaded = []

    if not os.path.exists(label_path):
        return loaded

    with open(label_path, "r", encoding="utf-8") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            try:

                cid = int(parts[0])

                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])

                if cid not in CLASS_NAMES:
                    continue

                loaded.append({
                    "class": cid,
                    "xc": xc,
                    "yc": yc,
                    "w": w,
                    "h": h
                })

            except ValueError:
                continue

    return loaded


def save_labels(label_path, annotation_boxes):

    with open(label_path, "w", encoding="utf-8") as f:

        for box in annotation_boxes:

            cid = int(box["class"])

            xc = np.clip(box["xc"], 0.0, 1.0)
            yc = np.clip(box["yc"], 0.0, 1.0)
            w = np.clip(box["w"], 0.0, 1.0)
            h = np.clip(box["h"], 0.0, 1.0)

            # Ignore invalid / zero-size boxes
            if w <= 0 or h <= 0:
                continue

            f.write(
                f"{cid} "
                f"{xc:.6f} "
                f"{yc:.6f} "
                f"{w:.6f} "
                f"{h:.6f}\n"
            )


def backup_label(label_path):

    os.makedirs(BACKUP_DIR, exist_ok=True)

    filename = os.path.basename(label_path)

    backup_path = os.path.join(
        BACKUP_DIR,
        filename
    )

    # Never overwrite an existing backup
    if not os.path.exists(backup_path):

        shutil.copy2(
            label_path,
            backup_path
        )

        print(
            f"[BACKUP] {backup_path}"
        )


# ============================================================
# COORDINATE CONVERSION
# ============================================================

def yolo_to_pixels(box, width, height):

    xc = box["xc"] * width
    yc = box["yc"] * height

    bw = box["w"] * width
    bh = box["h"] * height

    x1 = int(xc - bw / 2)
    y1 = int(yc - bh / 2)

    x2 = int(xc + bw / 2)
    y2 = int(yc + bh / 2)

    return x1, y1, x2, y2


def pixels_to_yolo(x1, y1, x2, y2, width, height):

    # Normalize ordering
    left = min(x1, x2)
    right = max(x1, x2)

    top = min(y1, y2)
    bottom = max(y1, y2)

    # Clamp to image
    left = max(0, min(left, width - 1))
    right = max(0, min(right, width - 1))

    top = max(0, min(top, height - 1))
    bottom = max(0, min(bottom, height - 1))

    bw = right - left
    bh = bottom - top

    if bw < 2 or bh < 2:
        return None

    xc = (left + right) / 2
    yc = (top + bottom) / 2

    return {
        "class": 1,  # New boxes default to Weed
        "xc": xc / width,
        "yc": yc / height,
        "w": bw / width,
        "h": bh / height
    }


# ============================================================
# HISTORY / UNDO
# ============================================================

def push_history():

    global history

    snapshot = [
        dict(box)
        for box in boxes
    ]

    history.append(snapshot)

    # Prevent unlimited memory growth
    if len(history) > 50:
        history.pop(0)


def undo():

    global boxes

    if not history:

        print("[UNDO] Nothing to undo.")
        return

    boxes = history.pop()

    print("[UNDO] Restored previous annotation state.")


# ============================================================
# DRAWING
# ============================================================

def render_image(image):

    global selected_index
    global drawing, draw_start, draw_end

    display = image.copy()

    h, w = display.shape[:2]

    # Existing boxes
    for i, box in enumerate(boxes):

        x1, y1, x2, y2 = yolo_to_pixels(
            box,
            w,
            h
        )

        cid = box["class"]

        color = CLASS_COLORS.get(
            cid,
            (255, 255, 255)
        )

        thickness = 4 if i == selected_index else 2

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            color,
            thickness
        )

        label = f"{i}: {CLASS_NAMES[cid]}"

        cv2.putText(
            display,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA
        )

    # Drawing preview
    if drawing and draw_start and draw_end:

        cv2.rectangle(
            display,
            draw_start,
            draw_end,
            (255, 255, 0),
            2
        )

    # Instructions
    instructions = [
        "LEFT: select | RIGHT: delete",
        "C: change class | A: add box",
        "ENTER: finish | ESC: cancel",
        "U: undo | S: save | N: next | P: previous | Q: quit"
    ]

    y = 25

    for text in instructions:

        cv2.putText(
            display,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        y += 25

    return display


# ============================================================
# MOUSE
# ============================================================

def mouse_callback(event, x, y, flags, param):

    global selected_index
    global drawing
    global draw_start
    global draw_end
    global boxes

    image = param
    h, w = image.shape[:2]

    # --------------------------------------------------------
    # LEFT CLICK
    # --------------------------------------------------------

    if event == cv2.EVENT_LBUTTONDOWN:

        if drawing:

            draw_start = (x, y)
            draw_end = (x, y)

        else:

            selected_index = -1

            best_distance = float("inf")

            for i, box in enumerate(boxes):

                x1, y1, x2, y2 = yolo_to_pixels(
                    box,
                    w,
                    h
                )

                # If click inside box, select immediately
                if x1 <= x <= x2 and y1 <= y <= y2:

                    selected_index = i
                    break

                # Otherwise calculate center distance
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                distance = (
                    (x - cx) ** 2 +
                    (y - cy) ** 2
                )

                if distance < best_distance:

                    best_distance = distance

                    selected_index = i

    # --------------------------------------------------------
    # MOUSE MOVE
    # --------------------------------------------------------

    elif event == cv2.EVENT_MOUSEMOVE:

        if drawing and draw_start:

            draw_end = (x, y)

    # --------------------------------------------------------
    # RIGHT CLICK = DELETE
    # --------------------------------------------------------

    elif event == cv2.EVENT_RBUTTONDOWN:

        if selected_index >= 0:

            push_history()

            removed = boxes.pop(selected_index)

            print(
                f"[DELETE] Box {selected_index}: "
                f"{CLASS_NAMES[removed['class']]}"
            )

            selected_index = -1


# ============================================================
# CORRECTION LOOP
# ============================================================

def edit_annotation(item, index, total):

    global boxes
    global selected_index
    global drawing
    global draw_start
    global draw_end
    global history

    image = cv2.imread(item["image"])

    if image is None:

        print(
            f"[ERROR] Could not load image: "
            f"{item['image']}"
        )

        return "next"

    boxes = load_labels(item["label"])

    selected_index = -1

    drawing = False
    draw_start = None
    draw_end = None

    history = []

    window_name = (
        f"Annotation Correction "
        f"[{index + 1}/{total}]"
    )

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    cv2.setMouseCallback(
        window_name,
        mouse_callback,
        image
    )

    print("\n" + "=" * 70)
    print(
        f"[{index + 1}/{total}] "
        f"{item['split'].upper()}"
    )
    print(f"IMAGE: {item['image']}")
    print(f"LABEL: {item['label']}")
    print(f"BOXES: {len(boxes)}")

    while True:

        display = render_image(image)

        cv2.imshow(
            window_name,
            display
        )

        key = cv2.waitKey(30) & 0xFF

        # ----------------------------------------------------
        # QUIT
        # ----------------------------------------------------

        if key == ord("q"):

            cv2.destroyWindow(window_name)

            return "quit"

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        elif key == ord("s"):

            backup_label(item["label"])

            save_labels(
                item["label"],
                boxes
            )

            print(
                f"[SAVED] {item['label']}"
            )

        # ----------------------------------------------------
        # NEXT
        # ----------------------------------------------------

        elif key == ord("n"):

            backup_label(item["label"])

            save_labels(
                item["label"],
                boxes
            )

            print(
                f"[SAVED] Moving to next image."
            )

            cv2.destroyWindow(window_name)

            return "next"

        # ----------------------------------------------------
        # PREVIOUS
        # ----------------------------------------------------

        elif key == ord("p"):

            backup_label(item["label"])

            save_labels(
                item["label"],
                boxes
            )

            cv2.destroyWindow(window_name)

            return "previous"

        # ----------------------------------------------------
        # CHANGE CLASS
        # ----------------------------------------------------

        elif key == ord("c"):

            if selected_index >= 0:

                push_history()

                old_class = boxes[selected_index]["class"]

                new_class = (
                    1
                    if old_class == 0
                    else 0
                )

                boxes[selected_index]["class"] = new_class

                print(
                    f"[CLASS] Box {selected_index}: "
                    f"{CLASS_NAMES[old_class]} -> "
                    f"{CLASS_NAMES[new_class]}"
                )

        # ----------------------------------------------------
        # UNDO
        # ----------------------------------------------------

        elif key == ord("u"):

            undo()

        # ----------------------------------------------------
        # ADD BOX
        # ----------------------------------------------------

        elif key == ord("a"):

            drawing = True

            draw_start = None
            draw_end = None

            print(
                "[ADD] Click and drag to create a box."
            )

        # ----------------------------------------------------
        # FINISH DRAWING
        # ----------------------------------------------------

        elif key == 13:  # ENTER

            if drawing and draw_start and draw_end:

                x1, y1 = draw_start
                x2, y2 = draw_end

                new_box = pixels_to_yolo(
                    x1,
                    y1,
                    x2,
                    y2,
                    image.shape[1],
                    image.shape[0]
                )

                if new_box:

                    push_history()

                    boxes.append(new_box)

                    print(
                        "[ADD] New Weed box added."
                    )

                else:

                    print(
                        "[ADD] Box too small."
                    )

                drawing = False
                draw_start = None
                draw_end = None

        # ----------------------------------------------------
        # CANCEL DRAWING
        # ----------------------------------------------------

        elif key == 27:  # ESC

            drawing = False
            draw_start = None
            draw_end = None


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AGRIDATAVALUE ANNOTATION CORRECTION UTILITY")
    print("=" * 70)

    dataset = build_dataset_index()

    if not dataset:

        print(
            "[ERROR] No image/label pairs found."
        )

        return

    print()
    print(
        f"[TOTAL] {len(dataset)} image/label pairs available."
    )

    print()
    print("IMPORTANT:")
    print("0 = Crop")
    print("1 = Weed")
    print()
    print("The first time a label is changed,")
    print("a backup is created.")
    print()

    index = 0

    while 0 <= index < len(dataset):

        action = edit_annotation(
            dataset[index],
            index,
            len(dataset)
        )

        if action == "quit":
            break

        elif action == "next":
            index += 1

        elif action == "previous":
            index = max(0, index - 1)

    cv2.destroyAllWindows()

    print()
    print("=" * 70)
    print("ANNOTATION CORRECTION SESSION FINISHED")
    print("=" * 70)
    print(
        f"Backups are stored in: {BACKUP_DIR}"
    )


if __name__ == "__main__":
    main()