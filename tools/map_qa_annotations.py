import os
import random

EXTRACTED_DIR = "dataset_agridatavalue/extracted"
QA_DIR = "dataset_agridatavalue/qa/initial_annotations"

def find_split_files(split):
    split_dir = os.path.join(EXTRACTED_DIR, split)

    img_map = {}
    lbl_map = {}

    for root, _, files in os.walk(split_dir):
        for f in files:
            base, ext = os.path.splitext(f)
            path = os.path.join(root, f)

            if ext.lower() in [".jpg", ".jpeg", ".png"]:
                img_map[base] = path

            elif ext.lower() == ".txt" and f.lower() != "classes.txt":
                lbl_map[base] = path

    return img_map, lbl_map


def main():
    random.seed(42)

    print("=" * 80)
    print("QA VISUALIZATION → ORIGINAL IMAGE → LABEL TXT MAPPING")
    print("=" * 80)

    total = 0

    for split in ["train", "valid", "test"]:

        img_map, lbl_map = find_split_files(split)

        common_bases = sorted(
            list(set(img_map.keys()) & set(lbl_map.keys()))
        )

        sample_bases = random.sample(
            common_bases,
            min(10, len(common_bases))
        )

        print(f"\n{'=' * 80}")
        print(f"{split.upper()} SPLIT")
        print(f"{'=' * 80}")

        for idx, base in enumerate(sample_bases):

            qa_name = f"{split}_sample_{idx+1:02d}_{base[:12]}.jpg"

            image_path = os.path.abspath(img_map[base])
            label_path = os.path.abspath(lbl_map[base])

            print(f"\n{total + 1:02d}. QA IMAGE:")
            print(f"    {qa_name}")

            print("    ORIGINAL IMAGE:")
            print(f"    {image_path}")

            print("    LABEL TXT:")
            print(f"    {label_path}")

            total += 1

    print(f"\n{'=' * 80}")
    print(f"TOTAL QA SAMPLE MAPPINGS: {total}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()