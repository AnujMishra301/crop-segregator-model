"""
Sample Downward Agricultural Dataset Generator for Testing Pipeline
Generates synthetic top-down agricultural field images and annotations (YOLO format)
to test and validate dataset pipelines, leak prevention, and report generation.
"""

import os
import random
import math
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import numpy as np

# Canonical class mapping
CLASS_MAP = {
    0: "weed",
    1: "crop",
    2: "grass_lawn",
    3: "other"
}

def create_soil_background(width=640, height=640, soil_type="loam"):
    """Generates realistic agricultural soil background textures."""
    if soil_type == "loam":
        base_color = (105, 75, 55) # Dark brown loam
    elif soil_type == "clay":
        base_color = (140, 85, 50) # Reddish clay
    elif soil_type == "sandy":
        base_color = (160, 135, 95) # Sandy light soil
    else:
        base_color = (80, 65, 50)

    # Base array with color variation
    img_np = np.zeros((height, width, 3), dtype=np.uint8)
    for c in range(3):
        noise = np.random.normal(0, 15, (height, width))
        channel = base_color[c] + noise
        img_np[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)

    img = Image.fromarray(img_np)
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    return img

def draw_plant(draw, x_center, y_center, radius, plant_type):
    """Draws leaf shapes representing crops, weeds, or grass."""
    if plant_type == "crop":
        # Green symmetric crop plant (e.g. young sugar beet / soybean)
        num_leaves = random.randint(4, 6)
        color = (random.randint(30, 60), random.randint(140, 200), random.randint(30, 60))
        for i in range(num_leaves):
            angle = (2 * math.pi / num_leaves) * i + random.uniform(-0.2, 0.2)
            lx = x_center + math.cos(angle) * radius * 0.8
            ly = y_center + math.sin(angle) * radius * 0.8
            bbox = [x_center - radius*0.6, y_center - radius*0.6, x_center + radius*0.6, y_center + radius*0.6]
            draw.ellipse(bbox, fill=color)

    elif plant_type == "weed":
        # Irregular dark green / yellowish broadleaf weed
        num_leaves = random.randint(3, 7)
        color = (random.randint(40, 90), random.randint(110, 170), random.randint(20, 50))
        for i in range(num_leaves):
            rx = random.uniform(radius * 0.4, radius * 1.1)
            ry = random.uniform(radius * 0.4, radius * 1.1)
            offset_x = random.uniform(-radius*0.4, radius*0.4)
            offset_y = random.uniform(-radius*0.4, radius*0.4)
            bbox = [x_center + offset_x - rx, y_center + offset_y - ry, 
                    x_center + offset_x + rx, y_center + offset_y + ry]
            draw.ellipse(bbox, fill=color)

    elif plant_type == "grass_lawn":
        # Thin blade-like grass patches
        color = (random.randint(50, 100), random.randint(150, 210), random.randint(30, 70))
        for _ in range(12):
            dx = random.uniform(-radius, radius)
            dy = random.uniform(-radius, radius)
            x2 = x_center + dx + random.uniform(-5, 5)
            y2 = y_center + dy - random.uniform(10, radius*1.2)
            draw.line([(x_center + dx, y_center + dy), (x2, y2)], fill=color, width=random.randint(2, 4))

    else:
        # Other (e.g. stones, debris, plastic)
        color = (random.randint(120, 160), random.randint(110, 150), random.randint(100, 140))
        bbox = [x_center - radius*0.5, y_center - radius*0.5, x_center + radius*0.5, y_center + radius*0.5]
        draw.rectangle(bbox, fill=color)

def generate_field_dataset(output_dir="dataset/raw", num_fields=5, images_per_field=20):
    """Generates synthetic dataset organized by field sequences to model non-leaking dataset splits."""
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

    img_count = 0
    annotation_count = 0
    soil_types = ["loam", "clay", "sandy"]

    print(f"Generating synthetic agricultural field data across {num_fields} fields...")

    for field_id in range(1, num_fields + 1):
        soil = soil_types[field_id % len(soil_types)]
        
        for frame_idx in range(1, images_per_field + 1):
            img_count += 1
            filename = f"field{field_id:02d}_frame{frame_idx:03d}"
            img_path = os.path.join(output_dir, "images", f"{filename}.jpg")
            label_path = os.path.join(output_dir, "labels", f"{filename}.txt")

            img = create_soil_background(640, 640, soil_type=soil)
            draw = ImageDraw.Draw(img)

            boxes = [] # List of (class_id, xc, yc, w, h)

            # Generate crops in rows (simulating agricultural crop rows)
            crop_row_x = [160, 320, 480]
            for rx in crop_row_x:
                for ry in range(80, 600, 120):
                    if random.random() < 0.85: # Crop present
                        radius = random.randint(25, 45)
                        cx = rx + random.randint(-15, 15)
                        cy = ry + random.randint(-15, 15)
                        draw_plant(draw, cx, cy, radius, "crop")
                        
                        # YOLO box: x_center, y_center, width, height normalized
                        xc = cx / 640.0
                        yc = cy / 640.0
                        w = (radius * 2.1) / 640.0
                        h = (radius * 2.1) / 640.0
                        boxes.append((1, xc, yc, w, h))

            # Randomly scatter weeds (random locations)
            num_weeds = random.randint(2, 6)
            for _ in range(num_weeds):
                cx = random.randint(40, 600)
                cy = random.randint(40, 600)
                radius = random.randint(12, 35) # Weeds vary from small to medium
                
                # Determine weed vs grass vs other
                rand_val = random.random()
                if rand_val < 0.70:
                    cls_id = 0 # weed
                    draw_plant(draw, cx, cy, radius, "weed")
                elif rand_val < 0.90:
                    cls_id = 2 # grass_lawn
                    draw_plant(draw, cx, cy, radius, "grass_lawn")
                else:
                    cls_id = 3 # other
                    draw_plant(draw, cx, cy, radius, "other")

                xc = cx / 640.0
                yc = cy / 640.0
                w = (radius * 2.2) / 640.0
                h = (radius * 2.2) / 640.0
                boxes.append((cls_id, xc, yc, w, h))

            # Apply environmental lighting & shadows
            if random.random() < 0.3:
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(random.uniform(0.7, 1.2))

            # Save image
            img.save(img_path, quality=92)

            # Save YOLO label file
            with open(label_path, "w") as f:
                for box in boxes:
                    cls_id, xc, yc, w, h = box
                    # Clamp bounding box coordinates to [0.0, 1.0]
                    xc = max(0.001, min(0.999, xc))
                    yc = max(0.001, min(0.999, yc))
                    w = max(0.001, min(0.999, w))
                    h = max(0.001, min(0.999, h))
                    f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                    annotation_count += 1

    print(f"Generated {img_count} images with {annotation_count} annotations in '{output_dir}'.")

if __name__ == "__main__":
    generate_field_dataset()
