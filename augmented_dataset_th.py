import cv2
import numpy as np
import random
import os
from glob import glob
from PIL import Image
import yaml

FABRIC_FOLDER = "/home/seif/Downloads/AugmentByDefect/backgrounds/"
DEFECT_FOLDER = "/home/seif/Downloads/AugmentByDefect/38_Defects-GrayScale/"
OUTPUT_FOLDER = "/home/seif/Downloads/AugmentByDefect/images/"
LABELS_FOLDER = "/home/seif/Downloads/AugmentByDefect/labels/"
TRAIN_IMAGES_FOLDER = "/home/seif/Downloads/AugmentByDefect/train/images/"
VAL_IMAGES_FOLDER = "/home/seif/Downloads/AugmentByDefect/val/images/"
TEST_IMAGES_FOLDER = "/home/seif/Downloads/AugmentByDefect/test/images/"
TRAIN_LABELS_FOLDER = "/home/seif/Downloads/AugmentByDefect/train/labels/"
VAL_LABELS_FOLDER = "/home/seif/Downloads/AugmentByDefect/val/labels/"
TEST_LABELS_FOLDER = "/home/seif/Downloads/AugmentByDefect/test/labels/"

# Create directories if they don't exist
os.makedirs(TRAIN_IMAGES_FOLDER, exist_ok=True)
os.makedirs(VAL_IMAGES_FOLDER, exist_ok=True)
os.makedirs(TEST_IMAGES_FOLDER, exist_ok=True)
os.makedirs(TRAIN_LABELS_FOLDER, exist_ok=True)
os.makedirs(VAL_LABELS_FOLDER, exist_ok=True)
os.makedirs(TEST_LABELS_FOLDER, exist_ok=True)

# List of defect classes
CLASSES = ["defect"]

# Number of augmentations to generate
NUM_AUGMENTATIONS = 1000

def load_image(image_path):
    """Loads an image, handling different formats."""
    image = Image.open(image_path).convert("RGBA")  # Ensure RGBA for defects
    return np.array(image)

def load_fabric(image_path):
    """Loads fabric images (JPG, PNG, WebP) and converts them to RGB."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

def random_perspective_transform(image):
    """Applies a random skew to simulate perspective distortions."""
    h, w = image.shape[:2]
    src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

    skew_amount = random.uniform(0.05, 0.2) * min(w, h)
    dst_pts = np.float32([
        [random.uniform(0, skew_amount), random.uniform(0, skew_amount)],
        [w - random.uniform(0, skew_amount), random.uniform(0, skew_amount)],
        [random.uniform(0, skew_amount), h - random.uniform(0, skew_amount)],
        [w - random.uniform(0, skew_amount), h - random.uniform(0, skew_amount)]
    ])

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

def blend_images(fabric, defect, position):
    """Blends the defect onto the fabric with Gaussian feathering."""
    fx, fy = position
    h, w = defect.shape[:2]

    # Convert fabric to RGB before blending
    fabric_rgb = fabric.copy()

    # Extract ROI
    roi = fabric_rgb[fy:fy+h, fx:fx+w]

    # Convert to float for blending
    roi_float = roi.astype(np.float32)
    defect_float = defect[:, :, :3].astype(np.float32)

    # Create mask for transparency
    mask = defect[:, :, 3] / 255.0
    mask = cv2.GaussianBlur(mask, (11, 11), 5)

    # Blend
    for c in range(3):
        roi_float[:, :, c] = roi_float[:, :, c] * (1 - mask) + defect_float[:, :, c] * mask

    fabric_rgb[fy:fy+h, fx:fx+w] = roi_float.astype(np.uint8)

    return fabric_rgb

def save_yolo_annotation(filename, pos_x, pos_y, defect_width, defect_height, fabric_width, fabric_height):
    """Saves bounding box in YOLO format."""
    x_center = (pos_x + defect_width / 2) / fabric_width
    y_center = (pos_y + defect_height / 2) / fabric_height
    width = defect_width / fabric_width
    height = defect_height / fabric_height

    with open(filename, "w") as f:
        f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def augment_image(fabric_path, defect_path, output_image_path, output_label_path):
    """Applies augmentation and saves YOLO annotation."""
    fabric = load_fabric(fabric_path)
    defect = load_image(defect_path)

    # Resize defect
    scale_factor = random.uniform(0.4, 0.8)
    defect = np.array(Image.fromarray(defect).resize((int(defect.shape[1] * scale_factor), int(defect.shape[0] * scale_factor)), Image.LANCZOS))

    # Rotate defect
    angle = random.randint(0, 360)
    defect = np.array(Image.fromarray(defect).rotate(angle, expand=True))

    # Apply skew
    defect = random_perspective_transform(defect)

    # Ensure it fits
    if defect.shape[0] > fabric.shape[0] or defect.shape[1] > fabric.shape[1]:
        return

    # Random position
    max_x = fabric.shape[1] - defect.shape[1]
    max_y = fabric.shape[0] - defect.shape[0]
    pos_x = random.randint(0, max_x)
    pos_y = random.randint(0, max_y)

    # Blend
    blended_fabric = blend_images(fabric, defect, (pos_x, pos_y))

    # Save image
    cv2.imwrite(output_image_path, cv2.cvtColor(blended_fabric, cv2.COLOR_RGB2BGR))

    # Save YOLO label
    save_yolo_annotation(output_label_path, pos_x, pos_y, defect.shape[1], defect.shape[0], fabric.shape[1], fabric.shape[0])

def create_yaml():
    """Generates a YOLOv5 dataset YAML file."""
    data = {
        'path': './AugmentByDefect',  # Path to dataset folder
        'train': 'train/images',  # Relative path to train images directory
        'val': 'val/images',  # Relative path to val images directory
        'test': 'test/images',  # Relative path to test images directory (optional)
        'names': {0: 'defect'}  # Class names
    }
    with open('dataset.yaml', 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)

def main():
    fabric_paths = glob(os.path.join(FABRIC_FOLDER, "*.[jp][npw][gpb]*"))  # Match JPG, PNG, WebP
    defect_paths = glob(os.path.join(DEFECT_FOLDER, "*.[jp][npw][gpb]*"))  # Match JPG, PNG, WebP

    for i in range(NUM_AUGMENTATIONS):
        fabric_path = random.choice(fabric_paths)
        defect_path = random.choice(defect_paths)

        # Split data into train, val, test sets (80-10-10 split)
        if i < NUM_AUGMENTATIONS * 0.8:
            output_image = os.path.join(TRAIN_IMAGES_FOLDER, f"aug_{i}.jpg")
            output_label = os.path.join(TRAIN_LABELS_FOLDER, f"aug_{i}.txt")
        elif i < NUM_AUGMENTATIONS * 0.9:
            output_image = os.path.join(VAL_IMAGES_FOLDER, f"aug_{i}.jpg")
            output_label = os.path.join(VAL_LABELS_FOLDER, f"aug_{i}.txt")
        else:
            output_image = os.path.join(TEST_IMAGES_FOLDER, f"aug_{i}.jpg")
            output_label = os.path.join(TEST_LABELS_FOLDER, f"aug_{i}.txt")

        augment_image(fabric_path, defect_path, output_image, output_label)
        print(f"Saved: {output_image}, {output_label}")

    # Create the YAML file
    create_yaml()

if __name__ == "__main__":
    main()