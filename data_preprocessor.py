import os
import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def resize_with_padding(img, target_size):
    """
    Resize image to target size while maintaining aspect ratio with padding.

    :param img: PIL Image object
    :param target_size: (width, height) tuple
    :return: Resized and padded PIL Image object
    """
    # Original image dimensions
    original_width, original_height = img.size

    # Target dimensions
    target_width, target_height = target_size

    # Calculate scaling ratio to ensure entire image fits within target size
    ratio = min(target_width / original_width, target_height / original_height)

    # New dimensions
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)

    # Resize image
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Create black background image
    new_img = Image.new("RGB", target_size, (0, 0, 0))

    # Calculate paste position to center the image
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2

    # Paste resized image onto background
    new_img.paste(img, (paste_x, paste_y))

    return new_img

def preprocess_data(input_dir, output_dir, target_size=(128, 128), val_size=0.1):
    """
    Preprocess all particle images: standardize size, split dataset and save.
    """
    # Create output directories
    img_output_path = os.path.join(output_dir, 'imgs')
    label_output_path = os.path.join(output_dir, 'labels')
    os.makedirs(img_output_path, exist_ok=True)
    os.makedirs(label_output_path, exist_ok=True)

    # Get all image paths
    img_paths = sorted(glob.glob(os.path.join(input_dir, 'imgs', '*.png')))
    label_paths = sorted(glob.glob(os.path.join(input_dir, 'labels_1', '*.png')))

    print(f"Found {len(img_paths)} images to preprocess.")

    processed_img_paths = []
    for img_path, label_path in tqdm(zip(img_paths, label_paths), total=len(img_paths), desc="Preprocessing Images"):
        img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('RGB')

        # Standardize size
        img_resized = resize_with_padding(img, target_size)
        label_resized = resize_with_padding(label, target_size)

        # Save processed images
        base_filename = os.path.basename(img_path)
        new_img_path = os.path.join(img_output_path, base_filename)
        new_label_path = os.path.join(label_output_path, os.path.basename(label_path))

        img_resized.save(new_img_path)
        label_resized.save(new_label_path)
        processed_img_paths.append(os.path.relpath(new_img_path, output_dir))

    # Split dataset - K-fold cross-validation doesn't need test set
    # Only split into training and validation sets
    train_paths, val_paths = train_test_split(processed_img_paths, test_size=val_size, random_state=42)

    # Save file lists
    def save_paths_to_file(paths, file_path, base_dir):
        abs_base_dir = os.path.abspath(base_dir)
        with open(file_path, 'w') as f:
            for rel_img_path in paths:
                # rel_img_path is like 'imgs\filename.png' or 'imgs/filename.png'
                abs_img_path = os.path.join(abs_base_dir, rel_img_path)

                # Construct corresponding absolute label path
                filename = os.path.basename(rel_img_path)
                label_filename = filename.replace('.png', '_label.png').replace('.jpg', '_label.jpg').replace('.jpeg', '_label.jpeg')
                abs_label_path = os.path.join(abs_base_dir, 'labels', label_filename)

                # Write both paths separated by a semicolon, using forward slashes for consistency
                img_path_normalized = abs_img_path.replace('\\', '/')
                label_path_normalized = abs_label_path.replace('\\', '/')
                f.write(f"{img_path_normalized};{label_path_normalized}\n")

    save_paths_to_file(train_paths, os.path.join(output_dir, 'train.txt'), output_dir)
    save_paths_to_file(val_paths, os.path.join(output_dir, 'val.txt'), output_dir)

    print("\nPreprocessing complete!")
    print(f"  - Total images: {len(processed_img_paths)}")
    print(f"  - Training set: {len(train_paths)} images")
    print(f"  - Validation set: {len(val_paths)} images")
    print(f"Processed data saved in: {output_dir}")
    print(f"Dataset lists saved to: {os.path.join(output_dir, '[train/val].txt')}")

if __name__ == '__main__':
    INPUT_DATA_DIR = ''  # Insert path to input data directory (e.g., 'data')
    OUTPUT_DATA_DIR = ''  # Insert path to output data directory (e.g., 'data_processed')
    TARGET_IMAGE_SIZE = (128, 128)  # Can be adjusted according to model requirements

    preprocess_data(INPUT_DATA_DIR, OUTPUT_DATA_DIR, target_size=TARGET_IMAGE_SIZE)