import cv2
import numpy as np
import os

# Adjustable parameters
class WatershedParams:
    def __init__(self):
        # sure_bg_dilate_ksize: Kernel size for dilating background regions
        # Larger values make background regions bigger, may misclassify particles as background
        self.sure_bg_dilate_ksize = (3, 3)

        # sure_fg_threshold: Distance transform threshold for determining foreground regions
        # Higher values: only points closer to center are considered foreground, may split one particle into multiple
        # Lower values: points farther from center may be considered foreground, may merge multiple particles
        self.sure_fg_threshold_ratio = 0.2  # Ratio of maximum distance

        # merge_aspect_ratio_threshold: Aspect ratio threshold for merging over-segmented regions
        # If a segmented small region has aspect ratio greater than this threshold, 
        # it will be considered as a thin strip and attempt to merge with adjacent regions
        self.merge_aspect_ratio_threshold = 4.0

        # roi_expansion_ratio: Ratio for expanding crop regions outward
        # Value of 0.1 means expanding 10% in each direction based on original ROI width and height
        self.roi_expansion_ratio = 0.1

def process_image(label_path, original_path, output_dir, params):
    """
    Perform watershed segmentation on label images and crop original and label images based on segmentation results.

    :param label_path: Path to label image (e.g., '1_label.png')
    :param original_path: Path to original image (e.g., '1.png')
    :param output_dir: Output directory
    :param params: WatershedParams instance
    """
    # 1. Load images
    label_img_orig = cv2.imread(label_path)
    original_img = cv2.imread(original_path)
    if label_img_orig is None or original_img is None:
        print(f"Error: Cannot read images from {label_path} or {original_path}")
        return

    # Create a copy for watershed algorithm
    label_img_for_watershed = label_img_orig.copy()

    gray = cv2.cvtColor(label_img_for_watershed, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. Determine foreground and background
    # Background
    sure_bg = cv2.dilate(thresh, np.ones(params.sure_bg_dilate_ksize, np.uint8), iterations=3)

    # Foreground
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    max_dist = dist_transform.max()
    ret, sure_fg = cv2.threshold(dist_transform, params.sure_fg_threshold_ratio * max_dist, 255, 0)

    # Unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 3. Create markers
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # 4. Apply watershed algorithm
    markers = cv2.watershed(label_img_for_watershed, markers)

    # 5. Post-processing: merge over-segmented regions
    unique_labels = np.unique(markers)
    particle_rois = {}
    img_h, img_w = gray.shape[:2]
    for label in unique_labels:
        if label <= 1:  # Ignore background and boundaries
            continue
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[markers == label] = 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            x, y, w, h = cv2.boundingRect(contours[0])
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
            is_on_border = x == 0 or y == 0 or (x + w) == gray.shape[1] or (y + h) == gray.shape[0]

            if aspect_ratio > params.merge_aspect_ratio_threshold and is_on_border:
                # This is a potentially over-segmented region, temporarily not processed, waiting for merge with other regions
                # For simplicity, we don't implement complex merge logic here, just segment first to observe effects
                # In actual applications, need to find adjacent thin strip regions for merging
                pass
            
            # Expand ROI region
            dx = int(w * params.roi_expansion_ratio)
            dy = int(h * params.roi_expansion_ratio)

            x_new = max(0, x - dx)
            y_new = max(0, y - dy)
            w_new = min(img_w - x_new, w + 2 * dx)
            h_new = min(img_h - y_new, h + 2 * dy)

            particle_rois[label] = (x_new, y_new, w_new, h_new)

    # 6. Crop, save and record positions
    img_output_path = os.path.join(output_dir, 'imgs')
    label_output_path = os.path.join(output_dir, 'labels')
    os.makedirs(img_output_path, exist_ok=True)
    os.makedirs(label_output_path, exist_ok=True)

    crop_info = []
    all_cropped_masks = np.zeros(gray.shape, dtype="uint8")

    for i, (label, (x, y, w, h)) in enumerate(particle_rois.items()):
        cropped_original = original_img[y:y+h, x:x+w]
        # Crop from original label image to avoid blue borders
        cropped_label = label_img_orig[y:y+h, x:x+w]

        cv2.imwrite(os.path.join(img_output_path, f'particle_{i}.png'), cropped_original)
        cv2.imwrite(os.path.join(label_output_path, f'particle_{i}_label.png'), cropped_label)
        
        crop_info.append({'id': i, 'roi': (x, y, w, h)})
        all_cropped_masks[y:y+h, x:x+w] = 255

    # 7. Save uncovered regions
    uncovered_mask = cv2.bitwise_not(all_cropped_masks)
    uncovered_mask[thresh == 0] = 0 # Only care about originally white regions
    contours, _ = cv2.findContours(uncovered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        if w > 5 and h > 5: # Ignore regions that are too small
            cropped_original = original_img[y:y+h, x:x+w]
            cropped_label = label_img_orig[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(img_output_path, f'uncovered_{i}.png'), cropped_original)
            cv2.imwrite(os.path.join(label_output_path, f'uncovered_{i}_label.png'), cropped_label)
            crop_info.append({'id': f'uncovered_{i}', 'roi': (x, y, w, h)})

    # 8. Save crop information
    with open(os.path.join(output_dir, 'crop_info.txt'), 'w') as f:
        for info in crop_info:
            f.write(f"{info['id']},{info['roi'][0]},{info['roi'][1]},{info['roi'][2]},{info['roi'][3]}\n")

    # 9. Reconstruct image for verification
    reconstructed_img = np.zeros_like(original_img)
    for info in crop_info:
        x, y, w, h = info['roi']
        if isinstance(info['id'], int):
            img_path = os.path.join(img_output_path, f'particle_{info["id"]}.png')
        else:
            img_path = os.path.join(img_output_path, f'{info["id"]}.png')
        
        if os.path.exists(img_path):
            patch = cv2.imread(img_path)
            if patch is not None and patch.shape[0] > 0 and patch.shape[1] > 0:
                 reconstructed_img[y:y+h, x:x+w] = patch

    # Color different particles
    colored_watershed = np.zeros_like(original_img)
    unique_labels = np.unique(markers)
    # Generate random colors
    colors = [np.random.randint(0, 255, 3).tolist() for i in range(len(unique_labels))]

    for i, label in enumerate(unique_labels):
        if label <= 1: # Don't color background and boundaries
            continue
        colored_watershed[markers == label] = colors[i]

    cv2.imwrite(os.path.join(output_dir, 'reconstructed.png'), reconstructed_img)
    cv2.imwrite(os.path.join(output_dir, 'watershed_result.png'), colored_watershed)

    print(f"Processing complete. Results saved in {output_dir}")

if __name__ == '__main__':
    # --- Configuration ---
    LABEL_IMAGE_PATH = ''  # Insert path to label image file (e.g., '1_label.png')
    ORIGINAL_IMAGE_PATH = ''  # Insert path to original image file (e.g., '1.png')
    OUTPUT_DIR = ''  # Insert path to output directory (e.g., 'data')
    params = WatershedParams()

    # --- Run ---
    process_image(LABEL_IMAGE_PATH, ORIGINAL_IMAGE_PATH, OUTPUT_DIR, params)