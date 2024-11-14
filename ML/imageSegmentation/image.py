from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import os
import random
from skimage.draw import polygon

def visualize_car_damage(dataDir, dataType, img_dir, category='damage', image_file=None):
    # Set annotation file paths
    annFile = os.path.join(dataDir, f"{dataType}.json")

    # Initialize COCO API for instance annotations
    coco = COCO(annFile)

    # Get all images containing the specified category
    catIds = coco.getCatIds(catNms=[category])
    imgIds = coco.getImgIds(catIds=catIds)

    # Use the specified image if provided, otherwise select a random image
    if image_file:
        # Extract the image ID from the filename
        image_name = os.path.basename(image_file)
        image_info = next((img for img in coco.loadImgs(imgIds) if img['file_name'] == image_name), None)
    elif imgIds:
        random_img_id = random.choice(imgIds)
        image_info = coco.loadImgs(random_img_id)[0]
    else:
        print(f"No images found for the '{category}' category.")
        return

    if image_info is None:
        print("Image file does not match any entry in the annotations.")
        return

    # Get annotations for the selected image
    annIds = coco.getAnnIds(imgIds=image_info['id'], iscrowd=None)
    anns = coco.loadAnns(annIds)

    # Load the selected image with error handling
    try:
        I = io.imread(image_file or os.path.join(img_dir, image_info['file_name']))
    except FileNotFoundError:
        print(f"Image file '{image_info['file_name']}' not found in directory '{img_dir}'.")
        return
    except Exception as e:
        print(f"Error loading image '{image_info['file_name']}': {e}")
        return

    # Create an empty mask for the segmented image
    full_car_mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)

    # Draw each annotation's polygon on the full car mask
    for ann in anns:
        if 'segmentation' in ann:
            seg = ann['segmentation']
            for polygon_points in seg:
                poly_x = polygon_points[0::2]
                poly_y = polygon_points[1::2]
                rr, cc = polygon(poly_y, poly_x, full_car_mask.shape)
                full_car_mask[rr, cc] = 1

    # Create a red mask for the damaged areas
    red_mask = np.zeros_like(I)
    for ann in anns:
        if 'segmentation' in ann:
            seg = ann['segmentation']
            for polygon_points in seg:
                poly_x = polygon_points[0::2]
                poly_y = polygon_points[1::2]
                rr, cc = polygon(poly_y, poly_x, red_mask.shape[:2])
                red_mask[rr, cc] = [255, 0, 0]

    # Create the full segmented image with transparency
    full_segmented_image = np.zeros_like(I)
    full_segmented_image[full_car_mask == 1] = I[full_car_mask == 1]
    full_segmented_image = np.where(red_mask != 0, red_mask, full_segmented_image)

    # Combine images into a single output for display
    overlay_image = I.copy()
    alpha = 0.5  # Transparency factor
    red_mask_rgb = red_mask / 255  # Normalize the red mask
    overlay_image = (1 - alpha) * overlay_image + alpha * red_mask_rgb * 255  # Blend the images

    # Draw bounding boxes for damage annotations on the overlayed image
    for ann in anns:
        if 'bbox' in ann:
            bbox = ann['bbox']
            overlay_image = draw_bounding_box(overlay_image, bbox, color=(255, 0, 0), thickness=2)

    # Create a figure to display the images
    plt.figure(figsize=(20, 5))

    # Display the original image
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.imshow(I)
    plt.title('Original Image')

    # Display the overlayed image with bounding boxes and red masks
    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.imshow(overlay_image.astype(np.uint8))
    plt.title('Overlayed Image with Damage Areas')

    # Display the segmented image with damaged areas
    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.imshow(full_segmented_image)
    plt.title('Full Segmented Car with Damage Areas (Red)')

    plt.tight_layout()
    plt.show()

def draw_bounding_box(image, bbox, color=(255, 0, 0), thickness=2):
    """Draw a bounding box on the image."""
    x, y, w, h = bbox
    # Draw top and bottom edges
    image[int(y):int(y) + thickness, int(x):int(x) + w] = color
    image[int(y + h):int(y + h) + thickness, int(x):int(x) + w] = color
    # Draw left and right edges
    image[int(y):int(y) + h, int(x):int(x) + thickness] = color
    image[int(y):int(y) + h, int(x + w):int(x + w) + thickness] = color
    return image
visualize_car_damage(
    r"C:\Users\DELL\Downloads\ML\val",
    "updated_generic_damage_annotations",  # Replace with the correct filename without '.json'
    r"C:\Users\DELL\Downloads\ML\img",
    image_file=None
)