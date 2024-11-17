import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Define paths
data_dir = 'H:/Deep Learning/Pest and Insect Detection/Image Data/train_augmented'
processed_data_dir = 'H:/Deep Learning/Pest and Insect Detection/Processed Image Data/Histogram Equalization'

def histogram_equalization(image_path, save_path):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply histogram equalization
    equalized_img = cv2.equalizeHist(img)

    # Save the result
    cv2.imwrite(save_path, equalized_img)

def process_and_save_images(data_dir, processed_data_dir):
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)

    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, data_dir)
                save_dir = os.path.join(processed_data_dir, relative_path)

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                save_path = os.path.join(save_dir, file)
                histogram_equalization(image_path, save_path)

# Process and save images
process_and_save_images(data_dir, processed_data_dir)

print("Histogram equalization completed. Processed images are saved.")