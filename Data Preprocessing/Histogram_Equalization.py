import cv2
import numpy as np
from matplotlib import pyplot as plt


def histogram_equalization(image_path):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply histogram equalization
    equalized_img = cv2.equalizeHist(img)

    # Show original and equalized images for comparison
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Equalized Image')
    plt.imshow(equalized_img, cmap='gray')
    plt.axis('off')

    plt.show()

    # Optionally, save the result
    cv2.imwrite('equalized_image.png', equalized_img)


# Call the function with your image path
histogram_equalization('H:/Deep Learning/Pest and Insect Detection/Image Data/train_augmented/bees/510.png')

#
