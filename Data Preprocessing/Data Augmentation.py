import os
from PIL import Image
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from tqdm import tqdm

# Path to the dataset
data_dir = 'H:/Deep Learning/Pest and Insect Detection/Image Data/train'
augmented_data_dir = 'H:/Deep Learning/Pest and Insect Detection/Image Data/train_augmented'

# Define the transformation for image augmentation
augment_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.RandomErasing()
])

# Function to load images from folders
def load_images_from_folders(data_dir):
    dataset = ImageFolder(root=data_dir)
    class_dict = {v: k for k, v in dataset.class_to_idx.items()}
    images, labels = [], []
    for img, label in dataset:
        images.append(img)
        labels.append(label)
    return images, labels, class_dict

# Function to save augmented images
def save_augmented_images(images, labels, class_dict, target_num_images):
    if not os.path.exists(augmented_data_dir):
        os.makedirs(augmented_data_dir)

    class_counts = {label: 0 for label in class_dict.keys()}

    for i in tqdm(range(len(images))):
        img, label = images[i], labels[i]
        class_name = class_dict[label]
        class_dir = os.path.join(augmented_data_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        # Save the original image
        img_tensor = transforms.ToTensor()(img)
        save_image(img_tensor, os.path.join(class_dir, f'{class_counts[label]}.png'))
        class_counts[label] += 1

        # Generate and save augmented images
        while class_counts[label] < target_num_images:
            augmented_img_tensor = augment_transforms(img)
            save_image(augmented_img_tensor, os.path.join(class_dir, f'{class_counts[label]}.png'))
            class_counts[label] += 1

# Load images
images, labels, class_dict = load_images_from_folders(data_dir)

# Find the maximum number of images in any class
max_images_per_class = max([labels.count(l) for l in class_dict.keys()])

# Save augmented images ensuring each class has the same number of images
save_augmented_images(images, labels, class_dict, target_num_images=max_images_per_class)
