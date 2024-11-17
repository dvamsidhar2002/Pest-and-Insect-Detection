import os

# Path to the augmented data directory
augmented_data_dir = 'H:/Deep Learning/Pest and Insect Detection/Image Data/train_augmented'

def count_images_in_folders(directory):
    class_counts = {}
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            num_images = len([file for file in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, file))])
            class_counts[class_name] = num_images
    return class_counts

# Count the number of images in each folder
image_counts = count_images_in_folders(augmented_data_dir)

# Print the number of images in each class
for class_name, num_images in image_counts.items():
    print(f"Class '{class_name}': {num_images} images")
