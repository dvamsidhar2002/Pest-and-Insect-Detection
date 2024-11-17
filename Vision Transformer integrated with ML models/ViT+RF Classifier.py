#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import torch
from torchvision import transforms
from torchvision.models import vit_b_16
from PIL import Image
from tqdm import tqdm
import joblib

# Path to the dataset
data_dir = 'H:/Deep Learning/Pest and Insect Detection/Image Data/train'

# Define the transformation for image preprocessing before passing to ViT
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to ViT input size
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for pre-trained ViT
])

# Function to load images from multiple folders and return them as a list
def load_images_from_folders(data_dir, transform=None):
    images = []
    labels = []
    class_labels = os.listdir(data_dir)
    class_dict = {label: idx for idx, label in enumerate(class_labels)}

    for label in class_labels:
        folder_path = os.path.join(data_dir, label)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert('RGB')  # Open the image as RGB
            if transform:
                img = transform(img)  # Apply transformations (e.g., normalization)
            images.append(img)
            labels.append(class_dict[label])
    
    return images, labels

# Load the images
images, labels = load_images_from_folders(data_dir, transform)

# Stack images into a tensor and labels into a numpy array
images_tensor = torch.stack(images)
labels = np.array(labels)

# Load pre-trained Vision Transformer (ViT) model
vit_model = vit_b_16(pretrained=True)
vit_model.eval()  # Set the model to evaluation mode

# Move the model to GPU if available (if not, will run on CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_model.to(device)

# Function to extract features from ViT model with batching
def extract_features(images_tensor, model, batch_size=32):
    all_features = []
    num_batches = len(images_tensor) // batch_size + (1 if len(images_tensor) % batch_size != 0 else 0)

    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(images_tensor))
        batch = images_tensor[start_idx:end_idx].to(device)  # Move batch to GPU if available
        
        with torch.no_grad():  # No need to compute gradients for feature extraction
            outputs = model(batch)  # Pass through ViT
            all_features.append(outputs.cpu().numpy())  # Move features to CPU

    return np.concatenate(all_features, axis=0)

# Extract features from the images in smaller batches
features = extract_features(images_tensor, vit_model, batch_size=32)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random Forest classifier
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Save the trained model using joblib
model_path = 'H:/Deep Learning/Pest and Insect Detection/Vision Transformer integrated with ML models/ViT_RF_model.pkl'
joblib.dump(rf_classifier, model_path)
print(f"Model saved to {model_path}")

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[1]:


import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import torch
from torchvision import transforms
from torchvision.models import vit_b_16
from PIL import Image
from tqdm import tqdm
import joblib

# Path to the dataset
data_dir = 'H:/Deep Learning/Pest and Insect Detection/Image Data/train_augmented'

# Define the transformation for image preprocessing before passing to ViT
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to ViT input size
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for pre-trained ViT
])

# Function to load images from multiple folders and return them as a list
def load_images_from_folders(data_dir, transform=None):
    images = []
    labels = []
    class_labels = os.listdir(data_dir)
    class_dict = {label: idx for idx, label in enumerate(class_labels)}

    for label in class_labels:
        folder_path = os.path.join(data_dir, label)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert('RGB')  # Open the image as RGB
            if transform:
                img = transform(img)  # Apply transformations (e.g., normalization)
            images.append(img)
            labels.append(class_dict[label])
    
    return images, labels

# Load the images
images, labels = load_images_from_folders(data_dir, transform)

# Stack images into a tensor and labels into a numpy array
images_tensor = torch.stack(images)
labels = np.array(labels)

# Load pre-trained Vision Transformer (ViT) model
vit_model = vit_b_16(pretrained=True)
vit_model.eval()  # Set the model to evaluation mode

# Move the model to GPU if available (if not, will run on CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_model.to(device)

# Function to extract features from ViT model with batching
def extract_features(images_tensor, model, batch_size=32):
    all_features = []
    num_batches = len(images_tensor) // batch_size + (1 if len(images_tensor) % batch_size != 0 else 0)

    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(images_tensor))
        batch = images_tensor[start_idx:end_idx].to(device)  # Move batch to GPU if available
        
        with torch.no_grad():  # No need to compute gradients for feature extraction
            outputs = model(batch)  # Pass through ViT
            all_features.append(outputs.cpu().numpy())  # Move features to CPU

    return np.concatenate(all_features, axis=0)

# Extract features from the images in smaller batches
features = extract_features(images_tensor, vit_model, batch_size=32)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random Forest classifier
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Save the trained model using joblib
model_path = 'H:/Deep Learning/Pest and Insect Detection/Vision Transformer integrated with ML models/ViT_RF_model_augmented_data.pkl'
joblib.dump(rf_classifier, model_path)
print(f"Model saved to {model_path}")

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[ ]:




