import os
import shutil
import pandas as pd

# Load the metadata CSV file
metadata_path = r"C:/Users/sayan/OneDrive/Desktop/new python/SKIN DESEASE/HAM10000_metadata (1).csv"  # Update with the correct path to your metadata CSV file
metadata = pd.read_csv(metadata_path)

# Define paths to the image directories
train_dir = r"C:/Users/sayan/OneDrive/Desktop/new python/SKIN DESEASE/HAM10000_images_part_1"
test_dir = r"C:/Users/sayan/OneDrive/Desktop/new python/SKIN DESEASE/HAM10000_images_part_2"

# Function to organize images in a given directory
def organize_images(directory, metadata):
    for _, row in metadata.iterrows():
        image_id = row['image_id']
        class_label = row['dx']
        image_filename = f"{image_id}.jpg"  # Assuming the images are in JPEG format

        # Check if the image exists in the directory
        image_path = os.path.join(directory, image_filename)
        if os.path.exists(image_path):
            class_dir = os.path.join(directory, class_label)
            os.makedirs(class_dir, exist_ok=True)
            shutil.move(image_path, os.path.join(class_dir, image_filename))

# Organize images in the train and test directories
organize_images(train_dir, metadata)
organize_images(test_dir, metadata)