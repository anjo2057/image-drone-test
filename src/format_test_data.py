import os
import shutil
from sklearn.model_selection import train_test_split
import yaml




# Define the dataset folder containing the images and labels
dataset_folder = "../drone-dataset-uav/4/drone_dataset_yolo/dataset_txt"  # Replace with your actual dataset folder name

# Define paths for output folders under /kaggle/working/dataset/
working_dataset = "working/dataset"
train_images_folder = os.path.join(working_dataset, "train/images")
train_labels_folder = os.path.join(working_dataset, "train/labels")
val_images_folder = os.path.join(working_dataset, "val/images")
val_labels_folder = os.path.join(working_dataset, "val/labels")

# Create the output directories if they don't exist
os.makedirs(train_images_folder, exist_ok=True)
os.makedirs(train_labels_folder, exist_ok=True)
os.makedirs(val_images_folder, exist_ok=True)
os.makedirs(val_labels_folder, exist_ok=True)

# Initialize lists to store image and label file paths
images = []
labels = []

# Collect image and label file paths from the dataset folder
for file_name in os.listdir(dataset_folder):
    if file_name.endswith((".jpg", ".png")):  # Assuming images are .jpg or .png
        base_name = os.path.splitext(file_name)[0]
        label_file = base_name + ".txt"  # Assuming labels are in YOLO format (.txt)
        
        # Check if the corresponding label file exists
        if os.path.exists(os.path.join(dataset_folder, label_file)):
            images.append(file_name)
            labels.append(label_file)

# Split the data into training and validation sets (80% train, 20% val)
train_images, val_images, train_labels, val_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

# Function to move files to the designated train/val directories
def move_files(file_list, src_folder, dest_folder):
    for file in file_list:
        shutil.copy(os.path.join(src_folder, file), os.path.join(dest_folder, file))

# Move the training images and labels to the train folder
move_files(train_images, dataset_folder, train_images_folder)
move_files(train_labels, dataset_folder, train_labels_folder)

# Move the validation images and labels to the val folder
move_files(val_images, dataset_folder, val_images_folder)
move_files(val_labels, dataset_folder, val_labels_folder)

# Create the YOLO YAML configuration
yolo_data = {
    "train": os.path.join(working_dataset, "train/images"),  # Path to training images
    "val": os.path.join(working_dataset, "val/images"),      # Path to validation images
    "nc": 1,  # Number of classes (update according to your dataset)
    "names": ["drone"]  # Replace with actual class names
}

# Define the output YAML file path
yaml_file = "../output/yolo_dataset.yaml"

# Write the YOLO dataset information to the YAML file
with open(yaml_file, "w") as file:
    yaml.dump(yolo_data, file)

print(f"YOLO dataset YAML file generated at: {yaml_file}")