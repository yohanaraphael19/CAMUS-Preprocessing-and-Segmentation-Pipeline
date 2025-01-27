import os
from PIL import Image

# Define paths
base_path = r"C:/Users/Blessedsoul/PycharmProjects/Segmentation 1/NIFTI TO PNG/output_data"
images_path = os.path.join(base_path, "images")
masks_path = os.path.join(base_path, "masks")
output_base_path = os.path.join(base_path, "good-output")
output_images_path = os.path.join(output_base_path, "images")
output_masks_path = os.path.join(output_base_path, "masks")

# Ensure output directories exist
os.makedirs(output_images_path, exist_ok=True)
os.makedirs(output_masks_path, exist_ok=True)

# Function to process and save rotated images
def rotate_and_save(input_dir, output_dir):
    for patient_folder in os.listdir(input_dir):
        patient_input_path = os.path.join(input_dir, patient_folder)
        patient_output_path = os.path.join(output_dir, patient_folder)
        os.makedirs(patient_output_path, exist_ok=True)

        for file_name in os.listdir(patient_input_path):
            file_path = os.path.join(patient_input_path, file_name)
            if file_name.endswith(".png"):  # Ensure only PNG files are processed
                with Image.open(file_path) as img:
                    rotated_img = img.rotate(90, expand=True)  # Rotate counterclockwise
                    rotated_img.save(os.path.join(patient_output_path, file_name))

# Rotate images and masks
rotate_and_save(images_path, output_images_path)
rotate_and_save(masks_path, output_masks_path)

print("All images and masks have been rotated and saved in the 'good-output' folder.")
