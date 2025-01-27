import os
import nibabel as nib
import numpy as np
from PIL import Image

def nifti_to_slices(nifti_path, output_dir, slice_axis=0, prefix=""):
    """
    Extract 2D slices from a NIFTI file and save as PNG images.

    Args:
        nifti_path (str): Path to the NIFTI file.
        output_dir (str): Directory to save the slices.
        slice_axis (int): Axis to slice the 3D volume (default: 2).
        prefix (str): Prefix for naming slices.
    """
    os.makedirs(output_dir, exist_ok=True)
    img = nib.load(nifti_path)
    data = img.get_fdata()

    # Check if the data is 3D or 2D
    if len(data.shape) == 2:
        # If 2D, treat the entire array as a single slice
        slice_data = data
        slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data)) * 255  # Normalize
        slice_data = slice_data.astype(np.uint8)

        output_path = os.path.join(output_dir, f"{prefix}_slice_001.png")
        Image.fromarray(slice_data).save(output_path)

    elif len(data.shape) == 3:
        # If 3D, slice along the given axis
        for i in range(data.shape[slice_axis]):
            slice_data = np.take(data, i, axis=slice_axis)
            slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data)) * 255  # Normalize
            slice_data = slice_data.astype(np.uint8)

            output_path = os.path.join(output_dir, f"{prefix}_slice_{i+1:03d}.png")
            Image.fromarray(slice_data).save(output_path)

    else:
        raise ValueError(f"Unsupported data shape: {data.shape}")


def process_camus_dataset(input_dir, output_image_dir, output_mask_dir, slice_axis=2):
    """
    Process the CAMUS dataset, extracting slices from images and masks.

    Args:
        input_dir (str): Path to the CAMUS NIFTI dataset.
        output_image_dir (str): Directory to save image slices.
        output_mask_dir (str): Directory to save mask slices.
        slice_axis (int): Axis to slice the 3D volumes.
    """
    for patient in os.listdir(input_dir):
        patient_path = os.path.join(input_dir, patient)
        if not os.path.isdir(patient_path):
            continue

        for file_name in os.listdir(patient_path):
            if file_name.endswith(".nii.gz"):
                file_path = os.path.join(patient_path, file_name)

                if "gt" in file_name:
                    # Ground truth mask
                    prefix = file_name.split(".")[0]
                    patient_mask_dir = os.path.join(output_mask_dir, patient)
                    nifti_to_slices(file_path, patient_mask_dir, slice_axis, prefix)
                else:
                    # Image
                    prefix = file_name.split(".")[0]
                    patient_image_dir = os.path.join(output_image_dir, patient)
                    nifti_to_slices(file_path, patient_image_dir, slice_axis, prefix)

# Paths
input_dir = "C:/Users/Blessedsoul/PycharmProjects/Segmentation 1/NIFTI TO PNG/CAMUS_public/database_nifti"
output_image_dir = "C:/Users/Blessedsoul/PycharmProjects/Segmentation 1/NIFTI TO PNG/output_data/images"
output_mask_dir = "C:/Users/Blessedsoul/PycharmProjects/Segmentation 1/NIFTI TO PNG/output_data/masks"

# Process the dataset
process_camus_dataset(input_dir, output_image_dir, output_mask_dir, slice_axis=2)
