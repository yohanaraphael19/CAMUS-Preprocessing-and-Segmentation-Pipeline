import nibabel as nib
import os

def check_nifti_shapes(directory):
    """
    Checks and prints the shape of NIFTI files in a given directory.
    
    Args:
        directory (str): Path to the directory containing NIFTI files.
    """
    for file_name in os.listdir(directory):
        if file_name.endswith(".nii.gz"):
            file_path = os.path.join(directory, file_name)
            nifti_image = nib.load(file_path)
            data = nifti_image.get_fdata()
            print(f"File: {file_name}")
            print(f"Shape: {data.shape}")
            print("-" * 40)

# Example Usage
directory_path = r"C:/Users/Blessedsoul/PycharmProjects/Segmentation 1/NIFTI TO PNG/CAMUS_public/database_nifti/patient0001"  # Replace with the path to your directory
check_nifti_shapes(directory_path)
