# CAMUS-Preprocessing-and-Segmentation-Pipeline
Pipeline for preprocessing the CAMUS echocardiography dataset and training a U-Net-based model for automated heart segmentation. Includes NIFTI to PNG conversion, image rotation, and a structured deep learning workflow for improved accuracy and efficiency.

# CAMUS Preprocessing and Segmentation Pipeline

## Problem Statement
Accurate segmentation of echocardiography images is crucial for diagnosing and managing cardiovascular diseases. Echocardiography provides a non-invasive method to assess heart structure and function, but manual segmentation of these images is time-consuming, prone to variability, and requires expert knowledge. Automated segmentation using deep learning can improve consistency, efficiency, and accessibility of echocardiographic analysis.

## Solution
This repository provides a comprehensive pipeline for preprocessing the CAMUS dataset and training a U-Net-based model for echocardiography segmentation. The preprocessing scripts convert the original NIFTI files into 2D PNG slices, rotate images and masks for correct alignment, and ensure compatibility with the segmentation model. Key enhancements include:

- **Improved Preprocessing**: Handling errors, normalization, and rotation alignment.
- **Efficient Codebase**: Modular functions, multithreading for rotation, and robust path handling.
- **Integration**: Outputs structured to feed directly into the U-Net segmentation workflow.
- **Results**: Improved predictions and Dice scores through refined preprocessing.

## Visuals![result_0](https://github.com/user-attachments/assets/bc5deffd-1054-4db1-853c-60273e2371d1)

![Ground Truth vs. Predictions]

## Usage

### Preprocessing steps
- python nifti_shape.py --input_dir CAMUS_public/ database_nifti
-   python process_camus_dataset.py --input_dir CAMUS_public/database_nifti \
    --output_image_dir output_data/images \
    --output_mask_dir output_data/masks \
    --slice_axis 2
- python png-rotate-90-ac.py --input_dir output_data/images \
    --output_dir output_data/rotated/images
python png-rotate-90-ac.py --input_dir output_data/masks \
    --output_dir output_data/rotated/masks

## Contribution
Contributions are welcome! Please open issues or submit pull requests for bugs, enhancements, or new features.



