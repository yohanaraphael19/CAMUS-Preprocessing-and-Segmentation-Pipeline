# CAMUS-Preprocessing-and-Segmentation-Pipeline
Pipeline for preprocessing the CAMUS echocardiography dataset and training a U-Net-based model for automated heart segmentation. Includes 
        1. **NIFTI to PNG conversion** (process_camus_dataset.py)
        2. **Image rotation** (rotate the PNG files 90° to get correct position)
        3. **A structured deep learning workflow for improved accuracy and efficiency**

#project structure
#├── dataset.py            Dataset and preprocessing for PNG files
#├── model.py              U-Net architecture
#├── utils.py              Helper functions (e.g., metrics, visualizations)
#├── train_validation.py   Training and validation loop
#├── main.py               Main script to coordinate everything
#├── saved_images/         Folder to save segmentation results

## Problem Statement
Accurate segmentation of echocardiography images is crucial for diagnosing and managing cardiovascular diseases. Echocardiography provides a non-invasive method to assess heart structure and function, but manual segmentation of these images is time-consuming, prone to variability, and requires expert knowledge. Automated segmentation using deep learning can improve consistency, efficiency, and accessibility of echocardiographic analysis.

## Solution
This repository provides a comprehensive pipeline for preprocessing the CAMUS dataset and training a U-Net-based model for echocardiography segmentation. The preprocessing scripts convert the original NIFTI files into 2D PNG slices, rotate images and masks for correct alignment, and ensure compatibility with the segmentation model. Key enhancements include:

- **Improved Preprocessing**: Handling errors, normalization, and rotation alignment.
- **Efficient Codebase**: Modular functions, multithreading for rotation, and robust path handling.
- **Integration**: Outputs structured to feed directly into the U-Net segmentation workflow.
- **Results**: Improved predictions and Dice scores through refined preprocessing.

## Visuals!

![result_3](https://github.com/user-attachments/assets/b5e97a3a-5c5c-457f-8f4f-bb4ab50913f8)
![training_metrics](https://github.com/user-attachments/assets/74c4dc2d-51b8-4720-8c56-f814e92ed367)


## Usage

# Echocardiography Segmentation Workflow

## Overview
This repository provides a structured workflow for processing and analyzing echocardiography images using deep learning. The primary steps include converting NIFTI files to PNG, correcting image orientation, and implementing a deep learning workflow to achieve accurate segmentation.

## Data Source
The dataset used in this workflow is the **CAMUS dataset**, which can be downloaded from the following link:  
[https://www.creatis.insa-lyon.fr/Challenge/camus/](https://www.creatis.insa-lyon.fr/Challenge/camus/)

### Citation
If you use the CAMUS dataset in your research, you **must cite** the following paper:

> S. Leclerc, E. Smistad, J. Pedrosa, A. Ostvik, et al.,  
> "**Deep Learning for Segmentation using an Open Large-Scale Dataset in 2D Echocardiography**,"  
> *IEEE Transactions on Medical Imaging*, vol. 38, no. 9, pp. 2198-2210, Sept. 2019.  
> DOI: [10.1109/TMI.2019.2900516](https://doi.org/10.1109/TMI.2019.2900516)

---

## Workflow Steps

### 1. Download and Unzip Data
- Download the CAMUS dataset from the official website.  
- After downloading, unzip the dataset to prepare for processing.

### 2. Convert NIFTI to PNG
- Use the provided script (`process_camus_dataset.py`) to convert NIFTI files into PNG images for easier manipulation and visualization.  

### 3. Rotate Images
- Rotate the PNG files by **90°** to ensure the correct orientation of the images.  

### 4. Deep Learning Workflow
- Follow the structured deep learning pipeline to process the data for training and segmentation.  
- The pipeline is designed to improve **accuracy** and **efficiency**.  

---

## Requirements
- Python 3.8+
- PyTorch
- NumPy
- Matplotlib

---

## Notes
- Ensure all dependencies are installed before running the scripts.
- For any questions or issues, feel free to open an issue or reach out to the repository maintainer.


## Contribution
Contributions are welcome! Please open issues or submit pull requests for bugs, enhancements, or new features.



