# Chest X-Ray Pneumonia Detection - CNN Assignment

This repository contains code and resources for a deep learning assignment focused on classifying chest X-ray images as NORMAL or PNEUMONIA using Convolutional Neural Networks (CNNs). The project is structured for submission and review by educators.

## Repository Structure

- **2025AB05088_cnn_assignment.ipynb**: Main Jupyter notebook containing code, analysis, and results for the assignment.
- **cnn_assignment_code.py**: Python script with core CNN implementation.
- **CNN_assignment_algo.txt**: Algorithm description and approach.
- **custom_cnn_best.pth, resnet18_best.pth, resnet50_best.pth, vgg16_best.pth, vgg19_best.pth**: Saved PyTorch model weights for different architectures.
- **chest_xray/**: Dataset directory with `train/`, `test/`, and (optionally) `val/` folders, each containing `NORMAL/` and `PNEUMONIA/` subfolders.
- **Other scripts**: Utilities for data processing, analysis, and notebook generation (e.g., `reorganize_dataset.py`, `update_analysis.py`).

## How to Use

1. **Dataset**: Ensure the `chest_xray/` directory contains the required data in the correct structure:
   - `chest_xray/train/NORMAL/`, `chest_xray/train/PNEUMONIA/`
   - `chest_xray/test/NORMAL/`, `chest_xray/test/PNEUMONIA/`
   - (Optional) `chest_xray/val/NORMAL/`, `chest_xray/val/PNEUMONIA/`

2. **Running the Notebook**:
   - Open `2025AB05088_cnn_assignment.ipynb` in Jupyter Notebook or VS Code.
   - Run all cells sequentially to train, evaluate, and analyze the CNN models.
   - The notebook includes code for data loading, preprocessing, model training, evaluation, and result visualization.

3. **Model Weights**:
   - Pretrained/final model weights are provided as `.pth` files for reproducibility.
   - To use a specific model, load the corresponding `.pth` file in the notebook or scripts.

4. **Scripts**:
   - Utility scripts are provided for dataset reorganization, notebook generation, and analysis updates. Refer to comments in each script for usage details.

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- numpy, matplotlib, pandas, scikit-learn
- Jupyter Notebook or VS Code (for running `.ipynb` files)

Install dependencies using:
```bash
pip install torch torchvision numpy matplotlib pandas scikit-learn
```

## Notes

- The main notebook is self-contained and includes explanations for each step.
- All code is original and intended for educational purposes.
- Please ensure the dataset is available locally as it is not included due to size constraints.

## Author
- Student ID: 2025AB05088
- Course: Deep Neural Network (Assignment 2)

---

For any questions or clarifications, please contact the repository owner.
