# Diagnostic Imaging Classification: Chest Radiograph Pathology Detection

<div align="center">

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Azure Machine Learning](https://img.shields.io/badge/Azure_ML-0078D4?style=for-the-badge&logo=microsoft-azure&logoColor=white)

</div>

<br>

## Executive Summary
This repository contains an end-to-end computer vision evaluation pipeline focused on classifying chest X-ray radiograph tensors as NORMAL or PNEUMONIA. Engineered for integration with Azure Machine Learning (AML), this project benchmarks a custom Convolutional Neural Network (CNN) against heavyweight, pre-trained transfer learning architectures (ResNet, VGG) to evaluate precision, computational efficiency, and edge deployment feasibility.

### Experiment Configuration
| Parameter | Specification |
|-----------|---------------|
| **Core Framework** | PyTorch (Optimized for Azure GPU Compute nodes) |
| **Ingestion Source** | `chest_xray/` (Standardized categorical image directory) |
| **Architectures Checked**| Custom CNN, ResNet-18, ResNet-50, VGG-16, VGG-19 |
| **Model Checkpoints** | Persisted locally as `.pth` telemetry weights |


## Pipeline Artifacts
```text
.
├── 2025AB05088_cnn_assignment.ipynb   # Main Computer Vision Experimentation Pipeline
└── Dockerfile                         # Enterprise Azure ML Container Specification
```

## Evaluated Topologies & Benchmarks
Our experimentation grid evaluates multiple spatial feature extractors:
1. **Custom CNN Architecture:** A lightweight, low-latency network optimized for rapid inference.
2. **ResNet Variants (18, 50):** Deep residual mapping to mitigate vanishing gradients in deep topological spaces.
3. **VGG Variants (16, 19):** Uniform block architectures evaluating the efficacy of deep spatial hierarchy representation.

> *Weights are captured systematically as `custom_cnn_best.pth`, `resnet18_best.pth`, etc., maintaining isolated checkpoint integrity for robust model roll-backs. These files were larger than 25 MB, hence unable to add them to the repository.*

## Environment Setup & Initialization
This vision pipeline requires `python 3.7+` structured with the underlying dimensional manipulation libraries:
`torch`, `torchvision`, `numpy`, `matplotlib`, `pandas`, `scikit-learn`

**Local Environment / Azure Compute Instance Bootstrap:**
```bash
# Provision foundational dependencies
python -m pip install --upgrade pip
python -m pip install torch torchvision numpy matplotlib pandas scikit-learn
```

> **Data Volume Note:** Please ensure the `chest_xray/` imaging dataset is mounted securely at the pipeline root (via local filesystem or an Azure Blob Storage Datalake mount) as raw imaging manifests are omitted from source control due to dimensional constraints.


## Containerization Strategy
To emulate our forthcoming Azure ML managed compute environment, this repository includes a hardened `Dockerfile`. The container specification utilizes Microsoft's official Azure ML OpenMPI/CUDA base (`ubuntu22.04`), securely provisions PyTorch's `torchvision` libraries mapping to `cuDNN`, and shifts to a non-root `mlops_sys` profile before exposing the computational environment.

**Container Build & Execution Run-Book:**
```bash
# Compile the Computer Vision MLOps image
docker build -t aml-radiograph-pipeline:v1 .

# Spin up a localized container instance (bind-mounting telemetry & imagery)
docker run --rm -v "$(pwd):/workspace/vision_experiment" \
                -v "/path/to/local/chest_xray:/workspace/vision_experiment/chest_xray" \
                aml-radiograph-pipeline:v1
```


## Execution & Deployment Sequence
1. **Volume Mount Validation:** Verify the structural integrity of the `chest_xray/` corpus (`train/`, `test/`, `val/`).
2. **Interactive Run:** Open the `2025AB05088_cnn_assignment.ipynb` pipeline. Execute all cells systematically to re-synthesize model weights, compute confusion matrices, and render gradient evaluations.


## Maintainers
- **P L V S ADITHYA** - Lead Architect / MLOps Engineer
- [2025ab05088@wilp.bits-pilani.ac.in](mailto:2025ab05088@wilp.bits-pilani.ac.in)

## License
Proprietary evaluation codebase. Internal distribution and review only.
