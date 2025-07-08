# Adversarial Attacks on Object Detection Models

This repository demonstrates white-box adversarial attacks on object detection models, specifically targeting YOLOv5. It provides implementations of Projected Gradient Descent (PGD) and Adversarial Patch attacks with interactive Jupyter notebooks for experimentation and analysis.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Available Attacks](#available-attacks)
- [Usage Examples](#usage-examples)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements and demonstrates adversarial attacks against object detection models. The attacks are designed to fool YOLOv5 models by either:
- **PGD Attack**: Adding imperceptible noise to entire images
- **Adversarial Patch Attack**: Placing visible patches that cause misclassification

The repository includes interactive notebooks that allow you to:
- Configure attack parameters
- Generate adversarial examples
- Visualize attack results
- Evaluate attack effectiveness

## Features

- 🎯 **Multiple Attack Types**: PGD and Adversarial Patch implementations
- 📊 **Interactive Notebooks**: Easy-to-use Jupyter notebooks for experimentation
- ⚙️ **Configurable Parameters**: JSON-based configuration system
- 📈 **Visualization**: Built-in plotting and analysis tools
- 🎨 **Custom Datasets**: Support for custom image datasets

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd example_adversarial_attacks
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLOv5 model** (if not already present):
   The YOLOv5s model (`yolov5s.pt`) should be placed in the `data/` directory.

## Project Structure

```
example_adversarial_attacks/
├── README.md                                    # This file
├── requirements.txt                             # Python dependencies
├── .gitignore                                   # Git ignore rules
├── data/                                        # Data and configuration files
│   ├── yolov5s.pt                              # YOLOv5 model weights
│   ├── configs/                                 # Configuration files
│   │   └── pgd_default_config.json             # Default PGD configuration
│   ├── custom_images/                           # Input images for attacks
│   │   ├── airplane0.jpg
│   │   └── ...
│   ├── original/                                # Original detection results
│   └── ProjectedGradientDescent/                # PGD attack results
└── src/                                         # Source code
    ├── notebook_white_box_pgd.ipynb            # PGD attack notebook
    ├── notebook_white_box_adversarial_patch.ipynb  # Patch attack notebook
    ├── attacks/                                 # Attack implementations
    │   └── white_boxes/                         # White-box attacks
    │       ├── local_projected_gradient_descent.py
    │       └── local_adversarial_patch_pytorch.py
    └── utils/                                   # Utility functions
        ├── utils.py                             # General utilities
        └── utils_detector_yolo.py               # YOLO-specific utilities
```

## Quick Start

1. **Navigate to the source directory**:
   ```bash
   cd src/
   ```

2. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

3. **Open a notebook**:
   - For PGD attacks: `notebook_white_box_pgd.ipynb`
   - For patch attacks: `notebook_white_box_adversarial_patch.ipynb`

4. **Run the cells** following the step-by-step instructions in the notebook.

## Configuration

Attacks are configured using JSON files in the `data/configs/` directory. The configuration includes:

### PGD Attack Parameters
```json
{
  "THRESHOLD": 0.6,
  "TARGET_CLASS": null,
  "VICTIM_CLASS": "airplane",
  "BATCH_SIZE": 5,
  "IOU_THRESHOLD": 0.5,
  "MAX_ITER": 100,
  "NORM": "inf",
  "EPS": 1.0,
  "TARGET_LOCATION": [50, 50],
  "TARGET_SHAPE": [3, 100, 100],
  "IMAGES_TO_DISPLAY": 3,
  "FOLDER_NAME": "original"
}
```

### Configuration Parameters
- **`THRESHOLD`**: Confidence threshold for filtering predictions
- **`TARGET_CLASS`**: Target class for attack (null for untargeted)
- **`VICTIM_CLASS`**: Class to attack (empty string for all classes)
- **`BATCH_SIZE`**: Number of images to process
- **`MAX_ITER`**: Number of attack iterations
- **`NORM`**: Perturbation norm ("1", "2", or "inf")
- **`EPS`**: Maximum perturbation magnitude

## Available Attacks

### 1. Projected Gradient Descent (PGD)
- **Type**: White-box attack
- **Method**: Iterative gradient-based optimization
- **Target**: Entire image with imperceptible noise
- **Notebook**: `notebook_white_box_pgd.ipynb`

### 2. Adversarial Patch
- **Type**: White-box attack
- **Method**: Optimized visible patches
- **Target**: Localized regions in images
- **Notebook**: `notebook_white_box_adversarial_patch.ipynb`

## Usage Examples

### Running a PGD Attack
```python
# Load configuration
config = load_config("../data/configs/pgd_default_config.json")

# Create detector
detector = UtilsDetectorYolo(config.BATCH_SIZE, config.THRESHOLD, config.VICTIM_CLASS)

# Initialize attack
attack = LocalProjectedGradientDescent(
    estimator=detector,
    images=detector.images,
    orig_predictions=original_predictions,
    target_class=config.TARGET_CLASS
)

# Generate adversarial examples
attack.generate(
    images=detector.images,
    norm=config.NORM,
    eps=config.EPS,
    max_iter=config.MAX_ITER
)

# Apply the attack to the images 
attack_predictions, adversarial_examples = attack.apply_attack_to_image(
    image=IMAGES,
    train_on=len(IMAGES),
    threshold=config.THRESHOLD,
)
```

### Custom Image Dataset
Place your images in `data/custom_images/` and update the configuration:
```json
{
  "FOLDER_NAME": "custom_images",
  "VICTIM_CLASS": "person"
}
```

## Results

Attack results are automatically saved in organized folders:
- **Original detections**: `data/original/`
- **PGD results**: `data/ProjectedGradientDescent/`
- **Patch results**: `data/AdversarialPatch/`

Each result includes:
- Visualized detection boxes
- Generated adversarial examples

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- NumPy
- Matplotlib
- OpenCV
- ART (Adversarial Robustness Toolbox)
- Jupyter Notebook

See `requirements.txt` for complete dependency list.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-attack`)
3. Commit your changes (`git commit -am 'Add new attack method'`)
4. Push to the branch (`git push origin feature/new-attack`)
5. Create a Pull Request


## Acknowledgments

- YOLOv5 by Ultralytics
- Adversarial Robustness Toolbox (ART)

---

**⚠️ Disclaimer**: This tool is for research and educational purposes only. Use responsibly and in accordance with applicable laws and regulations.