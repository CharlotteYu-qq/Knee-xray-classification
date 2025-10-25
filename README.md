# Knee X-ray Classification based on KL grades

## Project Description
This project implements a deep learning model for classifying knee X-ray images according to the Kellgren-Lawrence (KL) grading system (0-4). The project includes 5-fold cross-validation and comprehensive evaluation metrics.

## Project Structure
knee-xray-classification/
├── main.py # Main training script
├── trainer.py # Training and validation logic
├── model.py # Model architecture
├── dataset.py # Data loading and preprocessing
├── args.py # Configuration arguments
├── utils.py # Utility functions (plotting, etc.)
├── test_precision.py # Precision evaluation script
├── requirements.txt # Python dependencies
└── README.md # This file


## Installation
```bash
# Clone the repository
git clone https://github.com/YOUR-USERNAME/knee-xray-classification.git
cd knee-xray-classification

# Install dependencies
pip install torch torchvision opencv-python pandas numpy scikit-learn matplotlib

# Train the model with 5-fold cross-validation
python main.py

# Evaluate precision metrics
python test_precision.py