# PlantVision: Houseplant Identification System

This project implements a machine learning-based application that helps users identify common houseplants from images and provides basic care recommendations. The application uses computer vision techniques to classify houseplant images and displays relevant care information for the identified plants.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technical Architecture](#technical-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Training Process](#training-process)
- [Future Extensions](#future-extensions)
- [Acknowledgments](#acknowledgments)

## Overview

The Common Houseplant Identification Assistant was developed to help plant owners identify their houseplants and obtain proper care guidelines. Many houseplant owners struggle with plant identification, which often leads to improper care and preventable plant health issues. This application addresses this problem by providing a user-friendly interface to identify plants from images and access care recommendations.

The application can identify approximately 20 common houseplant species with reasonable accuracy and provides detailed care information covering light requirements, watering needs, temperature preferences, and more.

## Features

- **Image-based Plant Identification**: Upload or take a photo to identify your houseplant
- **Confidence Scores**: View confidence levels for predictions
- **Multiple Identification Options**: See alternative possibilities if the primary prediction is uncertain
- **Detailed Care Information**: Access specific care guidelines for identified plants, including:
  - Light requirements
  - Watering needs
  - Soil preferences
  - Temperature ranges
  - Humidity levels
  - Fertilization guidelines
  - Common issues and troubleshooting
  - Toxicity information (for pet owners)
- **User-friendly Interface**: Simple, intuitive web interface built with Gradio
- **Mobile-friendly Design**: Responsive interface that works well on various devices

## Technical Architecture

The application consists of three main components:

1. **Data Processing Module**: Scripts for collecting, organizing, and preprocessing plant images
2. **Model Component**: Fine-tuned vision model for plant classification
3. **User Interface**: Web-based interface for interacting with the model

The system uses transfer learning with pre-trained models (MobileNetV3, ResNet, EfficientNet, or Vision Transformer) fine-tuned on a dataset of common houseplant images. The web interface is built using Gradio, making it easy to deploy and use.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/houseplant-identification-assistant.git
   cd houseplant-identification-assistant
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download pre-trained model (optional):
   ```bash
   # If you have a pre-trained model, place it in the models/latest directory
   mkdir -p models/latest
   # Copy your model files to models/latest/
   ```

## Usage

### Running the Web Interface

1. Start the Gradio web application:
   ```bash
   python app/app.py
   ```

2. Open your web browser and navigate to the URL displayed in the terminal (typically http://127.0.0.1:7860).

3. Upload an image of your houseplant and click "Identify Plant" to get identification results and care recommendations.

### Training a New Model

If you want to train a custom model:

1. Prepare your dataset:
   ```bash
   python data/download_dataset.py
   ```

2. Train the model:
   ```bash
   python model/train.py --model mobilenet --epochs 10 --batch-size 32
   ```

3. Use the newly trained model:
   ```bash
   python app/app.py --model-dir outputs/mobilenet_default_YYYYMMDD_HHMMSS
   ```

## Project Structure

```
houseplant-assistant/
│
├── data/
│   ├── download_dataset.py     # Script to download and organize plant dataset
│   ├── preprocess_images.py    # Image preprocessing utilities
│   └── care_database.json      # Care information for each plant species
│
├── model/
│   ├── model_selection.py      # Model architecture selection
│   ├── train.py                # Training script
│   ├── inference.py            # Inference utilities
│   └── model_utils.py          # Utility functions for model operations
│
├── app/
│   ├── app.py                  # Main Gradio application
│   ├── utils.py                # Utility functions for the app
│   └── styling.py              # CSS and styling for the Gradio interface
│
├── models/                     # Directory for storing trained models
│   └── latest/                 # Latest/default model for inference
│
├── outputs/                    # Training outputs and logs
│
├── examples/                   # Example plant images for the interface
│
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## Training Process

The model training process follows these steps:

1. **Dataset Preparation**: The system collects and organizes a dataset of 50-100 common houseplant species, primarily from the PlantNet dataset.

2. **Data Preprocessing**: Images are resized, normalized, and augmented to improve model generalization.

3. **Model Selection**: The system selects an appropriate pre-trained vision model (MobileNetV3, ResNet, EfficientNet, or Vision Transformer).

4. **Fine-tuning**: The pre-trained model is fine-tuned on the houseplant dataset using transfer learning.

5. **Evaluation**: The model is evaluated on a validation set to assess its accuracy and performance.

6. **Export**: The trained model is exported in a format suitable for web deployment.

## Future Extensions

Potential enhancements for future versions:

1. **Expanded Species Coverage**: Increase the number of identifiable species beyond the initial 20
2. **Plant Health Analysis**: Add capability to detect common plant diseases and nutrient deficiencies
3. **Growth Tracking**: Implement features to monitor plant growth and health over time
4. **Community Features**: Add user contributions for rare species and care tips
5. **Offline Capabilities**: Develop a progressive web app with offline functionality

## Acknowledgments

- PlantNet dataset for providing plant images
- Hugging Face for model hosting and deployment infrastructure
- Gradio for the user interface framework
- The open-source community for various libraries and tools used in this project
