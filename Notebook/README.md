# Common Houseplant Identification with PlantNet

This project uses the pre-trained PlantNet model to identify common houseplants and provide care recommendations. The PlantNet model has been trained on thousands of plant images, making it more accurate for plant identification compared to training a model from scratch.

## How to Use This Project

### 1. Running the Jupyter Notebook

The main implementation is in the `Houseplant_Identification_with_PlantNet.ipynb` notebook. To run it:

1. Install the required dependencies: `pip install -r requirements.txt`
2. Launch Jupyter: `jupyter notebook`
3. Open `Houseplant_Identification_with_PlantNet.ipynb`
4. Run the cells in order

### 2. Using the PlantNet Pre-trained Model

The notebook demonstrates how to:
- Download and load the PlantNet pre-trained model
- Adapt it for houseplant identification
- Fine-tune it on your own dataset
- Use it for inference on new plant images

### 3. Obtaining PlantNet Weights

To use the actual PlantNet pre-trained weights:

1. Visit [PlantNet's GitHub repository](https://github.com/plantnet/plant-identification-api)
2. Follow their instructions for accessing pre-trained weights
3. Use the `utils.py` functions to load the weights

```python
from utils import load_model, get_plantnet_model

# Create the model architecture
model = get_plantnet_model(num_classes=1081)

# Load the pre-trained weights
model = load_model(model, "path/to/resnet18_weights_best_acc.tar", use_gpu=True)
```

## Project Structure

- `Houseplant_Identification_with_PlantNet.ipynb`: Main implementation notebook
- `utils.py`: Utility functions for using the PlantNet model
- `data/`: Directory for storing plant data and care information
- `models/`: Directory for storing model weights
- `requirements.txt`: List of required Python packages

## Key Features

1. **PlantNet Integration**: Uses a pre-trained model for better accuracy
2. **Transfer Learning**: Adapts a robust model to our specific houseplant classification task
3. **Care Recommendations**: Provides detailed care information for identified plants
4. **Interactive Interface**: Uses Gradio to create a user-friendly web interface

## Extending the Project

You can extend this project by:

1. **Adding more plant species** to the care database
2. **Collecting real training images** for each houseplant species
3. **Adding plant health analysis** to detect diseases and nutrient deficiencies
4. **Deploying the application** as a standalone web service or mobile app

## Credits

- [PlantNet](https://plantnet.org/) for their amazing work on plant identification
- [Gradio](https://gradio.app/) for the interactive interface framework
- [PyTorch](https://pytorch.org/) for the deep learning framework

## License

This project is provided for educational purposes. The PlantNet model and weights have their own licensing - please check their repository for details.
