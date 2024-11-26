# Thesis Project: Multi-Task Feature Autoencoder (MTFAE)

This repository contains the code and data for a thesis project focused on the development and evaluation of a Multi-Task Feature Autoencoder (MTFAE). The MTFAE is designed to learn useful representations for multiple tasks simultaneously, potentially enhancing the performance of downstream machine learning models.

## Project Structure

- **MTFAE.py**: This script defines the architecture of the Multi-Task Feature Autoencoder. It includes model components and layers used for feature extraction and reconstruction.

- **attn.py**: Implements attention mechanisms to enhance the MTFAE model's ability to focus on the most relevant parts of the input data, particularly useful in sequence-based tasks.

- **data_loader.py**: Handles the loading and preprocessing of datasets to ensure the data is in the correct format for training and evaluation.

- **embed.py**: Embedding utility script that provides functionalities for embedding data points in a suitable representation format for the model.

- **main.py**: The main script to orchestrate the entire training and testing workflow. This script initializes the model, sets hyperparameters, and controls the training loop based on the configuration files.

- **solver.py**: Manages the optimization process, including the loss functions and backpropagation logic for training the MTFAE model.

- **Configuration Files** (`*.conf`): These configuration files, such as `activity_user1.conf`, contain various training parameters and settings. They can be modified to customize the training scenarios for different datasets.

- **Data Files** (`*.npy` and `*.pth`): NumPy files store preprocessed data arrays, and PyTorch checkpoint files (`*.pth`) save the trained model's weights. These files facilitate resuming training and evaluating the model without starting from scratch.

- **Main.rar**: A compressed archive containing additional resources or data required for the project. Please extract its contents before running the scripts.

## Getting Started

### Prerequisites

- Python 3.11
- PyTorch
- NumPy
- Other dependencies as listed in `requirements.txt` (if available)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Gautmnikhil/Thesis.git
   cd Thesis
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. Configure the parameters in the configuration files (`*.conf`) to suit your dataset and model settings.
2. Run the main script to train the model:
   ```bash
   python main.py
   ```
3. Checkpoints will be saved automatically during training, which can be used for resuming or evaluating the model.

### Project Workflow

1. **Data Preparation**: Use `data_loader.py` to load and preprocess your data.
2. **Model Training**: Run `main.py` to train the MTFAE model.
3. **Evaluation**: Utilize saved checkpoints to evaluate model performance or fine-tune for different tasks.

## Results

The MTFAE model aims to improve feature representation for multiple tasks. Model checkpoints, logs, and metrics are saved during training and can be used to assess the model's performance.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact

For any questions or issues, please open an issue on the repository or contact the project maintainer.

---

Feel free to contribute or suggest improvements to the project. We appreciate your feedback!

