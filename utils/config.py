from pathlib import Path

import torch
from torchvision import transforms as T


def get_config():
    """
    Returns a dictionary containing the configuration for the experiment.

    Returns:
        dict: Configuration parameters including model settings, dataset details,
            optimizer, scheduler, and other hyperparameters.
    """
    return {
        "task": "regression",  # (regression, classification)
        "clip": False,  # clip predicted value 0-1 (only test data)
        "batch_size": 64,  # Number of samples per batch
        "num_epochs": 400,  # Total number of epochs for training
        "input_image_size": 224,  # Input image dimensions (square: width = height)
        "optimizer": "AdamW",  # Type of optimizer to use (e.g., Adam, SGD, AdamW)
        "lr": 0.00001,  # Learning rate for the optimizer
        "weight_decay": 0.00001,  # Regularization term to prevent overfitting
        "scheduler": "CosineAnnealingLR",  # Type of learning rate scheduler
        "scheduler_t_max": 10,  # Num of epochs over which to decay the learning rate for scheduler
        "scheduler_eta_min": 0.0001,  # Minimum learning rate value for the scheduler
        "loss_fn": "weighted_mse",  # Loss function to use (e.g., mse, weighted_mse, kl_loss)
        "seed": 42,  # Random seed for reproducibility of experiments
        "num_workers": -1,  # Number of workers for DataLoader (-1 = use all available CPU cores)
        "train_data": "data/downloads_train",  # Path to the train dataset
        "test_data": "data/downloads_test",  # Path to the test dataset
        "sample": 10,  # Sampling over video
        "frame_grabber": 3,  # Number of consecutive frames to grab
        "model_name": "resnet_optical_flow",  # Name of the model architecture to use
        "model_name_log": "resnet_optical_flow_weighted_mse",  # Name of the model log file
        "model_basename": "model_",  # Base name for saving and loading model weight files
        "preload": "latest",  # Preload setting to load weights: "latest", "none", or specific point
        "dataset_path": "data/test_data",  # Path to the dataset directory
        "device": "cuda:0",  # Device to use for training and evaluation (cuda:0 or cpu)
        "preprocess_data_path": "data/preprocess_data",  # Path to data ready to use in training
        # RESNET PARAMS
        "model_depth": 18,  # Depth of resnet.py
        "num_of_classes": 2,  # If not classification task set to 2
        # CLASSIFICATION
        "shape": 100,  # grid size 
        # "d_model": 32,  # Depth of transformer model
        # "heads": 4,  # Number of heads in the transformer model
        # "enc_depth": 2,  # Depth of the encoder in the transformer model
        # "dec_depth": 2,  # Depth of the decoder in the transformer model
    }


def get_transformations(config):
    """
    Generates data transformations for training and testing datasets.

    Parameters:
    - config (dict): A dictionary containing configuration values,
        specifically the key 'input_image_size' which defines the target size
        for resizing images.

    Returns:
    - tuple: A tuple containing:
        1. train_transform: Transformations to apply to training images.
        2. test_transform: Transformations to apply to testing images.
    """
    train_transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.Lambda(lambda x: torch.rot90(x, k=-1, dims=(1, 2))),
        ]
    )

    test_transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.Lambda(lambda x: torch.rot90(x, k=-1, dims=(1, 2))),
        ]
    )

    return train_transform, test_transform


def get_model_folder_path(config):
    """
    Generate the folder path for saving model logs.

    Args:
        config (dict): Configuration dictionary containing 'datasource'
                    and 'model_name_log' keys.

    Returns:
        str: A string representing the path to the model folder.
    """
    # Create a folder path using logs and model_name from the config dictionary
    return f"logs/{config['model_name_log']}"


def get_weights_file_path(config, epoch: str):
    """
    Constructs the path to a specific weights file based on the given epoch.

    Args:
        config (dict): Configuration dictionary containing experiment parameters.
        epoch (str): Epoch identifier (e.g., "latest" or a specific epoch number).

    Returns:
        str: Full path to the weights file for the specified epoch.
    """
    # Path to the model folder based on the log model name
    model_folder = get_model_folder_path(config)
    # Filename for the weights file, using the base name and epoch identifier
    model_filename = f"{config['model_basename']}{epoch}.pt"
    # Combine folder and filename to create the full path
    return str(Path(".") / model_folder / model_filename)


def latest_weights_file_path(config):
    """
    Finds the latest weights file in the weights folder.

    Args:
        config (dict): Configuration dictionary containing experiment parameters.

    Returns:
        str or None: Path to the most recent weights file. Returns None if no file is found.
    """
    # Path to the model folder based on the log model name
    model_folder = get_model_folder_path(config)
    # Glob pattern to match weight files in the folder
    model_filename = f"{config['model_basename']}*"
    # Find all matching files
    weights_files = list(Path(model_folder).glob(model_filename))
    # If no files are found, return None
    if len(weights_files) == 0:
        return None
    # Sort the files and return the latest one
    weights_files.sort()
    return str(weights_files[-1])

def get_config_validation():
    """
    Returns a dictionary containing the configuration for the validation process.

    Returns:
        dict: Configuration parameters including model settings and dataset details.
    """
    return {
        # Task and data handling
        "task": "classification",  # Task type (classification or regression)
        "clip": False,  # Clip predicted values between 0 and 1 (only for validation)
        "batch_size": 64,  # Number of samples per batch
        "input_image_size": 224,  # Input image dimensions
        "seed": 42,  # Random seed for reproducibility
        "num_workers": -1,  # Number of workers for DataLoader (-1 = use all available CPU cores)
        "test_data": "data/visualize_video_train",  # Path to the validation dataset
        "dataset_path": "data/visualize_video_train",  # Path to the dataset directory
        "preprocess_data_path": "data/preprocess_video_train",  # Path to preprocessed data
        "sample": 1,  # Sampling over video
        "frame_grabber": 3,  # Number of consecutive frames to grab

        
        # Model parameters
        "model_name": "resnet_optical_flow",  # Name of the model architecture
        "model_name_log": "resnet_optical_flow_kl_loss",  # Name for log files
        "model_basename": "model_",  # Base name for saving/loading model weights
        "preload": "latest",  # Preload option for loading weights
        "device": "cuda:0",  # Device to use for validation (cuda or cpu)
        "model_depth": 18,  # Depth of ResNet model
        "num_of_classes": 10000,  # Number of output classes
        "loss_fn": "kl_loss",  # Loss function to use (e.g., mse, weighted_mse, kl_loss)

      
        # Classification-specific parameters
        "shape": 100,  # Grid size for classification task
        
        # Visualization-specific parameters (for utils.visualize functions)
        "visualize_soft": True,  # Enable or disable soft visualization
        "visualize_hard": False,  # Enable or disable hard visualization

        # Video generation specific
        "video_format": "mp4v", # Use 'mp4v' for MP4 or 'XVID' for AVI
        "FPS": 16,
        "ground_truth_color": (0, 255, 0),  # GT : Green 
        "predicted_color": (0, 255, 255), # Pred : Yellow
        "generate_heatmap": True,
    }
