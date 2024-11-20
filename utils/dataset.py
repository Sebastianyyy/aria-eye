import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from .config import get_transformations

class SampleDataset(Dataset):
    """
    Custom Dataset class for the CIFAR-10 dataset.

    Args:
    - config (dict): Configuration dictionary containing dataset path and transformations.
    - train (bool): If True, loads the training dataset. If False, loads the test dataset.
    
    Attributes:
    - image_paths (list): List of image file paths.
    - labels (list): List of corresponding class labels.
    - transform (callable): Transformation function to apply to images.
    """
    def __init__(self, config, train=True):
        self.root_dir = config["dataset_path"]
        self.train = train
        train_transform, test_transform = get_transformations(config)
        self.transform = train_transform if self.train else test_transform

        # Determine subdirectory based on train/test mode
        sub_dir = "train" if self.train else "test"

        # Initialize image paths and labels
        self.image_paths = []
        self.labels = []

        # Path to the specific train/test subdirectory
        self.data_dir = os.path.join(self.root_dir, sub_dir)

        # Loop through each class directory
        for label, class_name in enumerate(os.listdir(self.data_dir)):
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_dir):
                # Add all image paths and their corresponding labels
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves an image and its label at the given index.
        
        Parameters:
        - idx (int): Index of the sample to fetch.

        Returns:
        - (torch.Tensor, int): Transformed image tensor and corresponding label.
        """
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # Ensure RGB mode
        
        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.labels[idx]
        
        return image, label