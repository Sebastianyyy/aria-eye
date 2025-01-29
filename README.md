# Aria Eye-Tracking Project
## Contributors
Jakub Kościukiewicz, Paweł Małecki, Sebastian Musiał, Jakub Soborski, Filip Soszyński
## About the Project
This project was created for the **Pattern Recognition** class at **Jagiellonian University**. The main idea is to predict what a person is looking at in an image by:
- Using **Project Aria dataset** or other **eye-tracker vision data**.
- Extracting gaze information to focus attention using **various ResNet modifications** and **Transformer model**.

## Abstract
*Estimating eye gaze from egocentric camera footage could be a crucial task with applications in **AR, smart glasses, and behavioral analysis** from a first-person perspective. In this paper, we explore the novel task of estimating eye gaze from an egocentric level. To the best of our knowledge, we are the first to tackle this task using the **ARIA dataset**, which provides egocentric video captured from smart glasses. We evaluate our approach on this dataset and demonstrate its effectiveness in various real-world scenarios. Experimental results show that our model achieves **interesting results**.*

## Repository
Clone the repository:
```sh
 git clone https://github.com/Sebastianyyy/aria-eye.git
 cd aria-eye
```

## Environment Setup
Create the required environment using `environment.yaml`:
```sh
conda env create -f environment.yaml
conda activate aria-eye
```

## Data
### Downloading the Dataset
Firstly you need download a json with the dataset URLS from the original site: 
```sh
https://www.projectaria.com/datasets/aea/
```
and put it into
```sh
data/aria_everyday_activities_dataset_download_urls.json
```
Navigate to the data directory and execute the scripts to download training and test data:
```sh
cd ./data
./download_data_train.sh
./download_data_test.sh
```

### Configuration

The configuration for training and evaluation is managed using the utils.config module. This module provides predefined configurations for different model architectures and hyperparameters.

For **training**, the configuration is loaded using:
```py
from utils.config import get_config
config = get_config()
```
For **evaluation** and **video generation**, the configuration is loaded using:
```py
from utils.config import get_config_validation
config_validation = get_config_validation()
```

## Running the Project
### Training the Model
```sh
python train.py
```

### Evaluating the Model
```sh
python evaluate.py
```

### Generating Video Output
```sh
python generate_video.py
```
