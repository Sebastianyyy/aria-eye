import os
import sys
from PIL import Image
from torch.utils.data import Dataset
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import StreamId
import numpy as np
from projectaria_tools.projects.aea import (
    AriaEverydayActivitiesDataProvider)
from projectaria_tools.utils.calibration_utils import undistort_image_and_calibration
from matplotlib import pyplot as plt
from projectaria_tools.core import mps
import torchvision.transforms.v2 as T
import torch
from .config import get_transformations, get_config

class AriaDataset(Dataset):
    def __init__(self, config, train=True):
        """Dataset for Aria Everyday Activities Dataset

        Args:
            root (str): path to folder containing the dataset
            config (dict): Configuration dictionary containing dataset path and transformations.
            sample (int, optional): Sampling over video . Defaults to 10.
            frame_grabber (int, optional): Number of consecutive frames to grab. Defaults to 3.
        """
        self.root = config["train_data"] if train else config["test_data"]
        self.all_files = [f"{self.root}/{f}" for f in os.listdir(self.root) if f.startswith('loc') ]
        self.rgb_stream_id = StreamId("214-1")
        self.sample = config["sample"]
        self.frame_grabber = config["frame_grabber"] - 1
        self.timestamps = []
        self.rgb_stream_label = "camera-rgb"
        train_transform, test_transform = get_transformations(config)
        self.transform = train_transform if train else test_transform
        for file in self.all_files:
            self.timestamps.extend(self._load_video(file))


    def _load_video(self,file):
        """Load video frames from vrs files and return list of encoded path@timestamps
        Args:
            file (str): path to vrs file

        Returns:
            list: list of encoded path@timestamps
        """
        timestamps = []
        data_provider = AriaEverydayActivitiesDataProvider(file)
        timecode_vec = data_provider.vrs.get_timestamps_ns(
            self.rgb_stream_id, TimeDomain.DEVICE_TIME)
        # print(timecode_vec)
        frame = 0
        temp_t = []
        for i, t in enumerate(timecode_vec):
            if i % self.sample == 0:
                if frame < self.frame_grabber:
                    frame += 1
                    name = f"{file}@{t}"
                    temp_t.append(name)
                else:
                    name = f"{file}@{t}"
                    temp_t.append(name)
                    frame = 0
                    timestamps.append(temp_t)
                    temp_t = []
        return timestamps

    def __len__(self):
        """Return number of samples in the dataset

        Returns:
            int: number of samples in the dataset
        """
        return len(self.timestamps)
        

    def __getitem__(self, index):
        """Return images and eye gazes for a given index

        Args:
            index (int): index of the sample

        Returns:
            tuple: Tuple of images and eye gazes
        """
        encoded_timestamps=self.timestamps[index]
        images,eye_gazes=[],[]
        for enc_timestamp in encoded_timestamps:
            # decode the path and timestamp
            path, timestamp = enc_timestamp.split('@')
            # read the vrs file
            data_provider = AriaEverydayActivitiesDataProvider(path)
            # get timestamp in ns
            device_time_ns = int(timestamp)
            # print(self.rgb_stream_label)

            # get device calibration and rgb calibration
            device_calibration = data_provider.vrs.get_device_calibration()

            rgb_camera_calibration = device_calibration.get_camera_calib(
                self.rgb_stream_label)

            # get image data
            image = data_provider.vrs.get_image_data_by_time_ns(
                self.rgb_stream_id, device_time_ns, TimeDomain.DEVICE_TIME, TimeQueryOptions.BEFORE)

            # undistort the image and get camera parameters after undistortion
            image, undistored_calib = undistort_image_and_calibration(
                image[0].to_numpy_array(), rgb_camera_calibration)

            # get eye gaze
            eye_gaze = data_provider.mps.get_general_eyegaze(
                device_time_ns, TimeQueryOptions.CLOSEST)

            # project the eye gaze to the undistorted image
            eye_gazes_projected = self._projection(
                eye_gaze, undistored_calib, device_calibration)
            
            # normalize the eye gaze so every gaze is between 0 and 1
            eye_gaze = self._norm_eye_gaze(eye_gazes_projected, image.shape[0])
            eye_gaze[0],eye_gaze[1]=1-eye_gaze[1],eye_gaze[0]
            eye_gazes.append(eye_gaze)
            # apply transform to the image
            image=self.transform(image)
            images.append(image)
            
        images_tensor = torch.stack(images)
        eye_gazes_tensor = torch.tensor(eye_gazes, dtype=torch.float32)

            
        return images_tensor,eye_gazes_tensor
    
    def _norm_eye_gaze(self, eye_gaze, scale):
        return np.array([eye_gaze[0]/scale, eye_gaze[1]/scale])

    def _projection(self, eye_gaze, rgb_camera_calibration, device_calibration):
        """Project the eye gaze to the image

        Args:
            eye_gaze: eye gaze data object
            rgb_camera_calibration: rgb camera calibration object
            device_calibration: device calibration object

        Returns:
            List: List of projected eye gaze
        """
        # Get the gaze vector in the camera coordinate system
        gaze_vector_in_cpf = mps.get_eyegaze_point_at_depth(
            eye_gaze.yaw, eye_gaze.pitch, 1.)
        T_device_CPF = device_calibration.get_transform_device_cpf()
        gaze_center_in_camera = (
            rgb_camera_calibration.get_transform_device_camera().inverse()
            @ T_device_CPF
            @ gaze_vector_in_cpf
        )

        gaze_projection = rgb_camera_calibration.project(gaze_center_in_camera)

        return gaze_projection
