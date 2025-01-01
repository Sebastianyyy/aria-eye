import os

import numpy as np
import torch
from projectaria_tools.core import mps
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.projects.aea import AriaEverydayActivitiesDataProvider
from projectaria_tools.utils.calibration_utils import \
    undistort_image_and_calibration
from torch.utils.data import Dataset

from .config import get_transformations


class AriaDataset(Dataset):
    def __init__(self, config, train=True, preprocess=False):
        """
        Dataset for Aria Everyday Activities Dataset.

        Args:
            config (dict): Configuration dictionary containing dataset path and transformations.
            train (bool): Whether the dataset is for training or testing.
            preprocess (bool): Whether to preprocess the data and save it to disk.
        """
        self.root = config["train_data"] if train else config["test_data"]
        self.processed_data_dir = config["preprocess_data_path"]
        self.sample = config["sample"]
        self.frame_grabber = config["frame_grabber"]
        self.rgb_stream_id = StreamId("214-1")
        self.rgb_stream_label = "camera-rgb"

        train_transform, test_transform = get_transformations(config)
        self.transform = train_transform if train else test_transform

        if not os.path.exists(self.processed_data_dir):
            os.makedirs(self.processed_data_dir)

        processed_file = os.path.join(
            self.processed_data_dir,
            f"{'train' if train else 'test'}_sample{self.sample}_frames{self.frame_grabber}.pt",
        )

        if preprocess or not os.path.exists(processed_file):
            self.data = self._process_and_save(processed_file)
        else:
            self.data = torch.load(processed_file)

    def _process_and_save(self, processed_file):
        """Process raw video files and save to disk."""
        all_files = [
            f"{self.root}/{f}" for f in os.listdir(self.root) if f.startswith("loc")
        ]
        data = []

        for file in all_files:
            video_timestamps = self._load_video(file)
            for encoded_timestamps in video_timestamps:
                images, eye_gazes = [], []
                for enc_timestamp in encoded_timestamps:
                    path, timestamp = enc_timestamp.split("@")
                    data_provider = AriaEverydayActivitiesDataProvider(path)
                    device_time_ns = int(timestamp)
                    device_calibration = data_provider.vrs.get_device_calibration()

                    rgb_camera_calibration = device_calibration.get_camera_calib(
                        self.rgb_stream_label
                    )
                    image = data_provider.vrs.get_image_data_by_time_ns(
                        self.rgb_stream_id,
                        device_time_ns,
                        TimeDomain.DEVICE_TIME,
                        TimeQueryOptions.BEFORE,
                    )
                    image, undistored_calib = undistort_image_and_calibration(
                        image[0].to_numpy_array(), rgb_camera_calibration
                    )
                    eye_gaze = data_provider.mps.get_general_eyegaze(
                        device_time_ns, TimeQueryOptions.CLOSEST
                    )
                    eye_gazes_projected = self._projection(
                        eye_gaze, undistored_calib, device_calibration
                    )
                    eye_gaze = self._norm_eye_gaze(eye_gazes_projected, image.shape[0])
                    eye_gaze[0], eye_gaze[1] = 1 - eye_gaze[1], eye_gaze[0]
                    eye_gazes.append(eye_gaze)
                    image = self.transform(image)
                    images.append(image)

                images_tensor = torch.stack(images) if len(images) > 1 else torch.tensor(images[0])
                eye_gazes_tensor = torch.tensor(eye_gazes, dtype=torch.float32)
                data.append((images_tensor, eye_gazes_tensor))

        torch.save(data, processed_file)
        return data

    def _load_video(self, file):
        """Load video frames from vrs files and return list of encoded path@timestamps."""
        timestamps = []
        data_provider = AriaEverydayActivitiesDataProvider(file)
        timecode_vec = data_provider.vrs.get_timestamps_ns(
            self.rgb_stream_id, TimeDomain.DEVICE_TIME
        )
        frame = 0
        temp_t = []
        for i, t in enumerate(timecode_vec):
            if i % self.sample == 0:
                if frame < self.frame_grabber - 1:
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
        """Return number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, index):
        """Return preprocessed tensors for a given index."""
        return self.data[index]

    def _norm_eye_gaze(self, eye_gaze, scale):
        return np.array([eye_gaze[0] / scale, eye_gaze[1] / scale])

    def _projection(self, eye_gaze, rgb_camera_calibration, device_calibration):
        gaze_vector_in_cpf = mps.get_eyegaze_point_at_depth(
            eye_gaze.yaw, eye_gaze.pitch, 1.0
        )
        T_device_CPF = device_calibration.get_transform_device_cpf()
        gaze_center_in_camera = (
            rgb_camera_calibration.get_transform_device_camera().inverse()
            @ T_device_CPF
            @ gaze_vector_in_cpf
        )
        gaze_projection = rgb_camera_calibration.project(gaze_center_in_camera)
        return gaze_projection
