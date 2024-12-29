import torch
import torch.nn as nn
import torchvision.models as models


class AutoregressiveGazeWithHistory(nn.Module):
    def __init__(
        self,
        backbone="resnet18",
        seq_len=3,
        image_hidden_dim=256,
        gaze_hidden_dim=128,
        num_lstm_layers=2,
        num_of_class=2,
        device="cpu",
    ):
        super(AutoregressiveGazeWithHistory, self).__init__()

        self.image_hidden_dim = image_hidden_dim
        self.gaze_h_dim = gaze_hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.device = device
        self.seq_len = seq_len
        self.num_of_class = num_of_class

        # Feature extractor for images
        if backbone == "resnet18":
            self.feature_extractor = models.resnet18(pretrained=False)
            image_features = self.feature_extractor.fc.in_features
            self.feature_extractor.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # LSTM for image features
        self.image_lstm = nn.LSTM(
            input_size=image_features,
            hidden_size=image_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
        )

        # LSTM for gaze history
        self.gaze_lstm = nn.LSTM(
            input_size=num_of_class,  # x, y coordinates
            hidden_size=gaze_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
        )

        # Fully connected layers for gaze prediction
        self.fc = nn.Sequential(
            nn.Linear(image_hidden_dim + gaze_hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_of_class),  # Predict x, y coordinates
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, in_channels, height, width) - images
            y: Tensor of shape (batch_size, seq_len, num_of_classes) - gaze history
        Returns:
            gaze_predictions: Tensor of shape (batch_size, seq_len, num_of_classes)
        """
        batch_size, seq_len, c, h, w = x.size()
        assert (
            seq_len == self.seq_len
        ), "Input sequence length must match the model's seq_len"

        # Prepare LSTM hidden states
        gaze_predictions = []
        hidden_image = (
            torch.zeros(self.num_lstm_layers, batch_size, self.image_hidden_dim).to(
                self.device
            ),
            torch.zeros(self.num_lstm_layers, batch_size, self.image_hidden_dim).to(
                self.device
            ),
        )

        hidden_gaze = (
            torch.zeros(self.num_lstm_layers, batch_size, self.gaze_h_dim).to(
                self.device
            ),
            torch.zeros(self.num_lstm_layers, batch_size, self.gaze_h_dim).to(
                self.device
            ),
        )

        # Initialize with the first gaze ground truth
        current_gaze = (torch.zeros(batch_size, self.num_of_class) + 0.5).to(
            self.device
        )

        for t in range(seq_len):
            # Extract image features for current frame
            current_image = x[:, t, :, :, :]
            current_image_ft = self.feature_extractor(current_image)

            # Pass image features through LSTM
            current_image_ft = current_image_ft.unsqueeze(1)  # Add sequence dimension
            _, (hidden_image_h, hidden_image_c) = self.image_lstm(
                current_image_ft, hidden_image
            )
            image_hidden_state = hidden_image_h[-1]  # Take last layer's hidden state
            hidden_image = (hidden_image_h, hidden_image_c)  # Update for next iteration

            # Pass gaze history through LSTM
            current_gaze = current_gaze.unsqueeze(1)  # Add sequence dimension
            _, (hidden_gaze_h, hidden_gaze_c) = self.gaze_lstm(
                current_gaze, hidden_gaze
            )
            gaze_hidden_state = hidden_gaze_h[-1]  # Take last layer's hidden state
            hidden_gaze = (hidden_gaze_h, hidden_gaze_c)  # Update for next iteration

            # Concatenate image and gaze hidden states
            combined_features = torch.cat(
                (image_hidden_state, gaze_hidden_state), dim=1
            )

            # Predict gaze for the current frame
            predicted_gaze = self.fc(combined_features)
            gaze_predictions.append(predicted_gaze)

            # Autoregressive step: use predicted gaze as input for the next step
            current_gaze = predicted_gaze  
            
        # Stack predictions into a single tensor
        gaze_predictions = torch.stack(gaze_predictions, dim=1)
        return gaze_predictions


def get_model(config):
    return AutoregressiveGazeWithHistory(
        num_of_class=config["num_of_classes"],
        seq_len=config["frame_grabber"],
        device=config["device"],
    )
