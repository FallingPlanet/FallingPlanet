import torch
import torch.nn as nn
import torch.nn.functional as F

class FPATF_Tiny(nn.Module):
    def __init__(self, target_channels, num_classes, num_heads=8, num_layers=2, dim_feedforward=1024, dropout=0.1, return_features=False):
        super().__init__()
        self.return_features = return_features  # Whether to return features before the final classification layer

        # Convolutional layers for MFCC features
        self.conv1 = nn.Conv1d(in_channels=target_channels, out_channels=256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2, 2)

        # Transformer Encoder Layers
        transformer_layer = nn.TransformerEncoderLayer(d_model=512, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers)

        # Additional layers
        self.layer_norm = nn.LayerNorm(512)
        self.fc1 = nn.Linear(512, 128)
        # The final classification layer is now optional
        self.fc2 = nn.Linear(128, num_classes) if not return_features else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input processing remains the same
        if x.dim() == 4:
            if x.size(1) in [1, 2]:
                x = x.mean(dim=1)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        if x.dim() != 3:
            raise ValueError(f"Input tensor should be 3D after processing, got shape: {x.shape}")

        # Feature extraction remains unchanged
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.permute(2, 0, 1)  # Adjust as needed for your model
        x = self.transformer_encoder(x)
        output = x[-1]

        output = self.layer_norm(output)
        output = self.dropout(output)
        output = F.relu(self.fc1(output))

        # Modify the behavior based on `return_features`
        if self.return_features:
            return output  # Return features before the final classification layer
        else:
            output = self.fc2(output)  # Apply the final classification layer
            return output

