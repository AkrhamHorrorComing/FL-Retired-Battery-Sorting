import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        A simpler autoencoder designed for OOD detection.
        - Simplified network structure
        - Removed BatchNorm and Dropout to reduce generalization ability
        - The bottleneck layer (hidden_dim) should be set to a small value
        """
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),  # Significantly reduced hidden layer dimension
            nn.ReLU(inplace=True),  # Can use simple ReLU or LeakyReLU
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, hidden_dim)  # Narrow bottleneck layer
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, input_dim)  # Output layer typically has no activation function
        )

        # Good weight initialization can still be used
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
