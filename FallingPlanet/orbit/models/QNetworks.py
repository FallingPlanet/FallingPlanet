import torch
import torch.nn as nn
import torch.nn.functional as F

class DCQN(nn.Module):
    def __init__(self, n_actions):
        super(DCQN, self).__init__()
        # Here, in_channels is set to 4 explicitly to handle 4 stacked frames
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Using a dummy input to determine the size of the fully connected layer
        dummy_input = torch.zeros(1, 4, 84, 84)
        output_size = self._get_conv_output(dummy_input)

        self.fc = nn.Linear(output_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 256)
        self.out = nn.Linear(256, n_actions)

    def _get_conv_output(self, shape):
        with torch.no_grad():
            output = self._forward_features(shape)
            return int(torch.numel(output) / output.size(0))

    def _forward_features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)
        return x







        






class DTQN(nn.Module):
    def __init__(self, num_actions, embed_size=528, num_heads=8, num_layers=1, fc1_out=512, fc2_out=128, fc_intermediate=256, num_stacked_frames=4):
        super(DTQN, self).__init__()
        self.embed_size = embed_size
        self.num_stacked_frames = num_stacked_frames
        # Assume each frame is flattened to a 84x84 grayscale image for simplicity in calculation
        self.frame_input_dim = 84 * 84  # This is for one frame
        self.total_input_dim = self.frame_input_dim * num_stacked_frames  # Total input dimension for stacked frames

        # Dimensionality reduction layer to match the embed_size
        self.dim_reduction = nn.Linear(self.total_input_dim, embed_size * num_stacked_frames)
        
        # Positional Embeddings added after dimensionality reduction
        self.positional_embeddings = nn.Parameter(torch.zeros(1, num_stacked_frames, embed_size))
        
        # Transformer Encoder Layer
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, batch_first=True),
            num_layers=num_layers
        )

        # Assuming the transformer output is flattened before passing to the fully connected layers
        self.fc1 = nn.Linear(embed_size * num_stacked_frames, fc1_out)
        self.fc2 = nn.Linear(fc1_out, fc2_out)
        self.fc3 = nn.Linear(fc2_out, fc_intermediate)
        self.fc_out = nn.Linear(fc_intermediate, num_actions)

    def forward(self, x):
        
        x = x.view(x.size(0), -1)  # Flatten input
        
        
        x = self.dim_reduction(x)
       
        
        x = x.view(-1, self.num_stacked_frames, self.embed_size)
        
        
        x += self.positional_embeddings
        
        
        x = self.transformer_encoder(x)
        
        
        x = x.view(x.size(0), -1)
        
        
        x = F.relu(self.fc1(x))
        
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        output = self.fc_out(x)
        
        return output





# Checking parameter size
def check_param_size(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params
# Example instantiation, adjust arguments as per your actual use case and class definitions

# For DCQN
# Assuming 'n_observation' is the number of channels in the input image (e.g., 4 for stacked frames) 
# and 'n_actions' is the number of possible actions in your environment
n_observation = 4
n_actions = 10  # Example value, replace with the actual number of actions for your specific environment

dcqn_model = DCQN(n_actions=n_actions)

# For DTQN
# Assuming 'num_actions' matches the 'n_actions' used in DCQN and other parameters are as per your DTQN definition

# Utilize the check_param_size function for instantiated models
# Assuming your DCQN and DTQN classes are defined as previously mentioned

# Correct instantiation of the models
dcqn_model = DCQN( n_actions=10)  # Example values, adjust as needed
dtqn_model = DTQN(num_actions=1, embed_size=256, num_heads=8, num_layers=4)  # Example values, adjust as needed

# Now pass these instances to check_param_size, not the class names
dcqn_param_size = check_param_size(dcqn_model)
dtqn_param_size = check_param_size(dtqn_model)

print("DCQN Param Size:", dcqn_param_size)
print("DTQN Param Size:", dtqn_param_size)




                 
