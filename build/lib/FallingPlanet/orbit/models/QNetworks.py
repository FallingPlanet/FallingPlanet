import torch
import torch.nn as nn
import torch.nn.functional as F

class DCQN(nn.Module):
    def __init__(self, n_observation, n_actions):
        super(DCQN, self).__init__()
        # Increase the number of channels in each layer to double the original capacity
        self.conv1 = nn.Conv2d(in_channels=n_observation, out_channels=32*2, kernel_size=8, stride=4)  # 64 channels
        self.conv2 = nn.Conv2d(32*2, 64*2, kernel_size=4, stride=2)  # 128 channels
        self.conv3 = nn.Conv2d(64*2, 64*2, kernel_size=3, stride=1)  # 128 channels

        # Use a dummy input to pass through the conv layers to calculate the output size
        dummy_input = torch.zeros(1, n_observation, 84, 84)
        output_size = self._get_conv_output(dummy_input)

        # Increase the size of the fully connected layers accordingly
        self.fc = nn.Linear(output_size, 512*2)  # Increased the number of neurons in the fc layer to 1024
        self.fc2 = nn.Linear(512*2,512)
        self.fc3 = nn.Linear(512,256)
        self.out = nn.Linear(256, n_actions)

    def _get_conv_output(self, shape):
        with torch.no_grad():
            # Pass the dummy input through the conv layers
            output = self._forward_features(shape)
            # Flatten the output and get the total number of features
            return int(torch.numel(output) / output.size(0))

    def _forward_features(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        x = self.out(x)
        return x




        







class DTQN(nn.Module):
    def __init__(self,num_actions, embed_size=512, num_heads=8, num_layers=6 ):
        super(DTQN, self).__init__()
        self.embedding = nn.Linear(4*84*84, embed_size)  # Adjust for actual input dimensions
        self.transformer_encoder = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, batch_first=True),
    num_layers=num_layers
)

        self.fc_out = nn.Linear(embed_size, num_actions)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten image input
        x = self.embedding(x)
        x = self.transformer_encoder(x.unsqueeze(1))
        x = x.squeeze(1)
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

dcqn_model = DCQN(n_observation=n_observation, n_actions=n_actions)

# For DTQN
# Assuming 'num_actions' matches the 'n_actions' used in DCQN and other parameters are as per your DTQN definition
embed_size = 512
num_heads = 8
num_layers = 6
dtqn_model = DTQN(num_actions=n_actions, embed_size=embed_size, num_heads=num_heads, num_layers=num_layers)

# Utilize the check_param_size function for instantiated models
# Assuming your DCQN and DTQN classes are defined as previously mentioned

# Correct instantiation of the models
dcqn_model = DCQN(n_observation=4, n_actions=10)  # Example values, adjust as needed
dtqn_model = DTQN(num_actions=10, embed_size=256, num_heads=8, num_layers=2)  # Example values, adjust as needed

# Now pass these instances to check_param_size, not the class names
dcqn_param_size = check_param_size(dcqn_model)
dtqn_param_size = check_param_size(dtqn_model)

print("DCQN Param Size:", dcqn_param_size)
print("DTQN Param Size:", dtqn_param_size)



                 
