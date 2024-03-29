import torch
import torch.nn as nn
import torch.nn.functional as F

class DCQN(nn.Module):
    def __init__(self, n_actions):
        super(DCQN, self).__init__()
        
        # Adjust the number of channels in convolutional layers
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=8, stride=4)  # Reduced from 128 to 64
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2)  # Reduced from 256 to 128
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=2, stride=1)  # Reduced from 512 to 256
        self.bn3 = nn.BatchNorm2d(256)

        # Dummy input to determine fully connected layer size
        dummy_input = torch.zeros(1, 4, 84, 84)
        output_size = self._get_conv_output(dummy_input)

        # Adjust the number of neurons in fully connected layers
        self.fc = nn.Linear(output_size, 2048)  # Significantly reduced to manage parameter count
        self.fc2 = nn.Linear(2048, 1024)  # Adjusted accordingly
        self.fc3 = nn.Linear(1024, 512)  # Further adjustment
        self.out = nn.Linear(512, n_actions)  # Final output layer

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
        if x.dim() == 5:
            x = x.squeeze(2)
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)
        return x








        





import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm

class GatedTransformerXLLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(GatedTransformerXLLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

        # Gating mechanism: two gates for the self-attention and feed-forward
        self.gate1 = nn.Linear(d_model * 2, d_model)
        self.gate2 = nn.Linear(d_model * 2, d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention sublayer with gating
        src2 = self.norm1(src)
        q = k = v = src2
        src2, _ = self.self_attn(q, k, v, attn_mask=src_mask,
                                 key_padding_mask=src_key_padding_mask)
        gate1_input = torch.cat([src, src2], dim=-1)
        gate1_output = self.gate1(gate1_input)
        src = src + self.dropout1(gate1_output)

        # Feed-forward sublayer with gating
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        gate2_input = torch.cat([src, src2], dim=-1)
        gate2_output = self.gate2(gate2_input)
        src = src + self.dropout2(gate2_output)

        return self.norm3(src)

class GatedTransformerXL(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(GatedTransformerXL, self).__init__()
        self.layers = nn.ModuleList([
            GatedTransformerXLLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        return output
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, embed_size, num_patches):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embed_size = embed_size
        self.num_patches = num_patches
        # Since channel is always 1, we remove it from the calculation
        self.projection = nn.Linear(patch_size * patch_size, embed_size)

    def forward(self, x):
        # Assuming x: [batch_size, num_stacked_frames, height, width], removing channels
        batch_size, nf, h, w = x.size()
        # Create patches
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(batch_size, nf, self.num_patches, -1)  # Flatten patches
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size * self.num_patches * nf, -1)
        # Project patches to embeddings
        x = self.projection(x)
        x = x.view(batch_size, self.num_patches, nf, self.embed_size).permute(0, 2, 1, 3)
        return x
    


class DTQN(nn.Module):
    def __init__(self, num_actions, embed_size=528, num_heads=8, num_layers=1, fc1_out=512, fc2_out=128, fc_intermediate=256, num_stacked_frames=4, patch_size=6):
        super(DTQN, self).__init__()
        self.embed_size = embed_size
        self.num_stacked_frames = num_stacked_frames
        self.patch_size = patch_size
        self.frame_input_dim = 84 * 84  # For one frame
        self.total_input_dim = self.frame_input_dim * num_stacked_frames  # For all stacked frames
        self.num_patches = (84 // patch_size) ** 2
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(patch_size, embed_size, self.num_patches)

        # Positional Embeddings
        self.positional_embeddings = nn.Parameter(torch.randn(1, num_stacked_frames*self.num_patches, embed_size))

        # Transformer Encoder Layer
        self.transformer_encoder = GatedTransformerXL(
            d_model=embed_size, nhead=num_heads, num_layers=num_layers
        )

        # Fully connected layers
        self.fc1 = nn.Linear(embed_size * num_stacked_frames * self.num_patches, fc1_out)
        self.fc2 = nn.Linear(fc1_out, fc2_out)
        self.fc3 = nn.Linear(fc2_out, fc_intermediate)
        self.fc_out = nn.Linear(fc_intermediate, num_actions)

    def forward(self, x):
        # Check and adjust for input dimensions
        if x.dim() == 5:
            x = x.squeeze(2)

        # x: [batch_size, num_stacked_frames, channels, height, width]
        
        # Embed patches
        x = self.patch_embedding(x)
        
        x = x.reshape(x.size(0), -1, self.embed_size)
        # Add positional embeddings
        

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)

        # Fully connected layers
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
n_actions = 6  # Example value, replace with the actual number of actions for your specific environment

dcqn_model = DCQN(n_actions=n_actions)

# For DTQN
# Assuming 'num_actions' matches the 'n_actions' used in DCQN and other parameters are as per your DTQN definition

# Utilize the check_param_size function for instantiated models
# Assuming your DCQN and DTQN classes are defined as previously mentioned

# Correct instantiation of the models
dcqn_model = DCQN( n_actions=10)  # Example values, adjust as needed
dtqn_model = DTQN(num_actions=6, embed_size=512, num_heads=16, num_layers=3,patch_size=16)  # Example values, adjust as needed

# Now pass these instances to check_param_size, not the class names
dcqn_param_size = check_param_size(dcqn_model)
dtqn_param_size = check_param_size(dtqn_model)

print("DCQN Param Size:", dcqn_param_size)
print("DTQN Param Size:", dtqn_param_size)




                 
