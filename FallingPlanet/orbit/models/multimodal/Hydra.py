import torch
import torch.nn as nn
import transformers
from models.AuTransformer import FPATF_Tiny

class HydraTiny(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
    