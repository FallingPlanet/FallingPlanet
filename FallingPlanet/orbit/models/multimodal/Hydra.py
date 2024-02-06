import torch
import torch.nn as nn
from transformers import ViTModel, ViTFeatureExtractor, DeiTForImageClassification
from AuTransformer import FPATF_Tiny
from BertFineTuneForSequenceClassification import BertFineTuneTiny

class HydraTiny(nn.Module):
    def __init__(self, num_classes, vision_model_name, text_model_config, audio_model_config):
        super(HydraTiny, self).__init__()
        
        # Assuming the vision model is a Vision Transformer
        self.vision_model = ViTModel.from_pretrained(vision_model_name)
        
        
        # Text model placeholder initialization (adjust with actual model)
        self.text_model = ...  # Initialize with text_model_config
        
        # Audio model (FPATF_Tiny) initialization
        self.audio_model = FPATF_Tiny(**audio_model_config)
        
        
        # Fusion mechanism (adjust dimensions as needed)
        self.fusion_layer = nn.Linear(128+128+256, num_classes)  # Update based on your fusion strategy
        f_transformer_layer_1 = nn.TransformerEncoderLayer(d_model = 512,nhead=8, dim_feedforward=1024, dropout = .01)
        self.f_transformer_1 = nn.TransformerEncoder(f_transformer_layer_1, num_layers = 6)
        self.f_layer_norm = nn.layer_norm = nn.LayerNorm(512)
        self.f_fc = nn.Linear(512,256)
        f_transformer_layer_2 = nn.TransformerEncoderLayer(d_model = 256, nhead=4, dim_feedforward = 512, dropout = .01)
        self.f_transformer_2 = nn.TransformerEncoder(f_transformer_layer_2, num_layers = 2)
        self.f_fc2 = nn.Linear(256,128)
        self.f_fc3 = nn.Linear(128, num_classes)

        
        self.num_classes = num_classes

    def forward(self, vision_inputs, text_input, audio_input):
        # Process vision inputs (a list of frames)
        
        vision_logits_list = [self.vision_model(pixel_values=features).logits for features in vision_inputs]
        
        # Average the logits from the vision model frames
        vision_logits = torch.mean(torch.stack(vision_logits_list), dim=0)
        
        # Process text input
        # Assuming text_model has a method .forward() or equivalent that takes text_input
        text_logits = self.text_model(text_input)
        
        # Process audio input (MFCCs)
        # Assuming audio_model takes MFCCs directly
        audio_logits = self.audio_model(audio_input)
        
        # Fusion of model outputs
        # Example concatenates logits; modify based on actual output shapes and fusion strategy
        combined_features = torch.cat((vision_logits, text_logits, audio_logits), dim=-1)
        final_output = self.fusion_layer(combined_features)
        
        return final_output


    