import torch
import torch.nn as nn
from transformers import ViTModel, ViTFeatureExtractor, DeiTForImageClassification, BertModel, DeiTModel
from AuTransformer import FPATF_Tiny
from BertFineTuneForSequenceClassification import BertFineTuneTiny
from FallingPlanet.orbit.utils import Tokenizers
import torch.nn.functional as F

class HydraTiny(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(HydraTiny, self).__init__()
        text_dict = kwargs.get('text_model_dict',False)
        vision_dict = kwargs.get('vision_model_dict',False)
        audio_dict = kwargs.get('audio_model_dict',False)
        requires_grad = kwargs.get('requires_grad',False)

        # Assuming the vision model is a Vision Transformer
        self.vision_model = ViTModel.from_pretrained('facebook/deit-tiny-patch16-224')
        v_classifier = nn.Sequential(
            nn.Linear(192, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256,num_classes)
            
        )
        
        
        # Text model placeholder initialization (adjust with actual model)
        self.text_model = BertModel.from_pretrained("prajjwal1/bert-tiny")
        
        # Audio model (FPATF_Tiny) initialization
        self.audio_model = FPATF_Tiny()
        if requires_grad == False:
            self.audio_model.parameters(requires_grad=False)
            self.vision_model.parameters(requires_grad=False)
            self.text_model.parameters(requires_grad=False)
            # Process vision inputs (a list of frames)
            self.audio_model.eval()
            self.text_model.eval()
            self.vision_model.eval()
        if text_dict != False:
            torch.load(text_dict)
        

        
        # Fusion mechanism (adjust dimensions as needed)
        self.fusion_layer = FusionLayer(unified_class_dim=num_classes)
        f_transformer_layer_1 = nn.TransformerEncoderLayer(d_model = 512,nhead=9, dim_feedforward=1024, dropout = .01)
        self.f_transformer_1 = nn.TransformerEncoder(f_transformer_layer_1, num_layers = 6)
        self.f_layer_norm = nn.layer_norm = nn.LayerNorm(512)
        self.f_fc = nn.Linear(512,256)
        f_transformer_layer_2 = nn.TransformerEncoderLayer(d_model = 256, nhead=9, dim_feedforward = 512, dropout = .01)
        self.f_transformer_2 = nn.TransformerEncoder(f_transformer_layer_2, num_layers = 2)
        self.f_fc2 = nn.Linear(256,128)
        self.f_fc3 = nn.Linear(128, num_classes)

        
        self.num_classes = num_classes

    def forward(self, vision_inputs, text_input, audio_input):
        
        
        
        vision_logits_list = [self.vision_model(pixel_values=features).logits for features in vision_inputs]
        
        # Average the logits from the vision model frames
        vision_logits = torch.mean(torch.stack(vision_logits_list), dim=0)
        
        input_ids, attention_masks = Tokenizers.BertTiny_tokenize(text_input, max_lenght = 256)
        # Assuming text_model has a method .forward() or equivalent that takes text_input
        text_logits = self.text_model(input_ids, attention_masks)
        
        
        # Process audio input (MFCCs)
        # Assuming audio_model takes MFCCs directly
        audio_logits = self.audio_model(audio_input)
        
        # Fusion of model outputs
        # Example concatenates logits; modify based on actual output shapes and fusion strategy
        fused_features = self.fusion_layer(vision_logits,text_logits,audio_logits)
        
        x = self.f_transformer_1(fused_features)
        
        return output

class FusionLayer(nn.Module):
    def __init__(self, input_dims, unified_class_dim):
        super(FusionLayer, self).__init__()
        
        self.projection_layer = nn.ModuleList(
            nn.Linear(dim, unified_class_dim) for dim in input_dims
        )
        
    def fuse_and_pad(self, *inputs):
        # Example padding with zeros to match the largest dimension (unified_class_dim)
        padded_inputs = [F.pad(input, (0, self.unified_dim - input.size(1)), "constnat", 0) for input in inputs]
        # Concatenate along the feature dimension
        fused = torch.cat(padded_inputs, dim=1)
        
        return fused

        