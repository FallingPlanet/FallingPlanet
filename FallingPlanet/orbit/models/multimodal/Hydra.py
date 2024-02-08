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
        f_transformer_layer_1 = nn.TransformerEncoderLayer(d_model = 512,nhead=8, dim_feedforward=1024, dropout = .01)
        self.f_transformer_1 = nn.TransformerEncoder(f_transformer_layer_1, num_layers = 6)
        self.f_layer_norm = nn.layer_norm = nn.LayerNorm(512)
        self.f_fc = nn.Linear(512,256)
        f_transformer_layer_2 = nn.TransformerEncoderLayer(d_model = 256, nhead=8, dim_feedforward = 512, dropout = .01)
        self.f_transformer_2 = nn.TransformerEncoder(f_transformer_layer_2, num_layers = 2)
        self.f_fc2 = nn.Linear(256,128)
        self.f_fc3 = nn.Linear(128, num_classes)

        
        self.num_classes = num_classes

    def forward(self, vision_inputs, text_input, audio_input):
        # Process vision inputs
        vision_features = [self.vision_model(pixel_values=frame).last_hidden_state for frame in vision_inputs]
        vision_logits = torch.mean(torch.stack(vision_features), dim=0)

        # Process text inputs
        input_ids, attention_masks = Tokenizers.BertTiny_tokenize(text_input, max_length=256)
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_masks)
        text_logits = text_outputs.pooler_output  # Assuming using pooler_output as logits

        # Process audio inputs
        audio_logits = self.audio_model(audio_input)

        # Fuse the outputs
        fused_features = self.fusion_layer(vision_logits, text_logits, audio_logits)

        # Pass through the transformer and linear layers
        x = self.f_transformer_1(fused_features)
        x = self.f_layer_norm(x)
        x = F.relu(self.f_fc(x))
        x = self.f_transformer_2(x)
        x = F.relu(self.f_fc2(x))
        output = self.f_fc3(x)

        return output


class FusionLayer(nn.Module):
    def __init__(self, vision_dim, text_dim, audio_dim, unified_dim):
        super(FusionLayer, self).__init__()
        self.unified_dim = unified_dim
        # Create projection layers for each modality to project them to a unified dimension
        self.vision_projection = nn.Linear(vision_dim, unified_dim)
        self.text_projection = nn.Linear(text_dim, unified_dim)
        self.audio_projection = nn.Linear(audio_dim, unified_dim)

    def forward(self, vision_logits, text_logits, audio_logits):
        # Project each modality's output to the unified dimension
        vision_projected = self.vision_projection(vision_logits)
        text_projected = self.text_projection(text_logits)
        audio_projected = self.audio_projection(audio_logits)

        # Concatenate along the feature dimension
        fused = torch.cat([vision_projected, text_projected, audio_projected], dim=1)
        return fused

class Penalty_Layer(nn.Module):
    def __init__(self):
        super(Penalty_Layer, self).__init__()
        
    def forward(self, vision_logits, text_logits, audio_logits):
        
        vision_probs =  [F.softmax(vision_logits,dim=1) if vision_logits is not None else None]
        text_probs = [F.softmax(text_logits,dim=1) if text_logits is not None else None]
        audio_probs = [F.softmax(audio_logits,dim=1) if audio_logits is not None else None]
        
        v_top_prob, v_top_class = torch.max(vision_probs, dim =  1)
        v_probs_copy = vision_probs.clone()
        v_probs_copy[0, v_top_class] = 0
        v_next_prob, v_next_class = torch.max(v_probs_copy, dim = 1)
        
        
        