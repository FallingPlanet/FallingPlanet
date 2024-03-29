import torch
import torch.nn as nn
from transformers import ViTModel, ViTFeatureExtractor, DeiTForImageClassification, BertModel, DeiTModel
from FallingPlanet.orbit.models.AuTransformer import FPATF_Tiny
from FallingPlanet.orbit.models.BertFineTuneForSequenceClassification import BertFineTuneTiny
from FallingPlanet.orbit.models.DeiTFineTuneForImageClassification import FPDeitFineTuneTiny
import torch.nn.functional as F



class FusionLayer(nn.Module):
    def __init__(self, vision_dim, text_dim, audio_dim, unified_dim, vision_label_map, text_label_map, audio_label_map):
        super(FusionLayer, self).__init__()
        self.vision_projection = nn.Linear(vision_dim, unified_dim)
        self.text_projection = nn.Linear(text_dim, unified_dim)
        self.audio_projection = nn.Linear(audio_dim, unified_dim)
        self.vision_label_map = vision_label_map
        self.text_label_map = text_label_map
        self.audio_label_map = audio_label_map

    def forward(self, vision_logits, text_input, audio_input):
        # Map vision logits to the unified label space
        vision_labels = [self.vision_label_map[label] for label in vision_logits.argmax(dim=1)]
        vision_projected = self.vision_projection(vision_logits)

        # Map text logits to the unified label space
        
        text_outputs = self.text_model(input_ids=text_input[0], attention_mask=text_input[1])
        text_labels = [self.text_label_map[label] for label in text_outputs.pooler_output.argmax(dim=1)]
        text_projected = self.text_projection(text_outputs.pooler_output)

        # Map audio logits to the unified label space
        audio_labels = [self.audio_label_map[label] for label in audio_input.argmax(dim=1)]
        audio_projected = self.audio_projection(audio_input)

        # Fuse outputs
        fused_features = torch.cat([vision_projected, text_projected, audio_projected], dim=1)

        return fused_features, vision_labels, text_labels, audio_labels

class HydraTiny(nn.Module):
    def __init__(self, num_classes, feature_dim,text_label_map,vision_label_map,audio_label_map, requires_grad=False):
        super(HydraTiny, self).__init__()

        # Vision model initialization with DeiT
        self.vision_model = FPDeitFineTuneTiny(num_labels=[7])
        if not requires_grad:
            for param in self.vision_model.parameters():
                param.requires_grad = False
       
        
        # Text model initialization
        self.text_model = BertFineTuneTiny(num_labels=[num_classes])
        if not requires_grad:
            for param in self.text_model.parameters():
                param.requires_grad = False

        # Audio model initialization with custom specifications
        self.audio_model = FPATF_Tiny(
            target_channels=feature_dim, 
            num_classes=num_classes, 
            num_heads=16, 
            dim_feedforward=1024, 
            num_layers=4, 
            dropout=0.1
        )
        if not requires_grad:
            for param in self.audio_model.parameters():
                param.requires_grad = False

        # Fusion layer initialization
        self.fusion_layer = FusionLayer(vision_dim=768, text_dim=768, audio_dim=feature_dim, unified_dim=768, vision_label_map=vision_label_map, text_label_map=text_label_map, audio_label_map=audio_label_map)

        # Additional layers for processing fused features
        self.f_transformer_1 = self.create_transformer_encoder_layer(512, 8, 1024, 6)
        self.f_layer_norm = nn.LayerNorm(512)
        self.f_fc = nn.Linear(512, 256)
        self.f_transformer_2 = self.create_transformer_encoder_layer(256, 8, 512, 2)
        self.f_fc2 = nn.Linear(256, 128)
        self.f_fc3 = nn.Linear(128, num_classes)
        
    def load_modal_state_dicts(self, text_dict=None, vision_dict=None, audio_dict=None):
        if text_dict:
            self.text_model.load_state_dict(torch.load(text_dict))
        if vision_dict:
            self.vision_model.load_state_dict(torch.load(vision_dict))
        if audio_dict:
            self.audio_model.load_state_dict(torch.load(audio_dict))
            
    def create_transformer_encoder_layer(self, d_model, nhead, dim_feedforward, num_layers, dropout=0.01):
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        return nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, vision_inputs, text_inputs, audio_input):
        if isinstance(vision_inputs, list):
            vision_inputs = torch.stack(vision_inputs)  # Converts list to tensor of shape (batch_size, C, H, W)

        # Process vision inputs as a batch
        vision_features = self.vision_model(pixel_values=vision_inputs).last_hidden_state
        vision_logits = torch.mean(vision_features, dim=1)

        # Process text inputs
        text_outputs = self.text_model(input_ids=text_inputs[0], attention_mask=text_inputs[1])
        text_logits = text_outputs.logits 
        text_logits = text_outputs.pooler_output

        # Process audio inputs
        audio_logits = self.audio_model(audio_input)

        # Fuse outputs
        fused_features, vision_labels, text_labels, audio_labels = self.fusion_layer(vision_logits, text_logits, audio_logits)

        # Process fused features through transformers and linear layers
        x = self.f_transformer_1(fused_features)
        x = self.f_layer_norm(x)
        x = F.relu(self.f_fc(x))
        x = self.f_transformer_2(x)
        x = F.relu(self.f_fc2(x))
        output = self.f_fc3(x)

        return output, vision_labels, text_labels, audio_labels


        
        
                
         
                    
        
        
        

       
       
"""class Penalty_Layer(nn.Module):
def __init__(self):
    super(Penalty_Layer, self).__init__()
    
def forward(self, vision_logits, text_logits, audio_logits):
    probs = [
    F.softmax(vision_logits,dim=1) if vision_logits is not None else None,
    F.softmax(text_logits,dim=1) if text_logits is not None else None,
    F.softmax(audio_logits,dim=1) if audio_logits is not None else None]
    
    # Extract top class probabilities and indices for each modality
    
    top_2_probs = []
    top_2_classes = []
    top_classes = []
    next_classes = []
    penalty = 0
    for prob in probs:
        if prob is not None:
            
            top_2_prob, top_2_class = torch.topk(prob,2,dim=1)
            
            top_2_probs.append(top_2_prob)
            top_2_classes.append(top_2_class)
            
            top_classes.append(top_2_class[:, 0])
            next_classes.append(top_2_class[:,1])
        else:
            
            top_2_classes.append(None)
            top_2_probs.append(None)
            
                    # Example conditional checks for vision modality compared to text and audio
    if top_classes[0] is not None:
        if top_classes[1] is not None and not torch.equal(top_classes[0], top_classes[1]):
            # Vision top class does not match Text top class
            if top_classes[2] is not None and torch.equal(top_classes[0],top_classes[2]):
                penalty=0
            if top_classes[0]

        if top_classes[2] is not None and not torch.equal(top_classes[0], top_classes[2]):
            # Vision top class does not match Audio top class
    
    #this condition is technically impossible
    if top_classes[1] is not None:  # Check if Text modality is present
        if top_classes[0] is not None and not torch.equal(top_classes[1], top_classes[0]):
            # Text top class does not match Vision top class
            # Apply penalty or specific logic here
            
        if top_classes[2] is not None and not torch.equal(top_classes[1], top_classes[2]):
            # Text top class does not match Audio top class
            # Apply penalty or specific logic here

    if top_classes[2] is not None:  # Check if Audio modality is present
        if top_classes[0] is not None and not torch.equal(top_classes[2], top_classes[0]):
            # Audio top class does not match Vision top class
            # Apply penalty or specific logic here
            
        if top_classes[1] is not None and not torch.equal(top_classes[2], top_classes[1]):
            # Audio top class does not match Text top class
            # Apply penalty or specific logic here
"""
    