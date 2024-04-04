import torch
import torch.nn as nn
from FallingPlanet.orbit.models.AuTransformer import FPATF_Tiny
from FallingPlanet.orbit.models.BertFineTuneForSequenceClassification import BertFineTuneTiny
from FallingPlanet.orbit.models.DeiTFineTuneForImageClassification import DeitFineTuneTiny
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
        self.unknown_label = unified_dim - 1  # Assign the last index to the "unknown" label

    def forward(self, vision_logits, text_logits, audio_logits):
        # Map vision logits to the unified label space
        _, vision_indices = torch.max(vision_logits, dim=1)
        vision_labels = [list(self.vision_label_map.keys())[list(self.vision_label_map.values()).index(i.item())] if i.item() in self.vision_label_map.values() else self.unknown_label for i in vision_indices]

        # Map text logits to the unified label space
        _, text_indices = torch.max(text_logits, dim=1)
        text_labels = [list(self.text_label_map.keys())[list(self.text_label_map.values()).index(i.item())] if i.item() in self.text_label_map.values() else self.unknown_label for i in text_indices]

        # Map audio logits to the unified label space
        _, audio_indices = torch.max(audio_logits, dim=1)
        audio_labels = [list(self.audio_label_map.keys())[list(self.audio_label_map.values()).index(i.item())] if i.item() in self.audio_label_map.values() else self.unknown_label for i in audio_indices]

        # Fuse outputs
        vision_projected = self.vision_projection(vision_logits)
        text_projected = self.text_projection(text_logits)
        audio_projected = self.audio_projection(audio_logits)
        # Debugging: Print shapes after projection
        
        fused_features = torch.cat([vision_projected, text_projected, audio_projected], dim=1)
        # Debugging: Print shape after concatenation
        

        return fused_features, vision_labels, text_labels, audio_labels

    
    
class HydraTiny(nn.Module):
    def __init__(self, num_classes, feature_dim,text_label_map,vision_label_map,audio_label_map, unified_label_map, requires_grad=False):
        super(HydraTiny, self).__init__()

        # Vision model initialization with DeiT
        self.vision_model = DeitFineTuneTiny(num_labels=[len(vision_label_map)-1])
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
            num_classes=len(audio_label_map)-2, 
            num_heads=16, 
            dim_feedforward=1024, 
            num_layers=4, 
            dropout=0.1
        )
        if not requires_grad:
            for param in self.audio_model.parameters():
                param.requires_grad = False

        # Fusion layer initialization
        self.fusion_layer = FusionLayer(vision_dim=8, text_dim=9, audio_dim=7, unified_dim=176, vision_label_map=vision_label_map, text_label_map=text_label_map, audio_label_map=audio_label_map)

        # Additional layers for processing fused features
        self.f_transformer_1 = self.create_transformer_encoder_layer(528, 8, 1024, 6)
        self.f_layer_norm = nn.LayerNorm(528)
        self.f_fc = nn.Linear(528, 256)
        self.f_transformer_2 = self.create_transformer_encoder_layer(d_model=256, nhead=8, dim_feedforward=512, num_layers=2, dropout=0.01)
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
        
                # Example adjustment, assuming vision_inputs ends up with an extra unwanted dimension
        if isinstance(vision_inputs, list):
            vision_inputs = torch.cat(vision_inputs)  # Use cat instead of stack if they're already batch tensors
        vision_inputs = torch.squeeze(vision_inputs, dim=1)
        # Verify the shape is as expected
        

        # Now proceed with your model processing
        vision_logits = self.vision_model(pixel_values=vision_inputs)
        vision_logits = vision_logits.mean(dim=0, keepdim=True)
    
                # Squeezing the extra dimension from input_ids and attention_mask
        squeezed_input_ids = torch.squeeze(text_inputs[0], dim=1)
        squeezed_attention_mask = torch.squeeze(text_inputs[1], dim=1)

        # Now, input_ids and attention_mask are of shape [1, 256]
        # You can then pass these squeezed tensors to your model
        text_logits = self.text_model(input_ids=squeezed_input_ids, attention_mask=squeezed_attention_mask)

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
    