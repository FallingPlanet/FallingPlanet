


import torch
import torch.nn as nn
import torch.nn.functional as F
from FallingPlanet.orbit.models.AuTransformer import FPATF_Tiny
from FallingPlanet.orbit.models.BertFineTuneForSequenceClassification import BertFineTuneTiny
from FallingPlanet.orbit.models.DeiTFineTuneForImageClassification import DeitFineTuneTiny
class MultiModalAttention(nn.Module):
    def __init__(self, text_feature_dim, vision_feature_dim, audio_feature_dim, num_heads):
        super(MultiModalAttention, self).__init__()
        common_dim = 192
        self.text_proj = nn.Linear(text_feature_dim, common_dim)  # Adjusted to use text_feature_dim
        self.vision_proj = nn.Linear(vision_feature_dim, common_dim)  # Adjusted to use vision_feature_dim
        self.audio_proj = nn.Linear(audio_feature_dim, common_dim)  # Adjusted to use audio_feature_dim
        self.attention = nn.MultiheadAttention(embed_dim=common_dim, num_heads=num_heads)
        self.norm = nn.LayerNorm(common_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, text_features, vision_features, audio_features):
        # Ensure features are properly projected to the common dimension
        text_proj = self.text_proj(text_features).unsqueeze(0)
        vision_proj = self.vision_proj(vision_features).unsqueeze(0)
        audio_proj = self.audio_proj(audio_features).unsqueeze(0)
        
        # Concatenate along the time dimension
        features = torch.cat((text_proj, vision_proj, audio_proj), dim=0)
        
        # Apply attention
        attn_output, _ = self.attention(features, features, features)
        
        # Normalize and apply dropout
        output = self.dropout(self.norm(attn_output))
        return output
import torch
import torch.nn as nn
import torch.nn.functional as F
from FallingPlanet.orbit.models.AuTransformer import FPATF_Tiny
from FallingPlanet.orbit.models.BertFineTuneForSequenceClassification import BertFineTuneTiny
from FallingPlanet.orbit.models.DeiTFineTuneForImageClassification import DeitFineTuneTiny

class HydraTiny(nn.Module):
    def __init__(self, num_classes, text_label_map, vision_label_map, audio_label_map, unified_label_map, mode='attention', requires_grad=False, text_dict=None, vision_dict=None, audio_dict=None):
        super(HydraTiny, self).__init__()
        self.mode = mode
        self.text_model = BertFineTuneTiny(num_labels=len(text_label_map), return_features=True)
        self.vision_model = DeitFineTuneTiny(num_labels=len(vision_label_map), return_features=True)
        self.audio_model = FPATF_Tiny(target_channels=30, num_classes=len(audio_label_map), num_heads=16, dim_feedforward=1024, num_layers=4, dropout=0.1, return_features=True)
        self.multi_modal_attention = MultiModalAttention(text_feature_dim=768, vision_feature_dim=768, audio_feature_dim=30, num_heads=3)
        self.classifier = nn.Linear(192, num_classes)
        self.text_label_map = text_label_map
        self.vision_label_map = vision_label_map
        self.audio_label_map = audio_label_map
        self.unified_label_map = unified_label_map
        
        # Load model weights
        self.load_modal_state_dicts(text_dict, vision_dict, audio_dict)

    def load_modal_state_dicts(self, text_dict=None, vision_dict=None, audio_dict=None):
        # Determine whether to remove classifier weights based on mode
        remove_classifier = self.mode == 'attention'  # Remove in 'attention' mode only

        def filter_state_dict(state_dict, model_prefix, remove_classifier):
            # For BertFineTuneTiny and DeitFineTuneTiny, keys to remove would typically include 'classifier.weight' and 'classifier.bias' if specified
            keys_to_remove = [f"{model_prefix}.classifier.weight", f"{model_prefix}.classifier.bias"] if remove_classifier else []
            return {k: v for k, v in state_dict.items() if k not in keys_to_remove}

        if text_dict:
            text_state_dict = torch.load(text_dict, map_location=torch.device('cuda'))
            filtered_text_state_dict = filter_state_dict(text_state_dict, 'text_model', remove_classifier)
            self.text_model.load_state_dict(filtered_text_state_dict, strict=False)

        if vision_dict:
            vision_state_dict = torch.load(vision_dict, map_location=torch.device('cuda'))
            filtered_vision_state_dict = filter_state_dict(vision_state_dict, 'vision_model', remove_classifier)
            self.vision_model.load_state_dict(filtered_vision_state_dict, strict=False)

        if audio_dict:
            audio_state_dict = torch.load(audio_dict, map_location=torch.device('cuda'))
            filtered_audio_state_dict = filter_state_dict(audio_state_dict, 'audio_model', remove_classifier)
            self.audio_model.load_state_dict(filtered_audio_state_dict, strict=False)

    def forward(self, text_inputs, vision_inputs, audio_inputs):
        if isinstance(vision_inputs, list):
            vision_inputs = torch.stack(vision_inputs).to(torch.float32)
        vision_inputs = vision_inputs.squeeze(1).squeeze(1)
        input_ids, attention_mask = text_inputs
        input_ids = torch.squeeze(input_ids, dim=1)
        attention_mask = torch.squeeze(attention_mask, dim=1)

        if self.mode == 'attention':
            vision_features = self.vision_model(vision_inputs)
            text_features = self.text_model(input_ids, attention_mask)
            audio_features = self.audio_model(audio_inputs)

            integrated_features = self.multi_modal_attention(text_features, vision_features, audio_features)
            integrated_features = integrated_features.mean(dim=0).unsqueeze(0)
            logits = self.classifier(integrated_features)
        elif self.mode == 'concat':
            logits_text = self.text_model(input_ids,attention_mask)
            logits_vision = self.vision_model(vision_inputs)
            logits_audio = self.audio_model(audio_inputs)

            logits_text = self.remap_logits_to_unified(logits_text, self.text_label_map)
            logits_vision = self.remap_logits_to_unified(logits_vision, self.vision_label_map)
            logits_audio = self.remap_logits_to_unified(logits_audio, self.audio_label_map)

            logits = (logits_text + logits_vision + logits_audio) / 3

        probs = F.softmax(logits, dim=1)
        return probs


    def remap_logits_to_unified(self, logits, label_map):
        remapped_logits = torch.zeros(logits.size(0), len(self.unified_label_map), device=logits.device)
        for label, index in label_map.items():
            unified_index = self.unified_label_map.get(label, -1)
            if unified_index != -1:
                remapped_logits[:, unified_index] = logits[:, index]
        return remapped_logits

    

    
