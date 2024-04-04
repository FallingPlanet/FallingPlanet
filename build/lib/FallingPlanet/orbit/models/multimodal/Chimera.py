import torch
import torch.nn as nn

class Chimera(nn.Module):
    def __init__(self, text_model, vision_model, audio_model, text_state_dict=None, vision_state_dict=None, audio_state_dict=None):
        super(Chimera, self).__init__()
        self.text_model = text_model
        self.vision_model = vision_model
        self.audio_model = audio_model

        if text_state_dict:
            self.text_model.load_state_dict(torch.load(text_state_dict))
        if vision_state_dict:
            self.vision_model.load_state_dict(torch.load(vision_state_dict))
        if audio_state_dict:
            self.audio_model.load_state_dict(torch.load(audio_state_dict))

        self.text_model.eval()
        self.vision_model.eval()
        self.audio_model.eval()

    def forward(self, text_inputs, vision_inputs, audio_inputs):
        input_ids, attention_mask = text_inputs['input_ids'], text_inputs['attention_mask']
        text_logits = self.text_model(input_ids=input_ids, attention_mask=attention_mask)

        # Handle vision inputs as a batch tensor
        vision_logits = self.vision_model(pixel_values=vision_inputs)
        vision_logits = vision_logits.mean(dim=1, keepdim=False)

        audio_logits = self.audio_model(audio_inputs)

        return text_logits, vision_logits, audio_logits


  