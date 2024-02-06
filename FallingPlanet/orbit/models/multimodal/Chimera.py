import torch
from torch import nn

class Chimera(nn.Module):
    def __init__(self,text_model, vision_model, audio_model, **kwargs):
        super(Chimera, self).__init__()
        self.text_model = text_model
        self.vision_model = vision_model
        self.audio_model = audio_model
        
    def forward(self, text_input, vision_inputs, audio_input):
        text_output = self.text_model(text_input)

        vision_output = torch.mean(torch.stack(vision_logits_list), dim=0)
        vision_logits_list = [self.vision_model(pixel_values=features).logits for features in vision_inputs]

        audio_output = self.audio_model(audio_input)
        
        combined_output = torch.cat([text_output,vision_output,audio_output])
        
        return combined_output