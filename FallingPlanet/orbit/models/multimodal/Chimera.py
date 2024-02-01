import torch
from torch import nn

class Chimera(nn.Module):
    def __init__(self,text_model, vision_model, audio_model):
        super(Chimera, self).__init__()
        self.text_model = text_model
        self.vision_model = vision_model
        self.audio_model = audio_model
        
    def forward(self, text_input, vision_input, audio_input):
        text_output = self.text_model(text_input)
        vision_output = self.vision_model(vision_input)
        audio_output = self.audio_model(vision_output)
        
        combined_output = torch.cat([text_output,vision_output,audio_output])
        
        return combined_output