import torch
import torch.nn as nn
from transformers import ViTModel
from torchvision import transforms

class DeitFineTuneTiny(nn.Module):
    def __init__(self, num_labels, from_saved_weights = None, num_tasks=1, image_size= 224, **kwargs):
        super(DeitFineTuneTiny, self).__init__()
        self.beit = ViTModel.from_pretrained('facebook/deit-tiny-patch16-224')
        
        
        if from_saved_weights:
            self.beit.load_state_dict(torch.load(from_saved_weights))
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size,image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3,1,1) if x.size(0) == 1 else x)
            
        ])
        if num_tasks > 1:
            self.classifiers = nn.ModulaList([nn.Linear(192,num_labels)])
        
        else: 
            self.classifier = nn.Linear(192, num_labels[0])
            
    def forward(self, pixel_values):
        #Process the entire patch through BEiT
        outputs = self.beit(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output #batched pooler output
        
        #apply classifiers
        if hasattr(self,'classifiers'):
            #mutliply classifiers for a multitask scenario
            all_logits = [classifier(pooled_output) for classifier in self.classifiers]
            
        else:
            # single classifier
            all_logits = self.classifier(pooled_output)
            
        return all_logits   
            
            
class DeitFineTuneSmall(nn.Module):
    def __init__(self, num_labels, from_saved_weights = None, num_tasks=1, image_size= 224, **kwargs):
        super(DeitFineTuneSmall, self).__init__()
        self.beit = ViTModel.from_pretrained('facebook/deit-small-patch16-224')
        
        
        if from_saved_weights:
            self.beit.load_state_dict(torch.load(from_saved_weights))
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size,image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3,1,1) if x.size(0) == 1 else x)
            
        ])
        if num_tasks > 1:
            self.classifiers = nn.ModulaList([nn.Linear(384,num_labels)])
        
        else: 
            self.classifier = nn.Linear(384, num_labels[0])
            
    def forward(self, pixel_values):
        #Process the entire patch through BEiT
        outputs = self.beit(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output #batched pooler output
        
        #apply classifiers
        if hasattr(self,'classifiers'):
            #mutliply classifiers for a multitask scenario
            all_logits = [classifier(pooled_output) for classifier in self.classifiers]
            
        else:
            # single classifier
            all_logits = self.classifier(pooled_output)
            
        return all_logits   
