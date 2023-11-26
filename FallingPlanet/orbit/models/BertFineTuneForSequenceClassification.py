from transformers import BertModel
import torch.nn as nn
import torch

class BertFineTuneBase(nn.Module):
    def __init__(self, num_labels, from_saved_weights=None, num_tasks=1, **kwargs):
        super(BertFineTuneBase, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        
        if from_saved_weights:
            self.bert.load_state_dict(torch.load(from_saved_weights))

        if num_tasks > 1:
            self.classifiers = nn.ModuleList([nn.Linear(768, n_labels) for n_labels in num_labels])
        else:
            self.classifier = nn.Linear(768, num_labels[0])

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        if hasattr(self, 'classifiers'):
            return [classifier(pooled_output) for classifier in self.classifiers]
        else:
            return self.classifier(pooled_output)
        
class BertFineTuneLarge(nn.Module):
    def __init__(self, num_labels, from_saved_weights=None, num_tasks=1, **kwargs):
        super(BertFineTuneLarge, self).__init__()
        self.bert = BertModel.from_pretrained("bert-large-cased")
        
        if from_saved_weights:
            self.bert.load_state_dict(torch.load(from_saved_weights))

        if num_tasks > 1:
            self.classifiers = nn.ModuleList([nn.Linear(1024, n_labels) for n_labels in num_labels])
        else:
            self.classifier = nn.Linear(1024, num_labels[0])

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        if hasattr(self, 'classifiers'):
            return [classifier(pooled_output) for classifier in self.classifiers]
        else:
            return self.classifier(pooled_output)
        
class BertFineTuneTiny(nn.Module):
    def __init__(self, num_labels, from_saved_weights=None, num_tasks=1, **kwargs):
        super(BertFineTuneTiny, self).__init__()
        self.bert = BertModel.from_pretrained("prajjwal1/bert-tiny")
        
        if from_saved_weights:
            self.bert.load_state_dict(torch.load(from_saved_weights))

        if num_tasks > 1:
            self.classifiers = nn.ModuleList([nn.Linear(128, n_labels) for n_labels in num_labels])
        else:
            self.classifier = nn.Linear(128, num_labels[0])

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        if hasattr(self, 'classifiers'):
            return [classifier(pooled_output) for classifier in self.classifiers]
        else:
            return self.classifier(pooled_output)
        
class BertFineTuneSmall(nn.Module):
    def __init__(self, num_labels, from_saved_weights=None, num_tasks=1, **kwargs):
        super(BertFineTuneTiny, self).__init__()
        self.bert = BertModel.from_pretrained("prajjwal1/bert-small")
        
        if from_saved_weights:
            self.bert.load_state_dict(torch.load(from_saved_weights))

        if num_tasks > 1:
            self.classifiers = nn.ModuleList([nn.Linear(512, n_labels) for n_labels in num_labels])
        else:
            self.classifier = nn.Linear(512, num_labels[0])

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        if hasattr(self, 'classifiers'):
            return [classifier(pooled_output) for classifier in self.classifiers]
        else:
            return self.classifier(pooled_output)

class BertFineTuneMini(nn.Module):
    def __init__(self, num_labels, from_saved_weights=None, num_tasks=1, **kwargs):
        super(BertFineTuneTiny, self).__init__()
        self.bert = BertModel.from_pretrained("prajjwal1/bert-small")
        
        if from_saved_weights:
            self.bert.load_state_dict(torch.load(from_saved_weights))

        if num_tasks > 1:
            self.classifiers = nn.ModuleList([nn.Linear(256, n_labels) for n_labels in num_labels])
        else:
            self.classifier = nn.Linear(256, num_labels[0])

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        if hasattr(self, 'classifiers'):
            return [classifier(pooled_output) for classifier in self.classifiers]
        else:
            return self.classifier(pooled_output)