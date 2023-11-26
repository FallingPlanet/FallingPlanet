from transformers import BertModel
import torch.nn as nn
import torch
class BertFineTuneForSequenceClassification(nn.Module):
    def __init__(self, num_labels, from_saved_weights=None, num_tasks=1, **kwargs):
        super(BertFineTuneForSequenceClassification, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        
        if from_saved_weights:
            self.bert.load_state_dict(torch.load(from_saved_weights))

        if num_tasks > 1:
            self.classifiers = nn.ModuleList([nn.Linear(1024, n_labels) for n_labels in num_labels])
        else:
            self.classifier = nn.Linear(128, num_labels[0])

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        if hasattr(self, 'classifiers'):
            return [classifier(pooled_output) for classifier in self.classifiers]
        else:
            return self.classifier(pooled_output)
