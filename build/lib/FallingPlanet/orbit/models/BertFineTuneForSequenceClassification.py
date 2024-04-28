from transformers import BertModel
import torch.nn as nn
import torch

class BertFineTuneBase(nn.Module):
    def __init__(self, num_labels, from_saved_weights = None, num_tasks=1, **kwargs):
        super(BertFineTuneBase, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        
        if from_saved_weights:
            self.bert.load_state_dict(torch.load(from_saved_weights))

        if num_tasks > 1:
            self.classifiers = nn.ModuleList([nn.Linear(768, n_labels) for n_labels in num_labels])
        else:
            self.classifier = nn.Linear(768, num_labels[0])

    def forward(self, input_ids, attention_mask):
        # Process the entire batch through BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # Batched pooled output

        # Apply the classifier(s)
        if hasattr(self, 'classifiers'):
            # Multiple classifiers for multi-task scenario
            all_logits = [classifier(pooled_output) for classifier in self.classifiers]
        else:
            # A single classifier for single-task scenario
            all_logits = self.classifier(pooled_output)

        return all_logits



        
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
        # Process the entire batch through BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # Batched pooled output

        # Apply the classifier(s)
        if hasattr(self, 'classifiers'):
            # Multiple classifiers for multi-task scenario
            all_logits = [classifier(pooled_output) for classifier in self.classifiers]
        else:
            # A single classifier for single-task scenario
            all_logits = self.classifier(pooled_output)

        return all_logits



class BertFineTuneTiny(nn.Module):
    def __init__(self, num_labels, from_saved_weights=None):
        super(BertFineTuneTiny, self).__init__()
        self.bert = BertModel.from_pretrained("prajjwal1/bert-tiny")
        
        if from_saved_weights:
            self.bert.load_state_dict(torch.load(from_saved_weights))
        
        self.classifiers = nn.ModuleList([
            nn.Linear(self.bert.config.hidden_size, labels) for labels in num_labels
        ])

    def forward(self, input_ids_list, attention_mask_list):
        # Input validation
        if len(input_ids_list) != len(self.classifiers) or len(attention_mask_list) != len(self.classifiers):
            raise ValueError(f"Expected {len(self.classifiers)} inputs for each of input_ids and attention_mask, got {len(input_ids_list)} and {len(attention_mask_list)} respectively.")

        task_outputs = []

        for i, classifier in enumerate(self.classifiers):
            # Extract the input_ids and attention_mask for the current task
            input_ids = input_ids_list[i]
            attention_mask = attention_mask_list[i]

            # BERT forward pass
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

            # Use mean pooling on the sequence output to retain batch size
            pooled_output = outputs.last_hidden_state.mean(dim=1)

            # Get logits for the current task
            logits = classifier(pooled_output)
            task_outputs.append(logits)

        return task_outputs




        
class BertFineTuneSmall(nn.Module):
    def __init__(self, num_labels, from_saved_weights=None, num_tasks=1, **kwargs):
        super(BertFineTuneSmall, self).__init__()
        self.bert = BertModel.from_pretrained("prajjwal1/bert-small")
        
        if from_saved_weights:
            self.bert.load_state_dict(torch.load(from_saved_weights))

        if num_tasks > 1:
            self.classifiers = nn.ModuleList([nn.Linear(512, n_labels) for n_labels in num_labels])
        else:
            self.classifier = nn.Linear(512, num_labels[0])

    def forward(self, input_ids, attention_mask):
        # Process the entire batch through BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # Batched pooled output

        # Apply the classifier(s)
        if hasattr(self, 'classifiers'):
            # Multiple classifiers for multi-task scenario
            all_logits = [classifier(pooled_output) for classifier in self.classifiers]
        else:
            # A single classifier for single-task scenario
            all_logits = self.classifier(pooled_output)

        return all_logits




class BertFineTuneMini(nn.Module):
    def __init__(self, num_labels, from_saved_weights=None, num_tasks=1, **kwargs):
        super(BertFineTuneMini, self).__init__()
        self.bert = BertModel.from_pretrained("prajjwal1/bert-mini")
        
        if from_saved_weights:
            self.bert.load_state_dict(torch.load(from_saved_weights))

        if num_tasks > 1:
            self.classifiers = nn.ModuleList([nn.Linear(256, n_labels) for n_labels in num_labels])
        else:
            self.classifier = nn.Linear(256, num_labels[0])

    def forward(self, input_ids, attention_mask):
        # Process the entire batch through BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # Batched pooled output

        # Apply the classifier(s)
        if hasattr(self, 'classifiers'):
            # Multiple classifiers for multi-task scenario
            all_logits = [classifier(pooled_output) for classifier in self.classifiers]
        else:
            # A single classifier for single-task scenario
            all_logits = self.classifier(pooled_output)

        return all_logits



        
class BertFineTuneMedium(nn.Module):
    def __init__(self, num_labels, from_saved_weights=None, num_tasks=1, **kwargs):
        super(BertFineTuneMedium, self).__init__()
        self.bert = BertModel.from_pretrained("prajjwal1/bert-medium")
        
        if from_saved_weights:
            self.bert.load_state_dict(torch.load(from_saved_weights))

        if num_tasks > 1:
            self.classifiers = nn.ModuleList([nn.Linear(512, n_labels) for n_labels in num_labels])
        else:
            self.classifier = nn.Linear(512, num_labels[0])

    def forward(self, input_ids, attention_mask):
        # Process the entire batch through BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # Batched pooled output

        # Apply the classifier(s)
        if hasattr(self, 'classifiers'):
            # Multiple classifiers for multi-task scenario
            all_logits = [classifier(pooled_output) for classifier in self.classifiers]
        else:
            # A single classifier for single-task scenario
            all_logits = self.classifier(pooled_output)

        return all_logits


