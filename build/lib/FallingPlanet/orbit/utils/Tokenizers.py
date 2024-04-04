import transformers
import torch
from transformers import BertTokenizerFast

def BertTiny_tokenize(text, max_length=256):
    tokenizer = BertTokenizerFast.from_pretrained("prajjwal1/bert-tiny")
    encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,  # Explicitly activate truncation
            padding='max_length',  # Pad to max_length
            return_attention_mask=True,
            return_tensors="pt"
    )
    
    return encoded["input_ids"], encoded["attention_mask"]



def BertTiny_batch_tokenize(texts, max_length=256):
    tokenizer = BertTokenizerFast.from_pretrained("prajjwal1/bert-tiny")
    encoded_batch = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_attention_mask=True,
        return_tensors="pt"
    )
    
    return encoded_batch["input_ids"], encoded_batch["attention_mask"]

def BertMini_tokenize(text, max_length=256):
    tokenizer = BertTokenizerFast.from_pretrained("prajjwal1/bert-mini")
    encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,  # Explicitly activate truncation
            padding='max_length',  # Pad to max_length
            return_attention_mask=True,
            return_tensors="pt"
    )
    
    return encoded["input_ids"], encoded["attention_mask"]

def BertSmall_tokenize(text, max_length =256):
    tokenizer = BertTokenizerFast.from_pretrained("prajjwal1/bert-small")
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens = True,
        max_length=max_length,
        truncation = True,
        padding = 'max_length',
        return_attention_mask = True,
        return_tensors = "pt"
    )
    return encoded["input_ids"], encoded["attention_mask"]
