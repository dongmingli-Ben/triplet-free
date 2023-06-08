from typing import List
from transformers import AutoTokenizer

TOKENIZER = {
    'fnlp/bart-base-chinese': 'bert-base-chinese',
    'fnlp/bart-large-chinese': 'bert-base-chinese',
    'bert-base-chinese': 'bert-base-chinese',
}

def build_tokenizer(tokenizer: str, backbone: str):
    if backbone != '':
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER[backbone], use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
    return tokenizer