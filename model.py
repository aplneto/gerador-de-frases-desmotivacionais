# -*- coding: utf-8 -*-
"""
Created on Sun May  8 23:17:28 2022

@author: aplneto
"""
import torch
import os
import gdown
from transformers import GPT2Tokenizer, AutoModelWithLMHead, set_seed

remote_files = {
    'tokenizer': '1B5ar78rVHLr1032bVJOLSMInI08jkmr7',
    'model': '16g51oB67dAkxvoDQXT9WvaREwjN6M_sL'
}
folder_download_base = 'https://drive.google.com/drive/folders/'

files = os.listdir()
if (not ('tokenizer' in files)):
    gdown.download_folder(url=folder_download_base + remote_files['tokenizer'])
if (not ('model' in files)):
    gdown.download_folder(url=folder_download_base + remote_files['model'])
    

tokenizer = GPT2Tokenizer.from_pretrained('tokenizer')
model = AutoModelWithLMHead.from_pretrained('model')
model.eval()

SPECIAL_TOKENS = tokenizer.special_tokens_map

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
    
default_config_dict = {
    'top_k': 50,
    'temperature': 0.9,
    'max_length': 50,
    'top_p': 0.7,
    'repetition_penalty': 10.0,
    'num_return_sequences': 3,
    'do_sample': True,
    'early_stopping': True
}

set_seed(4)

def get_completions(kw=[], tokenizer=tokenizer, model=model, **kwargs):
    bos = SPECIAL_TOKENS['bos_token']
    sep = SPECIAL_TOKENS['sep_token']
    prompt = (bos + ','.join(kw) + sep)
    p = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    p = p.to(device)
    samples = model.generate(p, **kwargs)
    outputs = []
    s = len(','.join(kw))
    for sample in samples:
        text = tokenizer.decode(sample, skip_special_tokens=True)
        outputs.append(text[s:])
    return outputs
