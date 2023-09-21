from rosemary import jpt_parse_args, jpt_setup, jpt_in_notebook; jpt_setup()

if jpt_in_notebook():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'

from functools import partial
import os
import sys
import numpy as np

from tqdm import tqdm 

import torch
from torch.utils.data import DataLoader

from datasets import load_dataset

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DataCollatorForSeq2Seq

from open_instruct.finetune_trainer import encode_with_prompt_completion_format, encode_with_messages_format


model_name_or_path = '../results/baselines/huggyllama/llama-7b'

train_file = '../data/processed/all.jsonl'
train_file = '../data/processed/flan_v2/flan_v2_data.jsonl'


data_files = {'train': train_file}
raw_datasets = load_dataset("json", data_files=data_files)
print(len(raw_datasets['train']))



model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map='auto',
    torch_dtype=torch.float16)



tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
tokenizer.padding_side = 'left'
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id


encode_function = partial(
    encode_with_messages_format,
    tokenizer=tokenizer,
    max_seq_length=2048,
)

lm_datasets = raw_datasets.map(
    encode_function,
    batched=False,
    num_proc=16,
    desc="Tokenizing and reformatting instruction data",
)

train_dataset = lm_datasets['train']

train_dataset.set_format(type="torch", 
                         output_all_columns=False, 
                         columns=['input_ids', 'labels', 'attention_mask'])




# collate_fn = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding='longest') 

loader = DataLoader(train_dataset, shuffle=False, batch_size=1) 



device = 'cuda'

text_embeddings = []
log_probs = []

for batch in tqdm(loader, total=len(loader)):
    batch = {k: v.to('cuda', non_blocking=True) for k, v in batch.items()}
    input_ids = batch['input_ids']
    
    with torch.inference_mode():
        outputs = model(**batch, output_hidden_states=True)
    
    # (bsz, seq_len, hidden_size) -> (bsz, hidden_size)
    text_embedding = outputs['hidden_states'][-1].mean(1)
    
    # sum of output token log probs
    log_prob = -outputs['loss']
    
    text_embeddings.append(text_embedding.detach().cpu().numpy())
    log_probs.append(log_prob.detach().cpu().numpy())
    


save_dir = '/gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts'
save_path = os.path.join(save_dir, 'note_explore_dpp_llama-7b_flan_v2_outputs.pkl')

import pickle

d = {'text_embeddings': np.vstack(text_embeddings),
     'log_probs': np.vstack(log_probs)}

with open(save_path, 'wb') as f:
    pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)