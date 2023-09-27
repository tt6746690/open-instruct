from functools import partial
import os
import numpy as np
import time

import pickle
from tqdm import tqdm 

import pyarrow # import before `torch`, `transformers`, `datasets`
import torch
from torch.utils.data import DataLoader

from datasets import load_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer

from open_instruct.finetune_trainer import encode_with_prompt_completion_format, encode_with_messages_format


test_run = False
device = 'cuda'
model_name_or_path = '../results/baselines/huggyllama/llama-7b'

processed_dir = '../data/processed'

save_dir = '/gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/llama-7b_outputs'
os.makedirs(save_dir, exist_ok=True)


model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map='auto',
    torch_dtype=torch.float16)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
tokenizer.padding_side = 'left'
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id



for dataset in os.listdir(processed_dir):
    dataset_path = os.path.join(processed_dir, dataset)
    if dataset in ['tulu'] or not os.path.isdir(dataset_path):
        continue
    train_file = os.path.join(processed_dir, dataset, f'{dataset}_data.jsonl')
    assert(os.path.isfile(train_file))
    
    s = time.time()
    
    data_files = {'train': train_file}
    raw_datasets = load_dataset("json", data_files=data_files)
    if test_run:
        raw_datasets['train'] = raw_datasets['train'].select(range(100))
    print(f"{dataset} dataset length = {len(raw_datasets['train'])}")

    encode_function = partial(
        encode_with_messages_format, tokenizer=tokenizer, max_seq_length=2048)

    lm_datasets = raw_datasets.map(
        encode_function, batched=False, num_proc=16,
        desc="Tokenizing and reformatting instruction data")

    train_dataset = lm_datasets['train']
    train_dataset.set_format(
        type="torch", output_all_columns=False, columns=['input_ids', 'labels', 'attention_mask'])
    loader = DataLoader(train_dataset, shuffle=False, batch_size=1) 


    text_embeddings = []
    log_probs = []
    for batch in tqdm(loader, total=len(loader)):
        batch = {k: v.to('cuda', non_blocking=True) for k, v in batch.items()}
        with torch.inference_mode():
            outputs = model(**batch, output_hidden_states=True)

        # (bsz, seq_len, hidden_size) -> (bsz, hidden_size)
        text_embedding = outputs['hidden_states'][-1].mean(1)
        # sum of output token log probs
        log_prob = -outputs['loss']

        text_embeddings.append(text_embedding.detach().cpu().numpy().astype(np.float32))
        log_probs.append(log_prob.detach().cpu().numpy())


    output = {'text_embeddings': np.vstack(text_embeddings),
              'log_probs': np.vstack(log_probs)}
    
    save_path = os.path.join(save_dir, f'{dataset}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

    e = time.time()
    print(f"Finished computing embedding/logprob for {dataset} in {e-s:.2f} seconds")

    