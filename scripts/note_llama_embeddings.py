from functools import partial
import os
import numpy as np
import time

import random
import pickle
from tqdm import tqdm 
import datetime

import pyarrow # import before `torch`, `transformers`, `datasets`
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist

from datasets import load_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer

from open_instruct.finetune_trainer import encode_with_prompt_completion_format, encode_with_messages_format


def combine_lm_outputs_for_mixes(dataset, save_dir, test_run):
        
    mixes = {
        'tulu_v1_human_mix': ['flan_v2', 'cot', 'dolly', 'oasst1'],
        'tulu_v2_human_mix': ['flan_v2', 'cot', 'oasst1', 'lima'],
    }

    mix_name = dataset
    mix_datasets = mixes[dataset]

    output_list = []
    for dataset in mix_datasets:
        save_path = os.path.join(save_dir, f'{dataset}.pkl')
        with open(save_path, 'rb') as f:
            output = pickle.load(f)
        output_list.append(output)

    output = {}
    for k in ['text_embeddings', 'log_probs']:
        output[k] = np.vstack([x[k] for x in output_list])

    save_path = os.path.join(save_dir, ('test_' if test_run else '')+f'{datasemix_name}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)


    print(f'dataset: {dataset}')
    print(f'Save output={[(k, v.shape) for k, v in output.items()]} to {save_path}')


def datasets_shard_chunk_size(N, num_shards, index):
    """Get chunk size for `datasets.shard(..., contiguous=True)`. """
    div = N // num_shards
    mod = N % num_shards
    start = div * index + min(index, mod)
    end = start + div + (1 if index < mod else 0)
    return end-start


def dist_gather_and_vstack_2d_tensors(tensor, tensor_list_sizes, rank):
    """ For ncll backend: requires roughly (world_size+1)/world_size x of 
            stacked tensor's memory to fit in 1 gpu. 
        For gloo backend, as long as everything fits in cpu memory.
            however, timeout if send/recv hangs for 180000ms. transfering 
            large data gives timeout error.
        """
    D = tensor.shape[1]
    max_chunk_size = max(tensor_list_sizes)
    tensor_list = [torch.zeros((max_chunk_size, D), 
                                dtype=tensor.dtype, device=tensor.device) 
                    for _ in range(len(tensor_list_sizes))]
    # Note `tensor_list` have to be same shape, 
    # pad `tensor` in all processes to same length before gather.
    if tensor.shape[0] != max_chunk_size:
        tensor = torch.vstack([
            tensor,
            torch.zeros((max_chunk_size-tensor.shape[0], D),
                            device=tensor.device, dtype=tensor.dtype)
        ])
    if rank == 0:
        dist.gather(tensor, gather_list=tensor_list, dst=0)
    else:
        dist.gather(tensor, gather_list=[], dst=0)
    # remove padding 
    tensor_list = [x[:B] for x, B in zip(tensor_list, tensor_list_sizes)]
    tensor = torch.vstack(tensor_list)
    return tensor


def compute_lm_outputs(
        dataset,
        model_name_or_path='../results/baselines/huggyllama/llama-7b',
        save_dir='/gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/llama-7b_outputs',
        use_dist=False,
        test_run=False,
        shuffle=False,
    ):
    """
        `shuffle` to allow each process to process roughly similar workload cross the dataset 
            to avoid unnecessary waiting.
    """

    os.makedirs(save_dir, exist_ok=True)

    if dataset in ['tulu_v1_human_mix', 'tulu_v2_human_mix']:
        combine_lm_outputs_for_mixes(dataset, save_dir, test_run)
        return

    if use_dist:
        dist.init_process_group("gloo", timeout=datetime.timedelta(hours=6))
        world_size = dist.get_world_size()
        rank = dist.get_rank() # global rank
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        local_rank = 0
        world_size = 1

    print(f'rank/local_rank/world_size: {rank}/{local_rank}/{world_size}\n')

    device = f'cuda:{str(local_rank)}'

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map=device,
        torch_dtype=torch.float16)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, use_fast=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    processed_dir = '../data/processed'
    if 'flan2022' in dataset:
        train_file = os.path.join(processed_dir, 'flan2022', f'{dataset}_data.jsonl')
    else:
        train_file = os.path.join(processed_dir, dataset, f'{dataset}_data.jsonl')
    assert(os.path.isfile(train_file))

    data_files = {'train': train_file}
    raw_datasets = load_dataset("json", data_files=data_files)
    # if test_run:
    #     raw_datasets['train'] = raw_datasets['train'].select(range(100))
    print(f"{dataset} dataset length = {len(raw_datasets['train'])}")

    encode_function = partial(
        encode_with_messages_format, tokenizer=tokenizer, max_seq_length=2048)

    if rank == 0:
        lm_datasets = raw_datasets.map(
            encode_function, batched=False, num_proc=16,
            desc="Tokenizing and reformatting instruction data")
    if use_dist:
        dist.barrier()
    if rank!= 0:
        lm_datasets = raw_datasets.map(
            encode_function, batched=False, num_proc=16,
            desc="Tokenizing and reformatting instruction data")

    train_dataset = lm_datasets['train']
    train_dataset.set_format(
        type="torch",
        output_all_columns=False,
        columns=['input_ids', 'labels', 'attention_mask'])
    if shuffle:
        random.seed(0)
        shuffle_inds = list(range(len(train_dataset)))
        random.shuffle(shuffle_inds)
        reverse_shuffle_inds = [(i, ind) for i, ind in enumerate(shuffle_inds)]
        reverse_shuffle_inds = sorted(reverse_shuffle_inds, key=lambda x: x[1])
        reverse_shuffle_inds = [x[0] for x in reverse_shuffle_inds]
        train_dataset = train_dataset.select(shuffle_inds)
    train_dataset_chunk_sizes = [datasets_shard_chunk_size(len(train_dataset), num_shards=world_size, index=i) 
                for i in range(world_size)]
    train_dataset = train_dataset.shard(
        num_shards=world_size, 
        index=rank,
        contiguous=True)
    loader = DataLoader(train_dataset, shuffle=False, batch_size=1, pin_memory=True) 


    text_embeddings = []
    log_probs = []
    for batch in tqdm(loader, disable=rank!=0, total=len(loader)):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.inference_mode():
            outputs = model(**batch, output_hidden_states=True)

        # (bsz, seq_len, hidden_size) -> (bsz, hidden_size)
        text_embedding = outputs['hidden_states'][-1].mean(1)
        # sum of output token log probs
        log_prob = -outputs['loss']

        text_embeddings.append(text_embedding.detach().cpu())
        log_probs.append(log_prob.detach().cpu())

    text_embeddings = torch.vstack(text_embeddings).to(torch.float32).numpy()
    log_probs = torch.vstack(log_probs).numpy()

    print(f'local_rank/global={local_rank}/{rank}: text_embeddings.shape = {text_embeddings.shape}, log_probs.shape = {log_probs.shape}, chunk_sizes = {train_dataset_chunk_sizes}')

    # if use_dist:
    #     text_embeddings = dist_gather_and_vstack_2d_tensors(
    #         text_embeddings, train_dataset_chunk_sizes, rank)
    #     log_probs = dist_gather_and_vstack_2d_tensors(
    #         log_probs, train_dataset_chunk_sizes, rank)

    if use_dist:
        save_path = os.path.join(save_dir, f'{dataset}_rank={rank}.pkl')
        with open(save_path, 'wb') as f:
            output = {'text_embeddings': text_embeddings,
                      'log_probs': log_probs}
            pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

    dist.barrier()

    if rank == 0:
        if use_dist:
            ## load and concat outputs from all ranks
            text_embeddings = []
            log_probs = []
            for r in range(world_size):
                save_path = os.path.join(save_dir, f'{dataset}_rank={r}.pkl')
                with open(save_path, 'rb') as f:
                    output = pickle.load(f)
                text_embeddings.append(output['text_embeddings'])
                log_probs.append(output['log_probs'])
                os.remove(save_path)
            text_embeddings = np.vstack(text_embeddings)
            log_probs = np.vstack(log_probs)

        output = {'text_embeddings': text_embeddings,
                  'log_probs': log_probs}
        if shuffle:
            output = {k: v[reverse_shuffle_inds] for k, v in output.items()}
        save_path = os.path.join(save_dir, ('test_' if test_run else '')+f'{dataset}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f'dataset: {dataset}')
        print(f'Save output={[(k, v.shape) for k, v in output.items()]} to {save_path}')


if __name__ == "__main__":

    """
    python note_llama_embeddings.py

    torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=localhost:29400 \
        note_llama_embeddings.py \
        --dataset lima \
        --model_name_or_path=../results/baselines/huggyllama/llama-7b \
        --save_dir=/gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/llama-7b_outputs \
    """

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="lima")
    parser.add_argument("--model_name_or_path", type=str, default="../results/baselines/huggyllama/llama-7b")
    parser.add_argument("--save_dir", type=str, default="/gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/llama-7b_outputs")
    parser.add_argument("--use_dist", action='store_true', default=False)
    parser.add_argument("--test_run", action='store_true', default=False)
    parser.add_argument("--shuffle", action='store_true', default=False)

    args = parser.parse_args()

    compute_lm_outputs(**vars(args))