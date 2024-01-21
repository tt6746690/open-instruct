#!/usr/bin/env python
# coding=utf-8
'''
This script is used to reformat the downloaded datasets into the format that can be used by the model.
Here we use jsonl for the converted data. Each line in the jsonl file is a json object formatted as follows:
{
    "dataset": "dataset_name",
    "id": "unique_id",
    "messages": [
        {"role": "system", "content": "message_text"}, # optional
        {"role": "user", "content": "message_text"},
        {"role": "assistant", "content": "message_text"},
        {"role": "user", "content": "message_text"},
        {"role": "assistant", "content": "message_text"},
        ...
    ],
}
'''

import json
import random
import re
import os
import glob
import pandas as pd
import argparse
from open_instruct.instruction_encode_templates import encode_instruction_example, encode_few_shot_example

import sys; sys.path.append('/gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts')
from note_pruning_analysis import filter_examples_by_numtoks, filter_json_by_numtoks

def convert_super_ni_data(data_dir, output_dir, zero_shot_examples_per_task=60, few_shot_examples_per_task=20, n_few_shot=2):
    os.makedirs(output_dir, exist_ok=True)
    train_tasks = []
    with open(os.path.join(data_dir, "splits", "xlingual", "train_tasks.txt"), "r") as fin:
        for line in fin:
            if not "_mmmlu_" in line:   # skip mmlu to avoid test leakage
                train_tasks.append(line.strip())
    with open(os.path.join(output_dir, "super_ni_data.jsonl"), "w") as fout:
        for task in train_tasks:
            with open(os.path.join(data_dir, "tasks", f"{task}.json"), "r") as fin:
                task_data = json.load(fin)
            instruction = task_data["Definition"][0]
            if zero_shot_examples_per_task + few_shot_examples_per_task < len(task_data["Instances"]):
                instances = random.sample(task_data["Instances"], k=zero_shot_examples_per_task+few_shot_examples_per_task)
            else:
                instances = task_data["Instances"]
            for instance in instances[:zero_shot_examples_per_task]:
                encoded_example = encode_instruction_example(
                    instruction=instruction, 
                    input=instance["input"], 
                    output=instance["output"][0],
                    random_template=True,
                    eos_token=None
                )
                fout.write(json.dumps({
                    "dataset": "super_ni",
                    "id": f"super_ni_{instance['id']}",
                    "messages": [
                        {"role": "user", "content": encoded_example["prompt"]},
                        {"role": "assistant", "content": encoded_example["completion"]},
                    ]
                }) + "\n")
            for instance in instances[zero_shot_examples_per_task:]:
                if n_few_shot < len(task_data["Positive Examples"]):
                    examplars = random.sample(task_data["Positive Examples"], k=n_few_shot)
                else:
                    examplars = task_data["Positive Examples"]
                encoded_example = encode_few_shot_example(
                    instruction=instruction,
                    examplars=examplars,
                    input=instance["input"],
                    output=instance["output"][0],
                    eos_token=None
                )
                fout.write(json.dumps({
                    "dataset": "super_ni",
                    "id": f"super_ni_{instance['id']}",
                    "messages": [
                        {"role": "user", "content": encoded_example["prompt"]},
                        {"role": "assistant", "content": encoded_example["completion"]},
                    ]
                }) + "\n")
            
            
def convert_cot_data(data_dir, output_dir, num_zero_shot_examples=50000, num_few_shot_examples=50000):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    if num_few_shot_examples > 0:
        with open(os.path.join(data_dir, "cot_zsopt.jsonl"), "r") as fin:
            zero_shot_examples = [json.loads(line) for line in fin]
            if num_zero_shot_examples < len(zero_shot_examples):
                zero_shot_examples = random.sample(zero_shot_examples, k=num_zero_shot_examples)
            examples.extend(zero_shot_examples)
    if num_few_shot_examples > 0:
        with open(os.path.join(data_dir, "cot_fsopt.jsonl"), "r") as fin:
            few_shot_examples = [json.loads(line) for line in fin]
            if num_few_shot_examples < len(few_shot_examples):
                few_shot_examples = random.sample(few_shot_examples, k=num_few_shot_examples)
            examples.extend(few_shot_examples)
    output_path = os.path.join(output_dir, "cot_data.jsonl")
    with open(output_path, "w") as fout:
        for idx, example in enumerate(examples):
            prompt = example["inputs"]
            if not prompt.endswith("\n") and not prompt.rstrip().endswith(":"):
                prompt += "\n"
            completion = example["targets"]
            fout.write(json.dumps({
                "dataset": "cot",
                "id": f"cot_{idx}",
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ]
            }) + "\n")
            

def convert_flan_v2_data(data_dir, output_dir, data_file="tulu_v1_resampled_flan_100k.jsonl", num_examples=None, max_seq_length=None, dataset_name="flan_v2"):
    """
        ```
        from open_instruct.reformat_datasets import convert_flan_v2_data
        data_dir = 'data/raw_train/flan_v2'
        output_dir = 'data/processed/flan_v2'
        convert_flan_v2_data(data_dir, output_dir, dataset_name='flan_v2')
        convert_flan_v2_data(data_dir, output_dir, num_examples=50_000, max_seq_length=2048, dataset_name='flan_v250k')
        ```
    """
    os.makedirs(output_dir, exist_ok=True)

    target_filename = f"{dataset_name}_data.jsonl"

    examples = []
    with open(os.path.join(data_dir, data_file), "r") as fin:
        for line in fin:
            examples.append(json.loads(line))
    if num_examples:
        random.seed(0)
        examples = random.sample(examples, k=int(num_examples*1.1))

    def convert_example_to_messages(example, idx):
        prompt = example["inputs"]
        if not prompt.endswith("\n") and not prompt.rstrip().endswith(":"):
            prompt += "\n"
        completion = example["targets"]
        return {
            "dataset": "flan_v2",
            "id": f"flan_v2_{idx}",
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ]
        }
    examples = [convert_example_to_messages(x, i) for i, x in enumerate(examples)]
    if max_seq_length is not None:
        examples = filter_examples_by_numtoks(examples, max_seq_length=max_seq_length)
    if num_examples is not None:
        if len(examples) >= num_examples:
            examples = examples[:num_examples]
        else:
            raise ValueError(f"Only {len(examples)} examples after filtering by max_seq_length=2048 but need {num_examples} examples.")

    output_path = os.path.join(output_dir, target_filename)
    with open(output_path, "w") as fout:
        for example in examples:
            fout.write(json.dumps(example) + "\n")



def convert_flan2022_data(data_dir, output_dir):
    """Same as `flan_v2` reformatting. """
    from datasets import load_dataset

    def convert_data_fn(example, idx):
        prompt = example['inputs']
        if not prompt.endswith("\n") and not prompt.rstrip().endswith(":"):
            prompt += "\n"
        completion = example["targets"]
        return {
            'dataset': "flan2022", 
            'id': f"flan2022_{idx}",
            'messages': [
                {'role': 'user', 'content': prompt},
                {"role": "assistant", "content": completion},
            ]}

    input_files = [
        # 'flan2022_1m.jsonl',
        # 'flan2022v2_1m.jsonl',
        'flan2022v1_1m.jsonl',
    ]
    for input_file in input_files:
        output_path = os.path.join(output_dir, input_file.split('.jsonl')[0]+'_data.jsonl')
        if os.path.isfile(output_path):
            continue

        input_path = os.path.join(data_dir, input_file)
        ds = load_dataset('json', data_files={'train': input_path}, split='train', cache_dir=data_dir)
        ds = ds.map(convert_data_fn, 
                    remove_columns=["inputs", "targets", "task"], 
                    with_indices=True,
                    num_proc=30,
                    desc=f'Convert data for {input_file}',
                    keep_in_memory=True)
        ds.to_json(output_path)


def convert_dolly_data(data_dir, output_dir, num_examples=None, max_seq_length=None, dataset_name="dolly"):
    """
        ```
        from open_instruct.reformat_datasets import convert_dolly_data
        data_dir = 'data/raw_train/dolly'
        output_dir = 'data/processed/dolly'
        convert_dolly_data(data_dir, output_dir, dataset_name='dolly')
        ```
    """
    os.makedirs(output_dir, exist_ok=True)
    target_filename = f"{dataset_name}_data.jsonl"

    examples = []
    with open(os.path.join(data_dir, "databricks-dolly-15k.jsonl"), "r") as fin:
        for line in fin:
            examples.append(json.loads(line))
    if num_examples:
        random.seed(0)
        examples = random.sample(examples, k=int(num_examples*1.1))
    def convert_example_to_messages(example, idx):
        encoded_example = encode_instruction_example(
            instruction=example["instruction"], 
            input=example["context"], 
            output=example["response"],
            random_template=True,
            eos_token=None
        )
        return {
            "dataset": "dolly",
            "id": f"dolly_{idx}",
            "messages": [
                {"role": "user", "content": encoded_example["prompt"]},
                {"role": "assistant", "content": encoded_example["completion"]},
            ]
        }
    examples = [convert_example_to_messages(x, i) for i, x in enumerate(examples)]
    if max_seq_length is not None:
        examples = filter_examples_by_numtoks(examples, max_seq_length=max_seq_length)
    if num_examples is not None:
        if len(examples) >= num_examples:
            examples = examples[:num_examples]
        else:
            raise ValueError(f"Only {len(examples)} examples after filtering by max_seq_length=2048 but need {num_examples} examples.")

    output_path = os.path.join(output_dir, target_filename)
    with open(output_path, "w") as fout:
        for example in examples:
            fout.write(json.dumps(example) + "\n")


def convert_self_instruct_data(data_dir, output_dir, num_examples=None, max_seq_length=None, dataset_name="self_instruct"):
    """
        ```
        from open_instruct.reformat_datasets import convert_self_instruct_data
        data_dir = 'data/raw_train/self_instruct'
        output_dir = 'data/processed/self_instruct'
        # convert_self_instruct_data(data_dir, output_dir, dataset_name='self_instruct')
        convert_self_instruct_data(data_dir, output_dir, num_examples=50_000, max_seq_length=2048, dataset_name='self_instruct50k')
        ```
    """
    os.makedirs(output_dir, exist_ok=True)
    target_filename = f"{dataset_name}_data.jsonl"

    examples = []
    with open(os.path.join(data_dir, "all_instances_82K.jsonl"), "r") as fin:
        for line in fin:
            examples.append(json.loads(line))
    if num_examples:
        random.seed(0)
        examples = random.sample(examples, k=int(num_examples*1.1))

    def convert_example_to_messages(example, idx):
        encoded_example = encode_instruction_example(
            instruction=example["instruction"], 
            input=example["input"], 
            output=example["output"],
            random_template=True,
            eos_token=None
        )
        return {
            "dataset": "self_instruct",
            "id": f"self_instruct_{idx}",
            "messages": [
                {"role": "user", "content": encoded_example["prompt"]},
                {"role": "assistant", "content": encoded_example["completion"]},
            ]
        }
    examples = [convert_example_to_messages(x, i) for i, x in enumerate(examples)]
    if max_seq_length is not None:
        examples = filter_examples_by_numtoks(examples, max_seq_length=max_seq_length)
    if num_examples is not None:
        if len(examples) >= num_examples:
            examples = examples[:num_examples]
        else:
            raise ValueError(f"Only {len(examples)} examples after filtering by max_seq_length=2048 but need {num_examples} examples.")

    output_path = os.path.join(output_dir, target_filename)
    with open(output_path, "w") as fout:
        for example in examples:
            fout.write(json.dumps(example) + "\n")


def convert_unnatural_instructions_data(data_dir, output_dir, num_examples=None):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    with open(os.path.join(data_dir, "core_data.jsonl"), "r") as fin:
        for line in fin:
            task_data = json.loads(line)
            instruction = task_data["instruction"]
            for instance in task_data["instances"]:
                if instance["constraints"] and instance["constraints"].lower() not in ["none", "none."]:
                    instance_instruction = instruction + "\n" + instance["constraints"]
                else:
                    instance_instruction = instruction
                encoded_example = encode_instruction_example(
                    instruction=instance_instruction,
                    input=instance["input"],
                    output=instance["output"],
                    random_template=True,
                    eos_token=None,
                )
                examples.append(encoded_example)
    if num_examples:
        examples = random.sample(examples, k=num_examples)
    with open((os.path.join(output_dir, "unnatural_instructions_data.jsonl")), "w") as fout:
        for idx, example in enumerate(examples):
            fout.write(json.dumps({
                "dataset": "unnatural_instructions",
                "id": f"unnatural_instructions_{idx}",
                "messages": [
                    {"role": "user", "content": example["prompt"]},
                    {"role": "assistant", "content": example["completion"]},
                ]
            }) + "\n")


def convert_stanford_alpaca_data(data_dir, output_dir, num_examples=None, max_seq_length=None, dataset_name="stanford_alpaca"):
    """
        ```
        from open_instruct.reformat_datasets import convert_stanford_alpaca_data
        data_dir = 'data/raw_train/stanford_alpaca'
        output_dir = 'data/processed/stanford_alpaca'
        # convert_stanford_alpaca_data(data_dir, output_dir, dataset_name='stanford_alpaca')
        convert_stanford_alpaca_data(data_dir, output_dir, num_examples=50_000, max_seq_length=2048, dataset_name='stanford_alpaca50k')
        ```
    """
    os.makedirs(output_dir, exist_ok=True)
    target_filename = f"{dataset_name}_data.jsonl"

    examples = []
    with open(os.path.join(data_dir, "alpaca_data.json"), "r") as fin:
        examples.extend(json.load(fin))
    if num_examples:
        random.seed(0)
        examples = random.sample(examples, k=min(int(num_examples*1.1), len(examples)))
    
    def convert_example_to_messages(example, idx):
        encoded_example = encode_instruction_example(
            instruction=example["instruction"], 
            input=example["input"], 
            output=example["output"],
            random_template=True,
            eos_token=None
        )
        return {
            "dataset": "stanford_alpaca",
            "id": f"stanford_alpaca_{idx}",
            "messages": [
                {"role": "user", "content": encoded_example["prompt"]},
                {"role": "assistant", "content": encoded_example["completion"]},
            ]
        }
    examples = [convert_example_to_messages(x, i) for i, x in enumerate(examples)]
    if max_seq_length is not None:
        examples = filter_examples_by_numtoks(examples, max_seq_length=max_seq_length)
    if num_examples is not None:
        if len(examples) >= num_examples:
            examples = examples[:num_examples]
        else:
            raise ValueError(f"Only {len(examples)} examples after filtering by max_seq_length=2048 but need {num_examples} examples.")
    
    output_path = os.path.join(output_dir, target_filename)
    with open(output_path, "w") as fout:
        for example in examples:
            fout.write(json.dumps(example) + "\n")


def convert_code_alpaca_data(data_dir, output_dir, num_examples=None):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    with open(os.path.join(data_dir, "code_alpaca_20k.json"), "r") as fin:
        examples.extend(json.load(fin))
    if num_examples:
        examples = random.sample(examples, k=num_examples)
    output_path = os.path.join(output_dir, "code_alpaca_data.jsonl")
    with open(output_path, "w") as fout:
        for idx, example in enumerate(examples):
            encoded_example = encode_instruction_example(
                instruction=example["instruction"], 
                input=example["input"], 
                output=example["output"],
                random_template=True,
                eos_token=None
            )
            fout.write(json.dumps({
                "dataset": "code_alpaca",
                "id": f"code_alpaca_{idx}",
                "messages": [
                    {"role": "user", "content": encoded_example["prompt"]},
                    {"role": "assistant", "content": encoded_example["completion"]},
                ]
            }) + "\n")


def convert_gpt4_alpaca_data(data_dir, output_dir, load_en=True, load_zh=False, num_examples=None, dataset_name="gpt4_alpaca", max_seq_length=None):
    """
        ```
        from open_instruct.reformat_datasets import convert_gpt4_alpaca_data
        data_dir = 'data/raw_train/gpt4_alpaca'
        output_dir = 'data/processed/gpt4_alpaca'
        # convert_gpt4_alpaca_data(data_dir, output_dir, dataset_name='gpt4_alpaca')
        convert_gpt4_alpaca_data(data_dir, output_dir, num_examples=50_000, max_seq_length=2048, dataset_name='gpt4_alpaca50k')
        ```
    """
    os.makedirs(output_dir, exist_ok=True)
    target_filename = f"{dataset_name}_data.jsonl"

    examples = []
    if load_en:
        with open(os.path.join(data_dir, "alpaca_gpt4_data.json"), "r") as fin:
            examples.extend(json.load(fin))
    if load_zh:
        with open(os.path.join(data_dir, "alpaca_gpt4_data_zh.json"), "r") as fin:
            examples.extend(json.load(fin))
    if num_examples:
        random.seed(0)
        examples = random.sample(examples, k=int(min(num_examples*1.1, len(examples))))
    def convert_example_to_messages(example, idx):
        encoded_example = encode_instruction_example(
            instruction=example["instruction"], 
            input=example["input"], 
            output=example["output"],
            random_template=True,
            eos_token=None
        )
        return {
            "dataset": "gpt4_alpaca",
            "id": f"gpt4_alpaca_{idx}",
            "messages": [
                {"role": "user", "content": encoded_example["prompt"]},
                {"role": "assistant", "content": encoded_example["completion"]},
            ]
        }
    examples = [convert_example_to_messages(x, i) for i, x in enumerate(examples)]
    if max_seq_length is not None:
        examples = filter_examples_by_numtoks(examples, max_seq_length=max_seq_length)
    if num_examples is not None:
        if len(examples) >= num_examples:
            examples = examples[:num_examples]
        else:
            raise ValueError(f"Only {len(examples)} examples after filtering by max_seq_length=2048 but need {num_examples} examples.")

    output_path = os.path.join(output_dir, target_filename)
    with open(output_path, "w") as fout:
        for example in examples:
            fout.write(json.dumps(example) + "\n")


def clean_starcoder_data(data_dir, filename, num_proc=16):
    """Some data cleaning."""

    if filename in []:
        return None

    from datasets import load_dataset
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('/gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/results/baselines/codellama/CodeLlama-7b-hf')

    ## take into account rathing threshold.
    match = re.search(r'rge(\d)', filename)
    if match:
        rating_threshold = int(match.group(1))
        input_filename = re.sub(r'_rge(\d)', '', filename)
    else:
        rating_threshold = None
        input_filename = filename

    ds = load_dataset('json', data_files={'train': os.path.join(data_dir, input_filename)}, 
             split='train', cache_dir=data_dir)

    def docstring_startswith_args(example):
        return example['docstring'][:10].lower().startswith('args')
    ds = ds.filter(lambda x: not docstring_startswith_args(x), num_proc=num_proc)
    def code_endsin_pass(example):
        return example['code'][-10:].lower().endswith('pass')
    ds = ds.filter(lambda x: not code_endsin_pass(x), num_proc=num_proc)
    def tokenize_fn(example):
        example.update({
            'instruction_'+k: v for k, v in 
            tokenizer(example['docstring'], truncation=False).items()
        })
        example.update({
            'output_'+k: v for k, v in 
            tokenizer(example['code'], truncation=False).items()
        })
        example.update({
            'numtoks_instruction': len(example['instruction_input_ids']),
            'numtoks_output': len(example['output_input_ids']),
        })
        return example
    ds = ds.map(tokenize_fn, num_proc=num_proc)
    def text_within_reasonable_range(example):
        """Removes
            - empty or uninformative instructions 
            - instruction, output pair that is too long.
            - functions that is not implemented.
        """
        return example['numtoks_instruction'] >= 5 and \
            example['numtoks_instruction'] + example['numtoks_output'] <= 2024 and \
            example['numtoks_output'] >= 40
    ds = ds.filter(text_within_reasonable_range, num_proc=num_proc)

    ## parse score from `generated_output`
    if filename.startswith('commentinstrv4'):
        assert('generated_output' in ds.column_names)
        def parse_score_fn(example):
            match = re.search(r'(Rating|[Ss]core):?\s?\n?(\d)', example['generated_output'])
            score = int(match.group(2)) if match else None
            if score is not None:
                score = 1 if score < 1 else score
                score = 5 if score > 5 else score
            if score not in [1, 2, 3, 4, 5]:
                score = None
            return {'llm_score': score}
        ds = ds.map(parse_score_fn, num_proc=num_proc)
        ds = ds.filter(lambda x: x['llm_score'] is not None)
        if rating_threshold is not None:
            ds = ds.filter(lambda x: x['llm_score'] >= rating_threshold)
    elif filename.startswith('commentinstrv5'):
        """Generated output contains back-translated instructions by deepseek coder
            - generation not terminated by <|EOT|>, need to remove chars after <|EOT|>
            - remove instructions that contains keywords like 'instruction'
        """
        assert('generated_output' in ds.column_names)
        def take_substr_before_eot_fn(example):
            s = example['generated_output']
            match = re.search(r'(<\|EOT\|>)', s)
            s = s[:match.start()].strip() if match else s
            return {'generated_output': s}
        ds = ds.map(take_substr_before_eot_fn, num_proc=num_proc)
        def find_keywords_in_generation_fn(example):
            return not any(x in example['generated_output'] for x in [
                'instruction',
            ])
        ds = ds.filter(find_keywords_in_generation_fn, num_proc=num_proc)
        ds = ds.remove_columns('docstring')
        ds = ds.rename_columns({'generated_output': 'docstring'})

    ds = ds.rename_columns({'docstring': 'instruction', 'code': 'output'})
    ds = ds.map(lambda _: {'input': ""}, num_proc=num_proc)
    columns = ['instruction', 'input', 'output']
    if 'llm_score' in ds.column_names:
        columns.append('llm_score')
    ds = ds.select_columns(columns)
    filename_cleaned = f"{filename.split('.json')[0]}_cleaned.jsonl"
    ds.to_json(os.path.join(data_dir, filename_cleaned))
    return filename_cleaned


def convert_starcoder_data(data_dir, output_dir):
    """
        ```
        data_dir = 'data/raw_train/starcoder'
        output_dir = 'data/processed/starcoder'
        ```

        python open_instruct/reformat_datasets.py --raw_data_dir data/raw_train --output_dir data/processed --dataset starcoder
    """
    filenames = [ 
        # 'commentinstr.json',
        # 'commentinstrv2_flppl.json',
        # 'commentinstrv2.json',
        # 'commentinstrv3.json',
        # 'commentinstrv4.json',
        # 'commentinstrv4_rge5.json',
        'commentinstrv5.json',
    ]
    # only keep cleaned data
    filenames = [clean_starcoder_data(data_dir, filename) for filename in filenames]
    filenames = [x for x in filenames if x is not None]
    print('cleaned filenames:', filenames)

    for filename in filenames:
        with open(os.path.join(data_dir, filename), 'r') as f:
            if filename.endswith('jsonl'):
                examples = [json.loads(l) for l in f]
            else:
                examples = json.load(f)

        output_path = os.path.join(output_dir, 'starcoder_'+filename.split('_cleaned.json')[0]+'.jsonl')
        with open(output_path, "w") as fout:
            for idx, example in enumerate(examples):
                if filename.startswith('commentinstr') and 'cleaned' not in filename:
                    instruction, input, output = example['docstring'], "", example['code']
                else:
                    instruction, input, output = example['instruction'], example['input'], example['output']
                encoded_example = encode_instruction_example(
                    instruction=instruction, 
                    input=input,
                    output=output,
                    random_template=True,
                    eos_token=None
                )
                fout.write(json.dumps({
                    "dataset": "starcoder",
                    "id": f"starcoder_{idx}",
                    "messages": [
                        {"role": "user", "content": encoded_example["prompt"]},
                        {"role": "assistant", "content": encoded_example["completion"]},
                    ]
                }) + "\n")
    
        ## wpq: to avoid truncate examples during training.
        print(f"Filtering {dataset} to max_seq_length=2048...")
        filter_json_by_numtoks(output_path, tokenizer_name='codellama', max_seq_length=2048)


def convert_sharegpt_data(data_dir, output_dir, data_file="sharegpt_html_cleaned_and_split_2048.json", num_examples=None, max_seq_length=None, dataset_name="sharegpt"):
    """
        python scripts/split_sharegpt_conversations.py \
            --in-files data/raw_train/sharegpt/sg_90k_part1_html_cleaned.json data/raw_train/sharegpt/sg_90k_part2_html_cleaned.json \
            --out-file data/raw_train/sharegpt/sharegpt_html_cleaned_and_split_2048_discardlongconv.json \
            --model-name-or-path results/baselines/huggyllama/llama-7b \
            --max-length 2048 \
            --special_tok_len 8

        set special_tok_len to 8 as max length of both user and assistant template
            ['_<s>', '▁<', '|', 'user', '|', '>', '<0x0A>']
            ['<0x0A>', '▁<', '|', 'ass', 'istant', '|', '>', '<0x0A>']

        ```
        from open_instruct.reformat_datasets import convert_sharegpt_data
        data_dir = 'data/raw_train/sharegpt'
        output_dir = 'data/processed/sharegpt'
        convert_sharegpt_data(data_dir, output_dir, data_file="sharegpt_html_cleaned_and_split_2048_discardlongconv.json", dataset_name="sharegptv2")
        convert_sharegpt_data(data_dir, output_dir, data_file="sharegpt_html_cleaned_and_split_2048_discardlongconv.json", num_examples=50_000, max_seq_length=2048, dataset_name="sharegpt50k")
        ```
    """
    os.makedirs(output_dir, exist_ok=True)
    target_filename = f'{dataset_name}_data.jsonl'

    examples = []
    with open(os.path.join(data_dir, data_file), "r") as fin:
        examples.extend(json.load(fin))
    if num_examples:
        random.seed(0)
        examples = random.sample(examples, k=int(num_examples*1.1))
    
    invalid_cnt = 0
    def convert_example_to_messages(example, idx):
        messages = []
        valid = True
        for message in example["conversations"]:
            if message["from"] == "human" or message["from"] == "user":
                messages.append({
                    "role": "user",
                    "content": message["value"]
                })
            elif message["from"] == "gpt" or message["from"] == "chatgpt":
                messages.append({
                    "role": "assistant",
                    "content": message["value"]
                })
            elif message["from"] == "system":
                valid = False
                invalid_cnt += 1
                break
            elif message["from"] == "bing":
                valid = False
                invalid_cnt += 1
                break
            else:
                raise ValueError(f"Unknown message sender: {message['from']}")
        if messages and valid:
            return {
                    "dataset": "sharegpt",
                    "id": f"sharegpt_{example['id']}",
                    "messages": messages
                }
        else:
            return None
    
    examples = [convert_example_to_messages(x, i) for i, x in enumerate(examples)]
    examples = [x for x in examples if x is not None]
    if max_seq_length is not None:
        examples = filter_examples_by_numtoks(examples, max_seq_length=max_seq_length)
    if num_examples is not None:
        if len(examples) >= num_examples:
            examples = examples[:num_examples]
        else:
            raise ValueError(f"Only {len(examples)} examples after filtering by max_seq_length=2048 but need {num_examples} examples.")

    output_path = os.path.join(output_dir, target_filename)
    with open(output_path, "w") as fout:
        for example in examples:
            fout.write(json.dumps(example) + "\n")

    if invalid_cnt > 0:
        print(f"# of invalid examples in sharegpt data: {invalid_cnt}")


def convert_baize_data(data_dir, output_dir, num_examples=None):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    for source in ["alpaca", "medical", "quora", "stackoverflow"]:
        with open(os.path.join(data_dir, f"{source}_chat_data.json"), "r") as fin:
            examples.extend(json.load(fin))
    if num_examples:
        random.seed(0)
        examples = random.sample(examples, k=num_examples)
    output_path = os.path.join(output_dir, "baize_data.jsonl")
    with open(output_path, "w") as fout:
        for idx, example in enumerate(examples):
            # split example["input"] by [|Human|] and [|AI|]
            messages = []
            rounds = example["input"].split("[|Human|]")[1:]
            for round in rounds:
                if not round.strip() or "[|AI|]" not in round:
                    continue
                human, assistant = round.split("[|AI|]")
                messages.append({
                    "role": "user",
                    "content": human.strip()
                })
                messages.append({
                    "role": "assistant",
                    "content": assistant.strip()
                })
            fout.write(json.dumps({
                "dataset": "baize",
                "id": f"baize_{idx}",
                "messages": messages
            }) + "\n")


def convert_oasst_data(data_dir, output_dir, top_k_reply=None, num_examples=None, max_seq_length=None, source_filename="2023-04-12_oasst_ready.trees.jsonl", dataset_name="oasst1"):
    '''
    For OASST1, because it's in a tree structure, where every user input might get multiple replies, 
    we have to save every path from the root node to the assistant reply (including both leaf node and intemediate node).
    This results in some of the messages being duplicated among different paths (instances).
    You can set top_k_reply to control how many replies to consider when traversing the tree, which will consider the replies with 
    the highest human-reviewed quality scores.

    Average/Max/Min number of child replies: 1.4414046396582094, 16, 0
    top_k_reply=1    -> 7k examples
    top_k_reply=None -> 33k examples

    ```
    from open_instruct.reformat_datasets import convert_oasst_data
    data_dir = 'data/raw_train/oasst1'
    output_dir = 'data/processed/oasst1'
    convert_oasst_data(data_dir, output_dir, top_k_reply=None, max_seq_length=2048, source_filename="2023-04-12_oasst_ready.trees.jsonl", dataset_name='oasst1')
    convert_oasst_data(data_dir, output_dir, top_k_reply=None, max_seq_length=2048, source_filename="2023-11-05_oasst2_ready.trees.jsonl", dataset_name='oasst2')
    ```
    '''
    os.makedirs(output_dir, exist_ok=True)
    target_filename = f"{dataset_name}_data.jsonl"

    conversations = []
    with open(os.path.join(data_dir, source_filename), "r") as fin:
        for line in fin:
            conversations.append(json.loads(line))

    output_path = os.path.join(output_dir, target_filename)

    child_replies_len = []

    # tranvers the conversation tree, and collect all valid sequences
    def dfs(reply, messages, valid_sequences):
        nonlocal child_replies_len
        if reply["deleted"]:
            return
        if reply["role"] == "assistant":
            messages.append(
                {"role": "assistant", "content": reply["text"]}
            )
            if not reply["replies"]:  # leaf node
                valid_sequences.append(messages[:])
            else:
                child_replies = [child for child in reply["replies"] if not child["deleted"]]
                for child in child_replies:
                    if not "quality" in child["labels"]:
                        child["labels"]["quality"] = {
                            "value": 0.0,
                            "count": 0,
                        }
                child_replies_len += [len(child_replies)]
                child_replies = child_replies if top_k_reply is None else sorted(child_replies, key=lambda x: x["labels"]["quality"]["value"], reverse=True)[:top_k_reply]
                for child in child_replies:
                    dfs(child, messages, valid_sequences)
            messages.pop()
        elif reply["role"] == "prompter":
            messages.append(
                {"role": "user", "content": reply["text"]}
            )
            child_replies = [child for child in reply["replies"] if not child["deleted"]]
            for child in child_replies:
                if not "quality" in child["labels"]:
                    child["labels"]["quality"] = {
                        "value": 0.0,
                        "count": 0,
                    }
            child_replies_len += [len(child_replies)]
            child_replies = child_replies if top_k_reply is None else sorted(child_replies, key=lambda x: x["labels"]["quality"]["value"], reverse=True)[:top_k_reply]
            for child in child_replies:
                dfs(child, messages, valid_sequences)
            messages.pop()
        else:
            raise ValueError(f"Unknown role: {reply['role']}")
        
    example_cnt = 0
    examples = []
    for conversation in conversations:
        valid_sequences = []
        dfs(conversation["prompt"], [], valid_sequences)
        for sequence in valid_sequences:
            examples.append({
                "dataset": dataset_name,
                "id": f"{dataset_name}_{example_cnt}",
                "messages": sequence
            })
            example_cnt += 1

    if max_seq_length is not None:
        examples = filter_examples_by_numtoks(examples, max_seq_length=max_seq_length)
    if num_examples:
        random.seed(0)
        examples = random.sample(examples, k=num_examples)
    if num_examples is not None:
        if len(examples) >= num_examples:
            examples = examples[:num_examples]
        else:
            raise ValueError(f"Only {len(examples)} examples after filtering by max_seq_length=2048 but need {num_examples} examples.")
        
    with open(output_path, "w") as fout:
        for example in examples:
            fout.write(json.dumps(example) + "\n")



def convert_lima_data(data_dir, output_dir, num_examples=None):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    with open(os.path.join(data_dir, "train.jsonl"), "r") as fin:
        for line in fin:
            examples.append(json.loads(line))
    if num_examples:
        examples = random.sample(examples, k=num_examples)
    output_path = os.path.join(output_dir, "lima_data.jsonl")
    with open(output_path, "w") as fout:
        for idx, example in enumerate(examples):
            messages = []
            if not len(example["conversations"]) % 2 == 0:
                print(f"Waring: example {idx} in LIMA has odd number of messages. Cutting off the last message.")
                example["conversations"] = example["conversations"][:-1]
            
            for i in range(0, len(example["conversations"]), 2):
                messages.append({
                    "role": "user",
                    "content": example["conversations"][i]
                })
                messages.append({
                    "role": "assistant",
                    "content": example["conversations"][i+1]
                })
            fout.write(json.dumps({
                "dataset": "lima",
                "id": f"lima_{idx}",
                "messages": messages,
            }) + "\n")



def convert_wizardlm_data(data_dir, output_dir, num_examples=30000, max_seq_length=None, dataset_name='wizardlm'):
    """
        ```
        from note_pruning_analysis import filter_json_by_numtoks
        from open_instruct.reformat_datasets import convert_wizardlm_data
        data_dir = 'data/raw_train/wizardlm'
        output_dir = 'data/processed/wizardlm'
        dataset = 'wizardlmv2'
        convert_wizardlm_data(data_dir, output_dir, num_examples=None, version=dataset)
        print(f"Filtering {dataset} to max_seq_length=2048...")
        filepath = os.path.join(output_dir, f"{dataset}_data.jsonl")
        filter_json_by_numtoks(filepath, max_seq_length=2048)
        ```
        wizardlm: original dataset
        wizardlmv2: truncate to 2048 max tokens.
    """
    os.makedirs(output_dir, exist_ok=True)
    target_filename = f"{dataset_name}_data.jsonl"

    examples = []
    with open(os.path.join(data_dir, "WizardLM_evol_instruct_V2_143k.json"), "r") as fin:
        examples = json.load(fin)
    if num_examples:
        random.seed(0)
        examples = random.sample(examples, k=int(num_examples*1.1))

    def convert_example_to_messages(example, idx):
        messages = []
        assert len(example["conversations"]) % 2 == 0
        for i in range(0, len(example["conversations"]), 2):
            assert example["conversations"][i]["from"] == "human"
            assert example["conversations"][i+1]["from"] == "gpt"
            messages.append({
                "role": "user",
                "content": example["conversations"][i]["value"]
            })
            messages.append({
                "role": "assistant",
                "content": example["conversations"][i+1]["value"]
            })
        return {
            "dataset": "wizardlm",
            "id": f"wizardlm_{example['idx']}",
            "messages": messages,
        }
    examples = [convert_example_to_messages(x, i) for i, x in enumerate(examples)]
    if max_seq_length is not None:
        examples = filter_examples_by_numtoks(examples, max_seq_length=max_seq_length)
    if num_examples is not None:
        if len(examples) >= num_examples:
            examples = examples[:num_examples]
        else:
            raise ValueError(f"Only {len(examples)} examples after filtering by max_seq_length=2048 but need {num_examples} examples.")

    output_path = os.path.join(output_dir, target_filename)
    with open(output_path, "w") as fout:
        for example in examples:
            fout.write(json.dumps(example) + "\n")



def convert_alpagasus_data(data_dir, output_dir, num_examples=30000):
    """
        ```
        import os
        from note_pruning_analysis import filter_json_by_numtoks
        from open_instruct.reformat_datasets import convert_alpagasus_data
        
        data_dir = 'data/raw_train/alpagasus'
        output_dir = 'data/processed/alpagasus'
        convert_alpagasus_data(data_dir, output_dir, num_examples=None)
        filepath = os.path.join(output_dir, "chatgpt_9k.jsonl")
        print(f"Filtering alpagasus to max_seq_length=2048...")
        filter_json_by_numtoks(filepath, max_seq_length=2048)
        ```
    """
    os.makedirs(output_dir, exist_ok=True)

    examples = []
    with open(os.path.join(data_dir, "chatgpt_9k.json"), "r") as fin:
        examples = json.load(fin)
    if num_examples:
        examples = random.sample(examples, k=num_examples)

    output_path = os.path.join(output_dir, "chatgpt_9k.jsonl")
    
    with open(output_path, "w") as fout:
        for idx, example in enumerate(examples):
            encoded_example = encode_instruction_example(
                instruction=example["instruction"], 
                input=example["input"], 
                output=example["output"],
                random_template=True,
                eos_token=None
            )
            fout.write(json.dumps({
                "dataset": "alpagasus",
                "id": f"alpagasus_{idx}",
                "messages": [
                    {"role": "user", "content": encoded_example["prompt"]},
                    {"role": "assistant", "content": encoded_example["completion"]},
                ]
            }) + "\n")


def convert_HelpSteer_data(data_dir, output_dir, num_examples=30000):
    """
        ```
        import os
        from note_pruning_analysis import filter_json_by_numtoks
        from open_instruct.reformat_datasets import convert_HelpSteer_data
        
        data_dir = 'data/raw_train/HelpSteer'
        output_dir = 'data/processed/HelpSteer'
        convert_HelpSteer_data(data_dir, output_dir, num_examples=None)
        filepath = os.path.join(output_dir, "HelpSteer_train.jsonl")
        print(f"Filtering HelpSteer to max_seq_length=2048...")
        filter_json_by_numtoks(filepath, max_seq_length=2048)
        ```
    """
    os.makedirs(output_dir, exist_ok=True)
    dataset = 'HelpSteer'

    source_filename = 'HelpSteer_train.parquet'
    target_filename = f'{dataset}_data.jsonl'

    df = pd.read_parquet(os.path.join(data_dir, source_filename))
    if num_examples is not None:
        df = df.sample(n=int(num_examples*1.1), random_state=0)
    examples = [row.to_dict() for _, row in df.iterrows()]
    if num_examples:
        examples = random.sample(examples, k=num_examples)

    output_path = os.path.join(output_dir, "HelpSteer_train.jsonl")
    
    with open(output_path, "w") as fout:
        for idx, example in enumerate(examples):

            fout.write(json.dumps({
                "dataset": "HelpSteer",
                "id": f"HelpSteer_{idx}",
                "messages": [
                    {"role": "user", "content": example["prompt"]},
                    {"role": "assistant", "content": example["response"]},
                ]
            }) + "\n")
    

def convert_orca_dpo_pairs_data(data_dir, output_dir, num_examples=None):
    """
        ```
        #### Including system prompt in the begining of the user prompt,
        #### seperate by '\n\n' with user question
        from open_instruct.reformat_datasets import convert_orca_dpo_pairs_data
        data_dir = 'data/raw_train/orca_dpo_pairs'
        output_dir = 'data/processed/orca_dpo_pairs'
        convert_orca_dpo_pairs_data(data_dir, output_dir, num_examples=None)
        ```
    """

    os.makedirs(output_dir, exist_ok=True)
    dataset = 'orca_dpo_pairs'

    source_filename = 'orca_dpo_pairs_train.parquet'
    target_filename = f'{dataset}_data.jsonl'

    df = pd.read_parquet(os.path.join(data_dir, source_filename))
    if num_examples is not None:
        df = df.sample(n=int(num_examples*1.1), random_state=0)
    examples = [row.to_dict() for _, row in df.iterrows()]
    
    def convert_example_to_messages(example, index):        
        chosen = [{'role': 'user', 'content': f"System Prompt: {example['system']}\n\n" + example['question']},
                  {'role': 'assistant', 'content': example['chosen']}]
        rejected = [{'role': 'user', 'content': f"System Prompt: {example['system']}\n\n" + example['question']},
                  {'role': 'assistant', 'content': example['rejected']}]
        assert(len(chosen) % 2 == 0 and len(rejected) % 2 == 0)
        example = {
            'dataset': dataset,
            'id': f'{dataset}_{index}',
            'chosen': chosen,
            'rejected': rejected,
        }
        return example
    examples = [convert_example_to_messages(x, i) for i, x in enumerate(examples)]
    examples = filter_examples_by_numtoks(examples, max_seq_length=2048)
    if num_examples is not None:
        if len(examples) >= num_examples:
            examples = examples[:num_examples]
        else:
            raise ValueError(f"Only {len(examples)} examples after filtering by max_seq_length=2048 but need {num_examples} examples.")

    output_path = os.path.join(output_dir, target_filename)
    with open(output_path, 'w') as fout:
        for example in examples:
            fout.write(json.dumps(example) + "\n")



def convert_deita_data(data_dir, output_dir, data_file="deita_6k_cleaned_and_split_2048.json", num_examples=None, dataset_name="deita_sharegpt"):
    """
        ## first convert the deita file into a proper json file like sharegpt.json
        import json
        for the_k in ['6k', '10k']:
            file_path = f'data/raw_train/DEITA{the_k}/deita_{the_k}.json'
            parsed_json_list = []
            
            # Reading and parsing the file
            with open(file_path, 'r') as file:
                for line in file:
                    # Each line is a separate JSON object
                    json_object = json.loads(line)
                    parsed_json_list.append(json_object)
    
            # Writing the list back to the file as JSON
            with open(file_path, 'w') as file:
                json.dump(parsed_json_list, file, indent=4)

        ## Next split, repeat for sharegpt.json
        python scripts/split_sharegpt_conversations.py \
            --in-files data/raw_train/DEITA6k/deita_6k.json \
            --out-file data/raw_train/DEITA6k/deita_6k_cleaned_and_split_2048_discardlongconv.json \
            --model-name-or-path results/baselines/huggyllama/llama-7b \
            --max-length 2048 \
            --special_tok_len 8

        python scripts/split_sharegpt_conversations.py \
            --in-files data/raw_train/DEITA10k/deita_10k.json \
            --out-file data/raw_train/DEITA10k/deita_10k_cleaned_and_split_2048_discardlongconv.json \
            --model-name-or-path results/baselines/huggyllama/llama-7b \
            --max-length 2048 \
            --special_tok_len 8

        set special_tok_len to 8 as max length of both user and assistant template
            ['_<s>', '▁<', '|', 'user', '|', '>', '<0x0A>']
            ['<0x0A>', '▁<', '|', 'ass', 'istant', '|', '>', '<0x0A>']

        ```
        from open_instruct.reformat_datasets import convert_deita_data

        data_dir = 'data/raw_train/DEITA6k'
        output_dir = 'data/processed/DEITA6k'
        convert_deita_data(data_dir, output_dir, data_file="deita_6k_cleaned_and_split_2048_discardlongconv.json", dataset_name="deita_sharegpt")
        
        data_dir = 'data/raw_train/DEITA10k'
        output_dir = 'data/processed/DEITA10k'
        convert_deita_data(data_dir, output_dir, data_file="deita_10k_cleaned_and_split_2048_discardlongconv.json", dataset_name="deita_sharegpt")
        ```
 
    """
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    with open(os.path.join(data_dir, data_file), "r") as fin:
        examples.extend(json.load(fin))
    if num_examples:
        examples = random.sample(examples, k=num_examples)

    output_path = os.path.join(output_dir, f"{dataset_name}_data.jsonl")
    with open(output_path, "w") as fout:
        invalid_cnt = 0
        for idx, example in enumerate(examples):
            messages = []
            valid = True
            for message in example["conversations"]:
                if message["from"] == "human" or message["from"] == "user":
                    messages.append({
                        "role": "user",
                        "content": message["value"]
                    })
                elif message["from"] == "gpt" or message["from"] == "chatgpt":
                    messages.append({
                        "role": "assistant",
                        "content": message["value"]
                    })
                elif message["from"] == "system":
                    valid = False
                    invalid_cnt += 1
                    break
                elif message["from"] == "bing":
                    valid = False
                    invalid_cnt += 1
                    break
                else:
                    raise ValueError(f"Unknown message sender: {message['from']}")
            if messages and valid:
                fout.write(json.dumps({
                    "dataset": dataset_name,
                    "id": f"sharegpt_{example['id']}",
                    "messages": messages
                }) + "\n")
        if invalid_cnt > 0:
            print(f"# of invalid examples in sharegpt data: {invalid_cnt}")
            



def convert_shp_data(data_dir, output_dir, num_examples=None, max_seq_length=None):
    """
        ```
        from open_instruct.reformat_datasets import convert_shp_data
        data_dir = 'data/raw_train/shp'
        output_dir = 'data/processed/shp'
        convert_shp_data(data_dir, output_dir, num_examples=50_000, max_seq_length=2048)
        ```
    """
    os.makedirs(output_dir, exist_ok=True)
    dataset = 'shp'

    source_filenames = [
        'shp_train_0.parquet',
        'shp_train_1.parquet'
    ]
    target_filename = 'shp_data.jsonl'

    dfs = []
    for x in source_filenames:
        df = pd.read_parquet(os.path.join(data_dir, x))
        dfs.append(df)
    df = pd.concat(dfs, axis=0)
    if num_examples is not None:
        df = df.sample(n=int(num_examples*1.1), random_state=0)
    examples = [row.to_dict() for _, row in df.iterrows()]

    def convert_example_to_messages(example):
        answer_chosen = example['human_ref_A'] if example['labels'] == 1 else example['human_ref_B']
        answer_rejected = example['human_ref_A'] if example['labels'] == 0 else example['human_ref_B']
        return {
            'dataset': dataset,
            'id': f"{dataset}_{example['post_id']}",
            'chosen': [
                {'role': 'user', 'content': example['history']},
                {'role': 'assistant', 'content': answer_chosen},
            ],
            'rejected': [
                {'role': 'user', 'content': example['history']},
                {'role': 'assistant', 'content': answer_rejected},
            ],
        }
    examples = [convert_example_to_messages(x) for x in examples]
    if max_seq_length is not None:
        examples = filter_examples_by_numtoks(examples, max_seq_length=max_seq_length)
    if num_examples is not None:
        if len(examples) >= num_examples:
            examples = examples[:num_examples]
        else:
            raise ValueError(f"Only {len(examples)} examples after filtering by max_seq_length=2048 but need {num_examples} examples.")

    output_path = os.path.join(output_dir, target_filename)
    with open(output_path, 'w') as fout:
        for example in examples:
            fout.write(json.dumps(example) + "\n")



def convert_hh_rlhf_data(data_dir, output_dir, num_examples=None, max_seq_length=None):
    """
        ```
        from open_instruct.reformat_datasets import convert_hh_rlhf_data
        data_dir = 'data/raw_train/hh_rlhf'
        output_dir = 'data/processed/hh_rlhf'
        convert_hh_rlhf_data(data_dir, output_dir, num_examples=50_000)
        ```
    """

    os.makedirs(output_dir, exist_ok=True)
    dataset = 'hh_rlhf'

    source_filename = 'hh_rlhf_train.parquet'
    target_filename = f'{dataset}_data.jsonl'

    df = pd.read_parquet(os.path.join(data_dir, source_filename))
    if num_examples is not None:
        df = df.sample(n=int(num_examples*1.1), random_state=0)
    examples = [row.to_dict() for _, row in df.iterrows()]
    roles = {'human': 'user', 'assistant': 'assistant'}
    def convert_example_to_messages(example, index):
        chosen = [{'role': roles[x['role']], 'content': x['text']} 
                    for x in example['context'].tolist() + [example['chosen']]]
        rejected = [{'role': roles[x['role']], 'content': x['text']} 
                    for x in example['context'].tolist() + [example['rejected']]]
        assert(len(chosen) % 2 == 0 and len(rejected) % 2 == 0)
        example = {
            'dataset': dataset,
            'id': f'{dataset}_{index}',
            'chosen': chosen,
            'rejected': rejected,
        }
        return example
    examples = [convert_example_to_messages(x, i) for i, x in enumerate(examples)]
    if max_seq_length is not None:
        examples = filter_examples_by_numtoks(examples, max_seq_length=max_seq_length)
    if num_examples is not None:
        if len(examples) >= num_examples:
            examples = examples[:num_examples]
        else:
            raise ValueError(f"Only {len(examples)} examples after filtering by max_seq_length=2048 but need {num_examples} examples.")

    output_path = os.path.join(output_dir, target_filename)
    with open(output_path, 'w') as fout:
        for example in examples:
            fout.write(json.dumps(example) + "\n")


def get_unique_ids_subset(L):
    random.seed(0)
    ids = set()
    S = []
    random.shuffle(L)
    for x in L:
        xid = x["info"]["id"]
        if xid not in ids and xid is not None:
            ids.add(xid)
            S.append(x)
    return S



def convert_openai_sum_data(data_dir, output_dir, num_examples=None, max_seq_length=None):
    """
        ```
        from open_instruct.reformat_datasets import convert_openai_sum_data
        data_dir = 'data/raw_train/openai_sum'
        output_dir = 'data/processed/openai_sum'
        convert_openai_sum_data(data_dir, output_dir, num_examples=50_000)
        ```
    """
    os.makedirs(output_dir, exist_ok=True)
    dataset = 'openai_sum'

    source_filename = 'openai_summarize_from_feedback_train.parquet'
    target_filename = 'openai_sum_data.jsonl'

    df = pd.read_parquet(os.path.join(data_dir, source_filename))
    if num_examples is not None:
        df = df.sample(n=int(num_examples*1.1), random_state=0)
    examples = [row.to_dict() for _, row in df.iterrows()]
    def convert_example_to_messages(example):
        post = example["info"]["post"]
        choice = example["choice"]
        answer_chosen = example["summaries"][choice]["text"]
        answer_rejected = example["summaries"][1-choice]["text"]
        return {
            "dataset": dataset,
            "id": dataset+'_'+example["info"]["id"],
            "chosen": [
                {"role": "user", "content": post},
                {"role": "assistant", "content": answer_chosen},
            ],
            "rejected": [
                {"role": "user", "content": post},
                {"role": "assistant", "content": answer_rejected},
            ]
        }
    examples = [convert_example_to_messages(x) for x in examples]
    if max_seq_length is not None:
        examples = filter_examples_by_numtoks(examples, max_seq_length=max_seq_length)
    if num_examples is not None:
        if len(examples) >= num_examples:
            examples = examples[:num_examples]
        else:
            raise ValueError(f"Only {len(examples)} examples after filtering by max_seq_length=2048 but need {num_examples} examples.")

    output_path = os.path.join(output_dir, target_filename)
    with open(output_path, 'w') as fout:
        for idx, example in enumerate(examples):
            fout.write(json.dumps(example) + "\n")
    


def convert_ultrafeedback_data(data_dir, output_dir, num_examples=None, max_seq_length=None):
    """Currently use cleaned version from allenai
            ultrafeedback: allenai/ultrafeedback_binarized_cleaned

        ```
        from open_instruct.reformat_datasets import convert_ultrafeedback_data
        data_dir = 'data/raw_train/ultrafeedback'
        output_dir = 'data/processed/ultrafeedback'
        convert_ultrafeedback_data(data_dir, output_dir, num_examples=50_000)
        ```
    """
    os.makedirs(output_dir, exist_ok=True)
    dataset = 'ultrafeedback'

    source_filename = 'allenai_ultrafeedback_binarized_cleaned_train_prefs.parquet'
    target_filename = f'{dataset}_data.jsonl'

    df = pd.read_parquet(os.path.join(data_dir, source_filename))
    if num_examples is not None:
        df = df.sample(n=int(num_examples*1.1), random_state=0)
    examples = [row.to_dict() for _, row in df.iterrows()]
    def convert_example_to_messages(example, index):
        return {
            'dataset': dataset,
            'id': f'{dataset}_{index}',
            'chosen': example['chosen'].tolist(),
            'rejected': example['rejected'].tolist(),
            'score_chosen': example['score_chosen'],
            'score_rejected': example['score_rejected'],
            'source': example['source'],
        }
    examples = [convert_example_to_messages(x, i) for i, x in enumerate(examples)]
    if max_seq_length is not None:
        examples = filter_examples_by_numtoks(examples, max_seq_length=max_seq_length)
    if num_examples is not None:
        if len(examples) >= num_examples:
            examples = examples[:num_examples]
        else:
            raise ValueError(f"Only {len(examples)} examples after filtering by max_seq_length=2048 but need {num_examples} examples.")

    output_path = os.path.join(output_dir, target_filename)
    with open(output_path, 'w') as fout:
        for example in examples:
            fout.write(json.dumps(example) + "\n")



def convert_open_orca_data(data_dir, output_dir, num_gpt4_examples=30000, num_gpt35_examples=0, version='open_orca'):
    """
        ```
        from open_instruct.reformat_datasets import convert_open_orca_data
        data_dir = 'data/raw_train/open_orca'
        output_dir = 'data/processed/open_orca'
        convert_open_orca_data(data_dir, output_dir, version='open_orca_slim')
        ```
    """
    os.makedirs(output_dir, exist_ok=True)

    if version == 'open_orca':
        examples = []

        df = pd.read_parquet(os.path.join(data_dir, "1M-GPT4-Augmented.parquet"))    
        gpt4_examples = [row.to_dict() for _, row in df.iterrows()]
        random.shuffle(gpt4_examples)
        examples.extend(gpt4_examples[:num_gpt4_examples])

        df = pd.read_parquet(os.path.join(data_dir, "3_5M-GPT3_5-Augmented.parquet"))
        gpt35_examples = [row.to_dict() for _, row in df.iterrows()]
        random.shuffle(gpt35_examples)
        examples.extend(gpt35_examples[:num_gpt35_examples])

        output_path = os.path.join(output_dir, "open_orca_data.jsonl")
        with open(output_path, "w") as fout:
            for idx, example in enumerate(examples):
                messages = [
                    ## wpq: for now don't include system prompt
                    # {"role": "system", "content": example["system_prompt"]},
                    {"role": "user", "content": example["question"]},
                    {"role": "assistant", "content": example["response"]}
                ]
                fout.write(json.dumps({
                    "dataset": "open_orca",
                    "id": f"open_orca_{example['id']}",
                    "messages": messages,
                }) + "\n")

    elif version == 'open_orca_slim':
        ## wpq: open_orca_slim
        examples = [] 
        with open(os.path.join(data_dir, "oo-labeled_correct.gpt4.sharegpt.jsonl"), "r") as f:
            for line in f:
                examples.append(json.loads(line))

        output_path = os.path.join(output_dir, "open_orca_slim_data.jsonl")
        with open(output_path, "w") as fout:
            for idx, example in enumerate(examples):
                messages = []
                for message in example["conversations"]:
                    if message["from"] == "system":
                        ## wpq: for now don't include system prompt
                        # messages.append({
                        #     "role": "system",
                        #     "content": message["value"] 
                        # })
                        pass
                    elif message["from"] == "human":
                        messages.append({
                            "role": "user",
                            "content": message["value"]
                        })
                    elif message["from"] == "gpt":
                        messages.append({
                            "role": "assistant",
                            "content": message["value"]
                        })
                    else:
                        raise ValueError(f"Unknown message sender: {message['from']}")
                if messages:
                    fout.write(json.dumps({
                        'dataset': 'open_orca_slim',
                        'id': f"open_orca_slim_{idx}",
                        'messages': messages,
                    }) + "\n")




def convert_ultrachat_data(data_dir, output_dir, num_examples=None, max_seq_length=None, dataset_name='ultrachat200k'):
    """

        ultrachat200k: originnal dataset split long conv, without discarding long conv examples properly.
        ultrachat200kv2: split & discard long conv! 
            - since discard long conv, don't need to filter examples to <2048 tokens.
        
            python split_sharegpt_conversations.py \
                --in-files HuggingFaceH4/ultrachat_200k_train \
                --out-file data/raw_train/ultrachat/ultrachat200k_train_2048_discardlongconv.json \
                --model-name-or-path results/baselines/huggyllama/llama-7b \
                --max-length 2048 \
                --special_tok_len 8

            python split_sharegpt_conversations.py \
                --in-files HuggingFaceH4/ultrachat_200k_test \
                --out-file data/raw_train/ultrachat/ultrachat200k_test_2048_discardlongconv.json \
                --model-name-or-path results/baselines/huggyllama/llama-7b \
                --max-length 2048 \
                --special_tok_len 8
        ```
        from open_instruct.reformat_datasets import convert_ultrachat_data
        data_dir = 'data/raw_train/ultrachat'
        output_dir = 'data/processed/ultrachat'
        convert_ultrachat_data(data_dir, output_dir, max_seq_length=2048, dataset_name='ultrachat200kv2')
        convert_ultrachat_data(data_dir, output_dir, num_examples=50_000, max_seq_length=2048, dataset_name='ultrachat50k')
        ```
    
    """
    from datasets import load_dataset, concatenate_datasets

    os.makedirs(output_dir, exist_ok=True)

    if not dataset_name.startswith('ultrachat15'): # subsample from 200k data
        if num_examples is None:
            if dataset_name.startswith('ultrachat200k'):
                data_files = {'train': os.path.join(data_dir, 'ultrachat_200k_train_splitlongconv.json'),
                              'test': os.path.join(data_dir, 'ultrachat_200k_test_splitlongconv.json')}
            else:
                data_files = {'train': os.path.join(data_dir, 'ultrachat200k_train_2048_discardlongconv.json'),
                              'test': os.path.join(data_dir, 'ultrachat200k_test_2048_discardlongconv.json')}
        else:
            data_files = {'train': os.path.join(data_dir, 'ultrachat200k_train_2048_discardlongconv.json'),
                          'test': os.path.join(data_dir, 'ultrachat200k_test_2048_discardlongconv.json')}
            
        for split in ['train', 'test']:
            ds = load_dataset('json', data_files=data_files, split=split)
            # splitting conversations removes "prompt" already.
            ds = ds.remove_columns(["prompt_id"])
            def add_metadata_fn(example, idx):
                example.update({'dataset': 'ultrachat', 'id': f'ultrachat_{idx}'})
                return example
            ds = ds.map(add_metadata_fn, 
                        with_indices=True, 
                        num_proc=10)
            # re-ordering the features
            ds =  ds.select_columns(['dataset', 'id']).add_column('messages', ds['messages'])

            examples = [ds[i] for i in range(len(ds))]

            if split == 'train':
                if num_examples:
                    random.seed(0)
                    examples = random.sample(examples, k=int(num_examples*1.1))

                if max_seq_length is not None:
                        examples = filter_examples_by_numtoks(examples, max_seq_length=max_seq_length)
                if num_examples is not None:
                    if len(examples) >= num_examples:
                        examples = examples[:num_examples]
                    else:
                        raise ValueError(f"Only {len(examples)} examples after filtering by max_seq_length=2048 but need {num_examples} examples.")

            output_path = os.path.join(output_dir, f'{dataset_name}_{split}_data.jsonl')
            with open(output_path, 'w') as fout:
                for idx, example in enumerate(examples):
                    fout.write(json.dumps(example) + "\n")

    else: # ultrachat15
        data_files = {f'train_{i}': os.path.join(data_dir, 'full_splitlongconv_2048', f'train_{i}.jsonl') 
                      for i in range(10)}
        ds = load_dataset('json', data_files=data_files)
        ds = concatenate_datasets([ds[f'train_{i}'] for i in range(10)])
        def add_metadata_fn(example, idx):
            example.update({'dataset': 'ultrachat', 'id': f'ultrachat_{idx}'})
            return example
        ds = ds.map(add_metadata_fn, with_indices=True, num_proc=10)
        # too memory/compute intense.
        # ds = ds.select_columns(['dataset', 'id']).add_column('messages', ds['messages'])

        save_path = os.path.join(output_dir, f'{dataset_name}_data.jsonl')
        ds.to_json(save_path)

        num_shards = 10
        for i in range(num_shards):
            ds_shard = ds.shard(num_shards, i, contiguous=True)
            save_path = os.path.join(output_dir, f'ultrachat15_{i}_data.jsonl')
            ds_shard.to_json(save_path)


def convert_hard_coded_data(data_dir, output_dir, repeat=1):
    os.makedirs(output_dir, exist_ok=True)
    data = pd.read_excel(os.path.join(data_dir, "hard_coded_examples.xlsx"), header=0)
    output_path = os.path.join(output_dir, "hard_coded_data.jsonl")
    with open(output_path, "w") as fout:
        for _ in range(repeat):
            for idx, row in data.iterrows():
                fout.write(json.dumps({
                    "dataset": "hard_coded",
                    "id": f"hard_coded_{idx}",
                    "messages": [
                        {"role": "user", "content": row["Prompt"]},
                        {"role": "assistant", "content": row["Output"]}
                    ]
                }) + "\n")


def convert_science_data(data_dir, output_dir, num_examples=None):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    with open(os.path.join(data_dir, "science_train.jsonl"), "r") as fin:
        for line in fin:
            examples.append(json.loads(line))
    if num_examples:
        examples = random.sample(examples, k=num_examples)
    output_path = os.path.join(output_dir, "science_data.jsonl")
    with open(output_path, "w") as fout:
        for idx, example in enumerate(examples):
            fout.write(json.dumps({
                "dataset": f"science.{example['dataset']}",
                "id": f"science_{idx}",
                "messages": [
                    {"role": "user", "content": example["input"]},
                    {"role": "assistant", "content": example["output"]}
                ],
            }) + "\n")



def should_be_filtered(example):
    # we filter out conversations that contain some specific strings
    filter_strings = [
        "OpenAI",
        "Open AI",
        "ChatGPT",
        "Chat GPT",
        "GPT-3",
        "GPT3",
        "GPT 3",
        "GPT-4",
        "GPT4",
        "GPT 4",
        "GPT-3.5",
        "GPT3.5",
        "GPT 3.5",
        "BingChat",
        "Bing Chat",
        "BARD",
        "Palm",
        "Anthropic",
        "Claude",
        "LAION",
        "Open Assistant",
        "OpenAssistant", 
    ]
    for message in example["messages"]:
        if any([filter_string.lower() in message["content"].lower() for filter_string in filter_strings]):
            return True
    return False
        



def generate_50k_sft_datasets():
    """Generate 50k subsample of sft dataset
        ```
        from open_instruct.reformat_datasets import *
        from note_pruning_analysis import get_dataset
        import glob
        L = glob.glob('data/processed/*/*50k_*data.jsonl')
        L += [
            'data/processed/lima/lima_data.jsonl',
            'data/processed/dolly/dolly_data.jsonl',
            'data/processed/oasst/oasst2_data.jsonl',
        ]
        for p in sorted(L):
            ds = get_dataset(p)
            print(p, len(ds))
        ```
    """
    datasets = [
        'flan_v2',
        'gpt4_alpaca',
        'oasst2',
        'self_instruct',
        'sharegpt',
        'stanford_alpaca',
        'wizardlm',
        'ultrachat',
    ]

    for dataset in datasets:
        print(dataset)

        extra_kwargs = {}
        if dataset == 'sharegpt':
            extra_kwargs['data_file'] = "sharegpt_html_cleaned_and_split_2048_discardlongconv.json"
        if dataset.startswith('oasst2'):
            extra_kwargs['dataset_name'] = f'oasst2'
        else:
            extra_kwargs['dataset_name'] = f'{dataset}50k'

        globals()[f'convert_{dataset}_data'](
            data_dir=f'data/raw_train/{dataset}',
            output_dir=f'data/processed/{dataset}',
            num_examples=50_000,
            max_seq_length=2048,
            **extra_kwargs)



def generate_mixes(raw_data_dir, output_dir, mix_name, mix_proportions, dataset_size):
    """
        ```
        from open_instruct.reformat_datasets import generate_mixes
        raw_data_dir = 'data/raw_train'
        output_dir = 'data/processed'    
        dataset_size = 50_000
        mix_name = 'mix_redundant'
        mix_proportions = {
            'dolly': .25,
            'flan_v2': .25,
            'stanford_alpaca': .25,
            'oasst2': .25,
        }
        mix_name = 'mix_diverse'
        mix_proportions = {
            'wizardlm': 1/3,
            'sharegpt': 1/3,
            'ultrachat': 1/3,
        }
        generate_mixes(raw_data_dir, output_dir, mix_name, mix_proportions, dataset_size)
        ```
    """

    mix_sizes = {k: int(v*dataset_size) for k, v in mix_proportions.items()}
    last = list(mix_sizes.keys())[-1]
    mix_sizes[last] = dataset_size - sum([v for k, v in mix_sizes.items() if k!=last])

    if sum(mix_sizes.values()) != dataset_size:
        raise ValueError(f"mix sizes {mix_sizes} should sum up to {dataset_size}")
    
    for sub_dataset in mix_sizes.keys():
        convert_kwargs = {
            'data_dir': os.path.join(raw_data_dir, sub_dataset),
            'output_dir': os.path.join(raw_data_dir, 'mix', mix_name, sub_dataset),
            'num_examples': mix_sizes[sub_dataset],
            'max_seq_length': 2048,
        }
        if sub_dataset in ['oasst2', 'oasst1']:
            convert_fn = 'convert_oasst_data'
            convert_kwargs.update({
                'data_dir': os.path.join(raw_data_dir, 'oasst'),
                'source_filename': "2023-04-12_oasst_ready.trees.jsonl" if sub_dataset=='oasst1' else \
                    "2023-11-05_oasst2_ready.trees.jsonl",
                'dataset_name': sub_dataset,
            })
        elif sub_dataset == 'sharegpt':
            convert_fn = 'convert_sharegpt_data'
            convert_kwargs.update({
                'data_file': "sharegpt_html_cleaned_and_split_2048_discardlongconv.json",
            })
        elif sub_dataset == 'ultrachat':
            convert_fn = 'convert_ultrachat_data'
            convert_kwargs.update({
                'dataset_name': 'ultrachat200k',
            })
        else:
            convert_fn = f'convert_{sub_dataset}_data'
        globals()[convert_fn](**convert_kwargs)


    subset_files = glob.glob(os.path.join(raw_data_dir, 'mix', mix_name, '*/*'))
    subset_files = [x for x in subset_files if 'test' not in x]
    print(f'Concatenating files: {subset_files}')
    examples = []
    for subset_file in subset_files:
        with open(subset_file, "r") as fin:
            for line in fin:
                examples.append(json.loads(line))
    random.seed(0)
    random.shuffle(examples)

    os.makedirs(os.path.join(output_dir, 'mix'), exist_ok=True)
    with open(os.path.join(output_dir, 'mix', f'{mix_name}_data.jsonl'), 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

        


        

if __name__ == "__main__":
    # all supported datasets    
    supported_datasets = []
    all_funcs = [func_name for func_name in globals() if callable(globals()[func_name])]
    for func_name in all_funcs:
        if re.match(r"convert_.+_data", func_name):
            supported_datasets.append(func_name[8:-5])
            

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--raw_data_dir", 
        type=str, 
        default="data/downloads"
    )
    arg_parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/processed"
    )
    arg_parser.add_argument(
        "--dataset", 
        type=str, 
        nargs="+",
        choices=supported_datasets+["tulu_v1", "tulu_v2"],
        default=supported_datasets+["tulu_v1", "tulu_v2"]
    )
    arg_parser.add_argument(
        "--seed", 
        type=int, 
        default=42
    )
    args = arg_parser.parse_args()
    random.seed(args.seed)

    # get the subfolder names in raw_data_dir
    subfolders = [f for f in os.listdir(args.raw_data_dir) if os.path.isdir(os.path.join(args.raw_data_dir, f))]

    for dataset in args.dataset: 
        if dataset == "tulu_v1":
            print(f"Processing tulu_v1 subsets...")
            convert_flan_v2_data(
                data_dir=os.path.join(args.raw_data_dir, "flan_v2"), 
                output_dir=os.path.join(args.output_dir, "tulu_v1", "flan_v2_subset"),
                data_file="tulu_v1_resampled_flan_100k.jsonl",
            )
            convert_cot_data(
                data_dir=os.path.join(args.raw_data_dir, "cot"), 
                output_dir=os.path.join(args.output_dir, "tulu_v1", "cot_subset"),
                num_few_shot_examples=50000,
                num_zero_shot_examples=50000
            )
            convert_oasst_data(
                data_dir=os.path.join(args.raw_data_dir, "oasst"), 
                output_dir=os.path.join(args.output_dir, "tulu_v1", "oasst_subset"), 
                top_k_reply=None
            )
            convert_dolly_data(
                data_dir=os.path.join(args.raw_data_dir, "dolly"),
                output_dir=os.path.join(args.output_dir, "tulu_v1", "dolly_subset"),
            )
            convert_gpt4_alpaca_data(
                data_dir=os.path.join(args.raw_data_dir, "gpt4_alpaca"),
                output_dir=os.path.join(args.output_dir, "tulu_v1", "gpt4_alpaca_subset"),
                load_en=True,
                load_zh=False,
                num_examples=None
            )
            convert_code_alpaca_data(
                data_dir=os.path.join(args.raw_data_dir, "code_alpaca"),
                output_dir=os.path.join(args.output_dir, "tulu_v1", "code_alpaca_subset"),
                num_examples=None
            )
            convert_sharegpt_data(
                data_dir=os.path.join(args.raw_data_dir, "sharegpt"),
                output_dir=os.path.join(args.output_dir, "tulu_v1", "sharegpt_subset"),
                data_file="sharegpt_html_cleaned_and_split_2048.json",
                num_examples=None
            )
            # merge all the subsets
            print("Merging all the subsets to create tulu v1...")
            all_subsets = [f for f in os.listdir(os.path.join(args.output_dir, "tulu_v1")) if f.endswith("_subset")]
            with open(os.path.join(args.output_dir, "tulu_v1", "tulu_v1_data.jsonl"), "w") as fout:
                for subset in all_subsets:
                    dataset_name = subset[:-len("_subset")]
                    with open(os.path.join(args.output_dir, "tulu_v1", subset, f"{dataset_name}_data.jsonl"), "r") as fin:
                        for line in fin:
                            fout.write(line)
        elif dataset == "tulu_v2":
            print(f"Processing tulu_v2 subsets...")
            convert_flan_v2_data(
                data_dir=os.path.join(args.raw_data_dir, "flan_v2"), 
                output_dir=os.path.join(args.output_dir, "tulu_v2", "flan_v2_subset"),
                data_file="tulu_v2_resampled_flan_50k.jsonl",
            )
            convert_cot_data(
                data_dir=os.path.join(args.raw_data_dir, "cot"), 
                output_dir=os.path.join(args.output_dir, "tulu_v2", "cot_subset"),
                num_few_shot_examples=25000,
                num_zero_shot_examples=25000,
            )
            convert_oasst_data(
                data_dir=os.path.join(args.raw_data_dir, "oasst"), 
                output_dir=os.path.join(args.output_dir, "tulu_v2", "oasst_subset"), 
                top_k_reply=1
            )
            convert_lima_data(
                data_dir=os.path.join(args.raw_data_dir, "lima"), 
                output_dir=os.path.join(args.output_dir, "tulu_v2", "lima_subset"), 
                num_examples=None
            )
            convert_gpt4_alpaca_data(
                data_dir=os.path.join(args.raw_data_dir, "gpt4_alpaca"), 
                output_dir=os.path.join(args.output_dir, "tulu_v2", "gpt4_alpaca_subset"), 
                load_en=True, 
                load_zh=False, 
                num_examples=20000
            )
            convert_code_alpaca_data(
                data_dir=os.path.join(args.raw_data_dir, "code_alpaca"), 
                output_dir=os.path.join(args.output_dir, "tulu_v2", "code_alpaca_subset"), 
                num_examples=None
            )
            convert_sharegpt_data(
                data_dir=os.path.join(args.raw_data_dir, "sharegpt"), 
                output_dir=os.path.join(args.output_dir, "tulu_v2", "sharegpt_subset"),
                ## wpq: use 2048 version and strictly enforce this.
                # data_file="sharegpt_html_cleaned_and_split_4096.json",
                data_file="sharegpt_html_cleaned_and_split_2048_discardlongconv.json",
                num_examples=None
            )
            convert_wizardlm_data(
                data_dir=os.path.join(args.raw_data_dir, "wizardlm"), 
                output_dir=os.path.join(args.output_dir, "tulu_v2", "wizardlm_subset"), 
                num_examples=30000
            )
            convert_open_orca_data(
                data_dir=os.path.join(args.raw_data_dir, "open_orca"), 
                output_dir=os.path.join(args.output_dir, "tulu_v2", "open_orca_subset"), 
                num_gpt4_examples=30000, 
                num_gpt35_examples=0
            )
            convert_science_data(
                data_dir=os.path.join(args.raw_data_dir, "science"), 
                output_dir=os.path.join(args.output_dir, "tulu_v2", "science_subset"),
                num_examples=None
            )
            convert_hard_coded_data(
                data_dir=os.path.join(args.raw_data_dir, "hard_coded"), 
                output_dir=os.path.join(args.output_dir, "tulu_v2", "hard_coded_subset"),
                repeat=10,
            )
            # merge all the subsets
            print("Merging all the subsets to create tulu v2...")
            all_subsets = [f for f in os.listdir(os.path.join(args.output_dir, "tulu_v2")) if f.endswith("_subset")]
            with open(os.path.join(args.output_dir, "tulu_v2", "tulu_v2_data.jsonl"), "w") as fout, \
                open(os.path.join(args.output_dir, "tulu_v2", "tulu_v2_filtered_data.jsonl"), "w") as fout_filtered:
                for subset in all_subsets:
                    dataset_name = subset[:-len("_subset")]
                    with open(os.path.join(args.output_dir, "tulu_v2", subset, f"{dataset_name}_data.jsonl"), "r") as fin:
                        for line in fin:
                            example = json.loads(line)
                            if subset not in ["hard_coded_subset"] and should_be_filtered(example):
                                fout_filtered.write(line)
                            else:
                                fout.write(line)
            """
            python open_instruct/reformat_datasets.py --raw_data_dir data/raw_train/ --output_dir data/processed/ --dataset tulu_v2
            """
            ## wpq: to avoid truncate examples during training.
            print("Filtering tulu_v2 to max_seq_length=2048...")
            filepath = os.path.join(args.output_dir, "tulu_v2", "tulu_v2_data.jsonl")
            filter_json_by_numtoks(filepath, max_seq_length=2048)
        else: 
            print(f"Processing {dataset} data with default configurations...")
            globals()[f"convert_{dataset}_data"](os.path.join(args.raw_data_dir, dataset), os.path.join(args.output_dir, dataset))

            ## wpq: to avoid truncate examples during training.
            print(f"Filtering {dataset} to max_seq_length=2048...")
            filepath = os.path.join(args.output_dir, dataset, f"{dataset}_data.jsonl")
            if os.path.exists(filepath):
                filter_json_by_numtoks(filepath, max_seq_length=2048)
            else:
                print(f"Warning: {filepath} does not exist. Skipping...")
            