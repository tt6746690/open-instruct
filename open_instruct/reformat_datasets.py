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
import pandas as pd
import argparse
from open_instruct.instruction_encode_templates import encode_instruction_example, encode_few_shot_example


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
            

def convert_flan_v2_data(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    with open(os.path.join(data_dir, "flan_v2_resampled_100k.jsonl"), "r") as fin:
        for line in fin:
            examples.append(json.loads(line))
    output_path = os.path.join(output_dir, "flan_v2_data.jsonl")
    with open(output_path, "w") as fout:
        for idx, example in enumerate(examples):
            prompt = example["inputs"]
            if not prompt.endswith("\n") and not prompt.rstrip().endswith(":"):
                prompt += "\n"
            completion = example["targets"]
            fout.write(json.dumps({
                "dataset": "flan_v2",
                "id": f"flan_v2_{idx}",
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ]
            }) + "\n")


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


def convert_dolly_data(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    with open(os.path.join(data_dir, "databricks-dolly-15k.jsonl"), "r") as fin:
        for line in fin:
            examples.append(json.loads(line))
    output_path = os.path.join(output_dir, "dolly_data.jsonl")
    with open(output_path, "w") as fout:
        for idx, example in enumerate(examples):
            encoded_example = encode_instruction_example(
                instruction=example["instruction"], 
                input=example["context"], 
                output=example["response"],
                random_template=True,
                eos_token=None
            )
            fout.write(json.dumps({
                "dataset": "dolly",
                "id": f"dolly_{idx}",
                "messages": [
                    {"role": "user", "content": encoded_example["prompt"]},
                    {"role": "assistant", "content": encoded_example["completion"]},
                ]
            }) + "\n")


def convert_self_instruct_data(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    with open(os.path.join(data_dir, "all_instances_82K.jsonl"), "r") as fin:
        for line in fin:
            examples.append(json.loads(line))
    output_path = os.path.join(output_dir, "self_instruct_data.jsonl")
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
                "dataset": "self_instruct",
                "id": f"self_instruct_{idx}",
                "messages": [
                    {"role": "user", "content": encoded_example["prompt"]},
                    {"role": "assistant", "content": encoded_example["completion"]},
                ]
            }) + "\n")


def convert_unnatural_instructions_data(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    instance_cnt = 0
    with open(os.path.join(data_dir, "core_data.jsonl"), "r") as fin, open((os.path.join(output_dir, "unnatural_instructions_data.jsonl")), "w") as fout:
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
                    eos_token=None
                )
                fout.write(json.dumps({
                    "dataset": "unnatural_instructions",
                    "id": f"unnatural_instructions_{instance_cnt}",
                    "messages": [
                        {"role": "user", "content": encoded_example["prompt"]},
                        {"role": "assistant", "content": encoded_example["completion"]},
                    ]
                }) + "\n")
                instance_cnt += 1


def convert_stanford_alpaca_data(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    with open(os.path.join(data_dir, "alpaca_data.json"), "r") as fin:
        examples.extend(json.load(fin))
    output_path = os.path.join(output_dir, "stanford_alpaca_data.jsonl")
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
                "dataset": "stanford_alpaca",
                "id": f"stanford_alpaca_{idx}",
                "messages": [
                    {"role": "user", "content": encoded_example["prompt"]},
                    {"role": "assistant", "content": encoded_example["completion"]},
                ]
            }) + "\n")


def convert_code_alpaca_data(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    with open(os.path.join(data_dir, "code_alpaca_20k.json"), "r") as fin:
        examples.extend(json.load(fin))
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


def convert_gpt4_alpaca_data(data_dir, output_dir, load_en=True, load_zh=False):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    if load_en:
        with open(os.path.join(data_dir, "alpaca_gpt4_data.json"), "r") as fin:
            examples.extend(json.load(fin))
    if load_zh:
        with open(os.path.join(data_dir, "alpaca_gpt4_data_zh.json"), "r") as fin:
            examples.extend(json.load(fin))
    output_path = os.path.join(output_dir, "gpt4_alpaca_data.jsonl")
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
                "dataset": "gpt4_alpaca",
                "id": f"gpt4_alpaca_{idx}",
                "messages": [
                    {"role": "user", "content": encoded_example["prompt"]},
                    {"role": "assistant", "content": encoded_example["completion"]},
                ]
            }) + "\n")


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
    if 'generated_output' in ds.column_names:
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
        'commentinstrv4.json',
        'commentinstrv4_rge5.json',
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


def convert_sharegpt_data(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    with open(os.path.join(data_dir, "sharegpt_html_cleaned_and_split.json"), "r") as fin:
        examples.extend(json.load(fin))

    output_path = os.path.join(output_dir, "sharegpt_data.jsonl")
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
                    "dataset": "sharegpt",
                    "id": f"sharegpt_{example['id']}",
                    "messages": messages
                }) + "\n")
        print(f"# of invalid examples in sharegpt data: {invalid_cnt}")


def convert_baize_data(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    for source in ["alpaca", "medical", "quora", "stackoverflow"]:
        with open(os.path.join(data_dir, f"{source}_chat_data.json"), "r") as fin:
            examples.extend(json.load(fin))

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


def convert_oasst1_data(data_dir, output_dir):
    '''
    For OASST1, because it's in a tree structure, where every user input might get multiple replies, 
    we have to save every path from the root node to the assistant reply (including both leaf node and intemediate node).
    This results in some of the messages being duplicated among different paths (instances).
    Be careful when using this dataset for training. Ideally, you should only minimize the loss of the last message in each path.
    '''
    os.makedirs(output_dir, exist_ok=True)
    conversations = []
    with open(os.path.join(data_dir, "2023-04-12_oasst_ready.trees.jsonl"), "r") as fin:
        for line in fin:
            conversations.append(json.loads(line))

    output_path = os.path.join(output_dir, "oasst1_data.jsonl")

    # we filter out the sequences that mention the creator information
    filter_strings = [
        "LAION",
        "Open Asssistant",
        "OpenAssistant",               
    ]

    # tranvers the conversation tree, and collect all valid sequences
    def dfs(reply, messages, valid_sequences):
        if any([filter_string in reply["text"] for filter_string in filter_strings]):
            return
        if reply["role"] == "assistant":
            messages.append(
                {"role": "assistant", "content": reply["text"]}
            )
            if not reply["replies"]:  # leaf node
                valid_sequences.append(messages[:])
            else:
                for child in reply["replies"]:
                    dfs(child, messages, valid_sequences)
            messages.pop()
        elif reply["role"] == "prompter":
            messages.append(
                {"role": "user", "content": reply["text"]}
            )
            for child in reply["replies"]:
                dfs(child, messages, valid_sequences)
            messages.pop()
        else:
            raise ValueError(f"Unknown role: {reply['role']}")
        
    with open(output_path, "w") as fout:
        example_cnt = 0
        for _, conversation in enumerate(conversations):
            valid_sequences = []
            dfs(conversation["prompt"], [], valid_sequences)
            for sequence in valid_sequences:
                fout.write(json.dumps({
                    "dataset": "oasst1",
                    "id": f"oasst1_{example_cnt}",
                    "messages": sequence
                }) + "\n")
                example_cnt += 1


def convert_lima_data(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    with open(os.path.join(data_dir, "train.jsonl"), "r") as fin:
        for line in fin:
            examples.append(json.loads(line))
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


def convert_wizardlm_data(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    with open(os.path.join(data_dir, "WizardLM_evol_instruct_V2_143k.json"), "r") as fin:
        examples = json.load(fin)

    output_path = os.path.join(output_dir, "wizardlm_data.jsonl")
    with open(output_path, "w") as fout:
        for idx, example in enumerate(examples):
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
            fout.write(json.dumps({
                "dataset": "wizardlm",
                "id": f"wizardlm_{example['idx']}",
                "messages": messages,
            }) + "\n")


def convert_open_orca_data(data_dir, output_dir, num_gpt4_examples=100000, num_gpt35_examples=0):
    os.makedirs(output_dir, exist_ok=True)
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
                {"role": "system", "content": example["system_prompt"]},
                {"role": "user", "content": example["question"]},
                {"role": "assistant", "content": example["response"]}
            ]
            fout.write(json.dumps({
                "dataset": "open_orca",
                "id": f"open_orca_{example['id']}",
                "messages": messages,
            }) + "\n")


def convert_ultrachat_data(data_dir, output_dir):
    from datasets import load_dataset, concatenate_datasets

    ## wpq: ultrachat200k
    # ds = load_dataset('HuggingFaceH4/ultrachat_200k', cache_dir=data_dir, split='train_sft')
    data_files = {'train': os.path.join(data_dir, 'ultrachat_200k_train_splitlongconv.json'),
                  'test': os.path.join(data_dir, 'ultrachat_200k_test_splitlongconv.json')}
    
    for split in ['train', 'test']:
        ds = load_dataset('json', data_files=data_files, split=split, cache_dir=data_dir)
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
        
        output_path = os.path.join(output_dir, f'ultrachat200k_{split}_data.jsonl')
        ds.to_json(output_path)

    # ## wpq: ultrachat1.5m
    # data_files = {f'train_{i}': os.path.join(data_dir, 'full_splitlongconv_2048', f'train_{i}.jsonl') 
    #               for i in range(10)}
    # ds = load_dataset('json', data_files=data_files, cache_dir=os.path.join(data_dir, 'full_splitlongconv_2048'))
    # ds = concatenate_datasets([ds[f'train_{i}'] for i in range(10)])
    # def add_metadata_fn(example, idx):
    #     example.update({'dataset': 'ultrachat', 'id': f'ultrachat_{idx}'})
    #     return example
    # ds = ds.map(add_metadata_fn, with_indices=True, num_proc=10)
    # # too memory/compute intense.
    # # ds = ds.select_columns(['dataset', 'id']).add_column('messages', ds['messages'])

    # save_path = os.path.join(output_dir, f'ultrachat15_data.jsonl')
    # ds.to_json(save_path)

    # num_shards = 10
    # for i in range(num_shards):
    #     ds_shard = ds.shard(num_shards, i, contiguous=True)
    #     save_path = os.path.join(output_dir, f'ultrachat15_{i}_data.jsonl')
    #     ds_shard.to_json(save_path)



def get_all_supported_datasets():   
    supported_datasets = []
    all_funcs = [func_name for func_name in globals() if callable(globals()[func_name])]
    for func_name in all_funcs:
        if re.match(r"convert_.+_data", func_name):
            supported_datasets.append(func_name[8:-5])
    return supported_datasets



if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--raw_data_dir", type=str, default="data/downloads")
    arg_parser.add_argument("--output_dir", type=str, default="data/processed")
    arg_parser.add_argument("--seed", type=int, default=42)
    arg_parser.add_argument("--dataset", type=str, default="all", help="all or a specific dataset")
    args = arg_parser.parse_args()
    random.seed(args.seed)

    # get the subfolder names in raw_data_dir
    subfolders = [f for f in os.listdir(args.raw_data_dir) if os.path.isdir(os.path.join(args.raw_data_dir, f))]

    # all supported datasets
    supported_datasets = []
    all_funcs = [func_name for func_name in globals() if callable(globals()[func_name])]
    for func_name in all_funcs:
        if re.match(r"convert_.+_data", func_name):
            supported_datasets.append(func_name[8:-5])

    # check if the subfolder names are supported datasets
    valid_subfolders = []
    for subfolder in subfolders:
        if subfolder not in supported_datasets:
            print(f"Warning: {subfolder} in the raw data folder is not a supported dataset. We will skip it.")
        else:
            valid_subfolders.append(subfolder)

    if args.dataset is not None and args.dataset != "all" and args.dataset in valid_subfolders:
        valid_subfolders = [args.dataset]
    
    # prepare data for each dataset
    statistics = {}
    for subfolder in valid_subfolders:
        print(f"Processing {subfolder} data...")
        globals()[f"convert_{subfolder}_data"](os.path.join(args.raw_data_dir, subfolder), os.path.join(args.output_dir, subfolder))