"""
Modified based on https://github.com/lm-sys/FastChat/blob/main/fastchat/data/split_long_conversation.py
Split long conversations based on certain max length.

Usage:
1. download json data files to `./raw`
2. run command below for each file
python -u split_long.py --in-file ./raw/input.json --out-file ./processed/output.json --begin 0 --model-name-or-path /path/to/huggingface/llama --max-length 2048


python scripts/split_ultrachat_conversations.py --in-file data/raw_train/ultrachat/full/train_0.jsonl --out-file data/raw_train/ultrachat/full/train_0_splitlongconv.jsonl --model-name-or-path results/baselines/huggyllama/llama-7b --max-length 2048
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from typing import Dict, Sequence, Optional

import transformers
from tqdm import tqdm


def make_sample(sample, start_idx, end_idx):
    assert (end_idx - start_idx) % 2 == 0
    return {
        "id": sample["id"] + "_" + str(start_idx),
        "data": sample["data"][start_idx:end_idx],
    }


tokenizer = max_length = None


def split_one_sample(sample):
    tokenized_lens = []
    conversations = sample["data"]
    assert len(conversations) %2 == 0, print(conversations)
    # conversations = conversations[: len(conversations) // 2 * 2]
    for c in conversations:
        length = len(tokenizer(c).input_ids) + 6
        tokenized_lens.append(length)

    start_idx = 0
    cur_len = 0

    # if len(conversations) % 2 != 0 or len(conversations) < 2:
    #     return []

    new_samples = []
    for i in range(0, len(conversations), 2):
        tmp_len = tokenized_lens[i] + tokenized_lens[i + 1]
        if cur_len + tmp_len > max_length:
            new_samples.append(make_sample(sample, start_idx, i))
            start_idx = i
            cur_len = 0
            ## wpq: discard the rest of the conversation since we have lots of data to work from.
            break
        elif i == len(conversations) - 2:
            new_samples.append(make_sample(sample, start_idx, i + 2))

        cur_len += tmp_len

    return new_samples


def split_all(content, begin, end, tokenizer_, max_length_):
    """
    Keep the maximum round of conversations within the max token length constraint
    """
    global tokenizer, max_length
    tokenizer = tokenizer_
    max_length = max_length_

    content = content[begin:end]
    new_content = []

    with ProcessPoolExecutor() as executor:
        for result in tqdm(executor.map(split_one_sample, content), total=len(content)):
            new_content.extend(result)

    return new_content

def check_content(content):
    new_content = []
    for c in content:
        if len(c["data"]) > 0 and len(c["data"]) % 2 == 0:
            new_content.append(c)
    return new_content


def main(args):
    # content = [json.loads(l) for l in open(args.in_file, "r")]
    content = []
    with open(args.in_file, 'r') as f:
        for i, l in enumerate(f):
            try:
                x = json.loads(l)
                content.append(x)
            except json.JSONDecodeError as e:
                ## wpq: handle cases where two json in one line, e..g, "{'a': 1}{'a': 2}"
                # probably happens when cat jsonl files without `\n` in the end of each file.
                assert(len(l.split('}{'))==2)
                l = l.split('}{')
                l[0] = l[0] + '}'
                l[1] = '{' + l[1]
                content += [json.loads(x) for x in l]


    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.max_length,
        padding_side="right",
        use_fast=False,
    )
    new_content = split_all(content, args.begin, args.end, tokenizer, args.max_length)
    new_content = check_content(new_content)


    ## wpq: some additional cleaning, and conveft to dialogue format
    from datasets import Dataset

    ds = Dataset.from_list(new_content)

    phrases = ["I do not have emotions", 
               "I don't have opinions"]
    def filter_non_refusal_fn(example):
        L = []
        for x in example['data']:
            l = [p in x for p in phrases]
            L.append(any(l))
        return not any(L)
    ds = ds.filter(filter_non_refusal_fn, num_proc=32, keep_in_memory=True)

    def convert_to_dialogue_format(example):
        example['messages'] = [
            {'role': 'user' if i%2==0 else 'assistant', 'content': v}
            for i, v in enumerate(example['data'])
        ]
        # remove the splitlongconv trailing id
        example['id'] = example['id'].split('_')[0]
        return example
    ds = ds.map(convert_to_dialogue_format, num_proc=32, keep_in_memory=True)
    ds = ds.remove_columns(['data'])

    print(f"total: {len(content)}, after splitlongconv: {len(new_content)}, cleaned: {len(ds)}")
    # with open(args.out_file, "w")as f:
    #     f.writelines("\n".join([json.dumps(l) for l in new_content]))
    ds.to_json(args.out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--out-file", type=str, default="sharegpt_split.json")
    parser.add_argument("--begin", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()
    main(args)