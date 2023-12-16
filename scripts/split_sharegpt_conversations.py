"""
This script is largely copied from the Vicuna repo: https://github.com/lm-sys/FastChat/blob/main/fastchat/data/split_long_conversation.py
We fixed a bug in `split_one_sample`, which previously includes long conversations in the processed data. Now we skip these long conversations.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import transformers
from tqdm import tqdm


def make_sample(sample, start_idx, end_idx):
    assert (end_idx - start_idx) % 2 == 0
    return {
        "id": sample["id"] + "_" + str(start_idx),
        "conversations": sample["conversations"][start_idx:end_idx],
    }


tokenizer = max_length = special_tok_len = None


def split_one_sample(sample):
    tokenized_lens = []
    conversations = sample["conversations"]
    conversations = conversations[: len(conversations) // 2 * 2]
    for c in conversations:
        length = len(tokenizer(c["value"]).input_ids) + special_tok_len
        if special_tok_len != 6:
            if c['from'] == 'gpt':
                length += 1 ## wpq: take into account </s> token.
        tokenized_lens.append(length)

    start_idx = 0
    cur_len = 0

    if len(conversations) % 2 != 0 or len(conversations) < 2:
        return []

    new_samples = []
    for i in range(0, len(conversations), 2):
        tmp_len = tokenized_lens[i] + tokenized_lens[i + 1]
        if cur_len + tmp_len > max_length:
            if cur_len <= max_length: ## wpq: ensure added example < max_length
                new_samples.append(make_sample(sample, start_idx, i))
            if tmp_len > max_length:  # if the current conversation is too long, we should skip it
                start_idx = i + 2
            else:
                start_idx = i
            cur_len = 0
            ## wpq: just discard conversation starting from the middle, since there is no context.
            break ## for now comment out, since the data in use are created without this break
        elif i == len(conversations) - 2:
            if cur_len + tmp_len <= max_length: ## wpq: ensure added example < max_length
                new_samples.append(make_sample(sample, start_idx, i + 2))

        cur_len += tmp_len

    return new_samples


def split_all(content, begin, end, tokenizer_, max_length_, special_tok_len_):
    """
    Keep the maximum round of conversations within the max token length constraint
    """
    global tokenizer, max_length, special_tok_len
    tokenizer = tokenizer_
    max_length = max_length_
    special_tok_len = special_tok_len_

    content = content[begin:end]
    new_content = []

    with ProcessPoolExecutor(max_workers=128) as executor:
        for result in tqdm(executor.map(split_one_sample, content), total=len(content)):
            new_content.extend(result)

    return new_content


def filter_invalid_roles(content):
    new_content = []
    for i, c in enumerate(content):
        roles = ["human", "gpt"]
        if len(c["conversations"]) <= 0:
            continue

        valid = True
        for j, s in enumerate(c["conversations"]):
            if s["from"] != roles[j % 2]:
                valid = False
                break

        if valid:
            new_content.append(c)

    return new_content


def main(args):
    if args.in_files[0].startswith('HuggingFaceH4/ultrachat_200k'):
        if 'train' in args.in_files[0]:
            split = 'train_sft'
        elif 'test' in args.in_files[0]:
            split = 'test_sft'
        else:
            raise ValueError('unknown split')

        data_dir = '/gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data/raw_train/ultrachat'
        from datasets import load_dataset
        ds = load_dataset('HuggingFaceH4/ultrachat_200k', cache_dir=data_dir, split=split)
        ## wpq: convert the formatting to use the split conversation code.
        def convert_message_to_sharegpt_chat_format(m):
            return {'from': 'gpt' if m['role']=='assistant' else 'human',
                    'value': m['content']}
        def convert_conversations_to_sharegpt_chat_format(conv):
            return [convert_message_to_sharegpt_chat_format(m) for m in conv]
        prompt_id = ds['prompt_id']
        conversations = ds['messages']
        conversations = [convert_conversations_to_sharegpt_chat_format(x)
                        for x in conversations]
        assert(len(conversations)==len(prompt_id))
        content = [{'id': pid, 'conversations': conv} for pid, conv in zip(prompt_id, conversations)]
    else:
        content = []
        for file in args.in_files:
            if file.endswith('json'):
                with open(file, 'rb') as f:
                    content.extend(json.load(f))
            elif file.endswith('jsonl'):
                with open(file, 'rb') as f:
                    for line in f:
                        content.append(json.loads(line))

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
    )
    new_content = split_all(content, args.begin, args.end, tokenizer, args.max_length, args.special_tok_len)
    print(f"after split:  {len(new_content)}")
    new_content = filter_invalid_roles(new_content)
    print(f"after filter: {len(new_content)}")

    if args.in_files[0].startswith('HuggingFaceH4/ultrachat_200k'):
        ## wpq: convert back to ultrachat format
        def convert_messages_to_ultrachat_format(m):
            return {'content': m['value'], 
                    'role': 'user' if m['from']=='human' else 'assistant'}
        def convert_conversations_to_ultrachat_format(conv):
            return [convert_messages_to_ultrachat_format(m) for m in conv] 
        new_content = [{'prompt_id': x['id'], 
                        'messages': convert_conversations_to_ultrachat_format(x['conversations'])} 
            for x in new_content]

    print(f"total: {len(content)}, new: {len(new_content)}")
    json.dump(new_content, open(args.out_file, "w"), indent=2)


    ## wpq: more logging.
    from collections import Counter
    print('conversation lengths:')
    nconv = [len(x['conversations']) for x in content]
    print('before split: ', dict(Counter(nconv)))
    nconv = [len(x['conversations']) if 'conversations' in x else x['messages'] for x in new_content]
    print('after split: ', dict(Counter(nconv)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-files", nargs="+", type=str)
    parser.add_argument("--out-file", type=str, default="sharegpt_split.json")
    parser.add_argument("--begin", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--special_tok_len", type=int, default=6, help="the length of special tokens, e.g., <s>, \\n, <|assistant|>, <|user|>, etc.")
    args = parser.parse_args()
    main(args)