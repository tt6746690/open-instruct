# import os
# from note_pruning_analysis import filter_json_by_numtoks
# from open_instruct.reformat_datasets import convert_alpagasus_data
# from open_instruct.instruction_encode_templates import encode_instruction_example

# data_dir = 'data/raw_train/alpagasus'
# output_dir = 'data/processed/alpagasus'
# convert_alpagasus_data(data_dir, output_dir, num_examples=None)
# filepath = os.path.join(output_dir, "chatgpt_9k.jsonl")
# print(f"Filtering alpagasus to max_seq_length=2048...")
# filter_json_by_numtoks(filepath, max_seq_length=2048)

import json
file_path = 'data/raw_train/DEITA10k/deita_10k.json'
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