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

# import json
# file_path = 'data/raw_train/DEITA10k/deita_10k.json'
# parsed_json_list = []

# # Reading and parsing the file
# with open(file_path, 'r') as file:
#     for line in file:
#         # Each line is a separate JSON object
#         json_object = json.loads(line)
#         parsed_json_list.append(json_object)

# # Writing the list back to the file as JSON
# with open(file_path, 'w') as file:
#     json.dump(parsed_json_list, file, indent=4)

# from open_instruct.reformat_datasets import convert_deita_data

# data_dir = 'data/raw_train/DEITA6k'
# output_dir = 'data/processed/DEITA6k'
# convert_deita_data(data_dir, output_dir, data_file="deita_6k_cleaned_and_split_2048_discardlongconv.json", dataset_name="deita_sharegpt")

# data_dir = 'data/raw_train/DEITA10k'
# output_dir = 'data/processed/DEITA10k'
# convert_deita_data(data_dir, output_dir, data_file="deita_10k_cleaned_and_split_2048_discardlongconv.json", dataset_name="deita_sharegpt")

# import os
# #from note_pruning_analysis import filter_json_by_numtoks
# from open_instruct.reformat_datasets import convert_HelpSteer_data

# data_dir = 'data/raw_train/HelpSteer'
# output_dir = 'data/processed/HelpSteer'
# convert_HelpSteer_data(data_dir, output_dir, num_examples=None)
# filepath = os.path.join(output_dir, "HelpSteer_train.jsonl")
# print(f"Filtering HelpSteer to max_seq_length=2048...")
# #filter_json_by_numtoks(filepath, max_seq_length=2048)

from open_instruct.reformat_datasets import convert_orca_dpo_pairs_data
data_dir = 'data/raw_train/orca_dpo_pairs'
output_dir = 'data/processed/orca_dpo_pairs'
convert_orca_dpo_pairs_data(data_dir, output_dir, num_examples=None)