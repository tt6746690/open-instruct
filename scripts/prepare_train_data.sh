# check if there is $HF_TOKEN in the environment variables
if [ -z "$HF_TOKEN" ]
then
    echo "Warning: HuggingFace dataset LIMA requires permissive access."
    echo "Warning: Please request the access at https://huggingface.co/datasets/GAIR/lima and set the HF_TOKEN environment variable before running this script."
    exit 1
fi

echo "Downloading Super-NaturalInstructions dataset..."
wget -P data/raw_train/super_ni/ https://github.com/allenai/natural-instructions/archive/refs/heads/master.zip
unzip data/raw_train/super_ni/master.zip -d data/raw_train/super_ni/ && rm data/raw_train/super_ni/master.zip
mv data/raw_train/super_ni/natural-instructions-master/* data/raw_train/super_ni/ && rm -r data/raw_train/super_ni/natural-instructions-master


echo "Downloading the flan_v2 chain-of-thought submix..."
wget -P data/raw_train/cot/ https://beaker.org/api/v3/datasets/01GXZ52K2Q932H6KZY499A7FE8/files/cot_zsopt.jsonl
wget -P data/raw_train/cot/ https://beaker.org/api/v3/datasets/01GXZ51ZV283RAZW7J3ECM4S58/files/cot_fsopt.jsonl


echo "Downloading the flan_v2 collection, here we subsampled only 100K instances..."
wget -P data/raw_train/flan_v2/ https://beaker.org/api/v3/datasets/01GZTTS2EJFPA83PXS4FQCS1SA/files/flan_v2_resampled_100k.jsonl


echo "Downloading self-instruct data..."
wget -P data/raw_train/self_instruct/ https://raw.githubusercontent.com/yizhongw/self-instruct/main/data/gpt3_generations/batch_221203/all_instances_82K.jsonl


echo "Downloading unnatural-instructions data..."
wget -P data/raw_train/unnatural_instructions/ https://github.com/orhonovich/unnatural-instructions/raw/main/data/core_data.zip
unzip data/raw_train/unnatural_instructions/core_data.zip -d data/raw_train/unnatural_instructions/


echo "Downloading Stanford alpaca data..."
wget -P data/raw_train/stanford_alpaca/ https://github.com/tatsu-lab/stanford_alpaca/raw/main/alpaca_data.json


echo "Downloading the dolly dataset..."
wget -P data/raw_train/dolly/ https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl


echo "Downloading the OpenAssistant data (oasst1)..."
wget -P data/raw_train/oasst1/ https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/2023-04-12_oasst_ready.trees.jsonl.gz
gzip -d data/raw_train/oasst1/2023-04-12_oasst_ready.trees.jsonl.gz


echo "Downloading the code alpaca dataset..."
wget -P data/raw_train/code_alpaca/ https://github.com/sahil280114/codealpaca/raw/master/data/code_alpaca_20k.json


echo "Downloading the gpt4-llm dataset..."
wget -P data/raw_train/gpt4_alpaca/ https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/raw/main/data/alpaca_gpt4_data.json
wget -P data/raw_train/gpt4_alpaca/ https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/raw/main/data/alpaca_gpt4_data_zh.json


echo "Downloading the baize dataset..."
wget -P data/raw_train/baize/ https://github.com/project-baize/baize-chatbot/raw/main/data/alpaca_chat_data.json
wget -P data/raw_train/baize/ https://github.com/project-baize/baize-chatbot/raw/main/data/medical_chat_data.json
wget -P data/raw_train/baize/ https://github.com/project-baize/baize-chatbot/raw/main/data/quora_chat_data.json
wget -P data/raw_train/baize/ https://github.com/project-baize/baize-chatbot/raw/main/data/stackoverflow_chat_data.json


echo "Downloading ShareGPT dataset..."
# 76920 conversations in total.
wget -P data/raw_train/sharegpt/ https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/HTML_cleaned_raw_dataset/sg_90k_part1_html_cleaned.json
wget -P data/raw_train/sharegpt/ https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/HTML_cleaned_raw_dataset/sg_90k_part2_html_cleaned.json
echo "Splitting the ShareGPT dataset..."
python scripts/split_sharegpt_conversations.py \
    --in-files data/raw_train/sharegpt/sg_90k_part1_html_cleaned.json data/raw_train/sharegpt/sg_90k_part2_html_cleaned.json \
    --out-file data/raw_train/sharegpt/sharegpt_html_cleaned_and_split.json \
    --model-name-or-path results/baselines/huggyllama/llama-7b



echo "Downloading LIMA dataset..."
wget --header="Authorization: Bearer $HF_TOKEN" -P data/raw_train/lima/ https://huggingface.co/datasets/GAIR/lima/raw/main/train.jsonl


echo "Downloading WizardLM dataset..."
wget -P data/raw_train/wizardlm/ https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_V2_196k/resolve/main/WizardLM_evol_instruct_V2_143k.json


echo "Downloading the OpenOrca dataset..."
wget -P data/raw_train/open_orca/ https://huggingface.co/datasets/Open-Orca/OpenOrca/resolve/main/1M-GPT4-Augmented.parquet
wget -P data/raw_train/open_orca/ https://huggingface.co/datasets/Open-Orca/OpenOrca/resolve/main/3_5M-GPT3_5-Augmented.parquet


# ## flan v2 full dataset processing
# 
# # download full flan_v2 dataset with snap_download
# python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='SirNeural/flan_v2', repo_type='dataset', local_dir='data/raw_train/flan2022', local_dir_use_symlinks=False)"
#
# # resample 1m/2m subset of flan_v2
# python scripts/resample_flan_v2.py --flan_v2_data_dir data/raw_train/flan2022 --total_num_samples 1000000 --output_path data/raw_train/flan2022/flan2022_1m.jsonl
#
# # reformat flan2022 mixture data resampled 1m
# python open_instruct/reformat_datasets.py --raw_data_dir data/raw_train --output_dir data/processed --dataset flan2022
#
## Use pre-downloaded huggingface's dataset and split long ultrachat conversations
python scripts/split_sharegpt_conversations.py \
    --in-files HuggingFaceH4/ultrachat_200k \
    --out-file data/raw_train/ultrachat/ultrachat_200k_splitlongconv.json \
    --model-name-or-path results/baselines/huggyllama/llama-7b \
    --max-length 2048

# # reformat ultrachat 200k
# python open_instruct/reformat_datasets.py --raw_data_dir data/raw_train --output_dir data/processed --dataset ultrachat




echo "Reformatting the datasets..."
python open_instruct/reformat_datasets.py --raw_data_dir data/raw_train/ --output_dir data/processed/


echo "Creating Tulu data mixtures..."
mkdir -p data/processed/tulu/
cat data/processed/flan_v2/flan_v2_data.jsonl \
    data/processed/cot/cot_data.jsonl \
    data/processed/dolly/dolly_data.jsonl \
    data/processed/oasst1/oasst1_data.jsonl \
    data/processed/gpt4_alpaca/gpt4_alpaca_data.jsonl \
    data/processed/code_alpaca/code_alpaca_data.jsonl \
    data/processed/sharegpt/sharegpt_data.jsonl \
    > data/processed/tulu/tulu_v1_mix.jsonl


cat data/processed/flan_v2/flan_v2_data.jsonl \
    data/processed/cot/cot_data.jsonl \
    data/processed/dolly/dolly_data.jsonl \
    data/processed/oasst1/oasst1_data.jsonl \
    > data/processed/tulu/tulu_v1_human_mix.jsonl

cat data/processed/flan_v2/flan_v2_data.jsonl \
    data/processed/cot/cot_data.jsonl \
    data/processed/oasst1/oasst1_data.jsonl \
    data/processed/lima/lima_data.jsonl \
    data/processed/code_alpaca/code_alpaca_data.jsonl \
    data/processed/sharegpt/sharegpt_data.jsonl \
    data/processed/wizardlm/wizardlm_data.jsonl \
    data/processed/open_orca/open_orca_data.jsonl \
    > data/processed/tulu/tulu_v2_mix.jsonl

cat data/processed/flan_v2/flan_v2_data.jsonl \
    data/processed/cot/cot_data.jsonl \
    data/processed/oasst1/oasst1_data.jsonl \
    data/processed/lima/lima_data.jsonl \
    > data/processed/tulu/tulu_v2_human_mix.jsonl


mkdir -p data/processed/wpq

cat data/processed/cot/cot_data.jsonl \
    data/processed/flan_v2/flan_v2_data.jsonl \
    > data/processed/wpq/cot_flanv2_data.jsonl