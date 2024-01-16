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


echo "Downloading the flan_v2 collection, here we use two subsampled versions: for tulu v1 we subsampled 100K, for tulu v2 we subsampled 50K..."
mkdir -p data/raw_train/flan_v2/
wget -O data/raw_train/flan_v2/tulu_v1_resampled_flan_100k.jsonl https://beaker.org/api/v3/datasets/01GZTTS2EJFPA83PXS4FQCS1SA/files/flan_v2_resampled_100k.jsonl
wget -O data/raw_train/flan_v2/tulu_v2_resampled_flan_50k.jsonl https://beaker.org/api/v3/datasets/01HBS0N5ZSDF5AECA9VMB1RKXQ/files/flan_v2_resampled_50k.jsonl


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
wget -P data/raw_train/sharegpt/ https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/HTML_cleaned_raw_dataset/sg_90k_part1_html_cleaned.json
wget -P data/raw_train/sharegpt/ https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/HTML_cleaned_raw_dataset/sg_90k_part2_html_cleaned.json
echo "Splitting the ShareGPT dataset with 2048 max tokens per conversation..."
python scripts/split_sharegpt_conversations.py \
    --in-files data/raw_train/sharegpt/sg_90k_part1_html_cleaned.json data/raw_train/sharegpt/sg_90k_part2_html_cleaned.json \
    --out-file data/raw_train/sharegpt/sharegpt_html_cleaned_and_split_2048.json \
    --model-name-or-path results/baselines/huggyllama/llama-7b \
    --max-length 2048
echo "Splitting the ShareGPT dataset with 4096 max tokens per conversation..."
python scripts/split_sharegpt_conversations.py \
    --in-files data/raw_train/sharegpt/sg_90k_part1_html_cleaned.json data/raw_train/sharegpt/sg_90k_part2_html_cleaned.json \
    --out-file data/raw_train/sharegpt/sharegpt_html_cleaned_and_split_4096.json \
    --model-name-or-path results/baselines/huggyllama/llama-7b \
    --max-length 4096


echo "Downloading LIMA dataset..."
wget --header="Authorization: Bearer $HF_TOKEN" -P data/raw_train/lima/ https://huggingface.co/datasets/GAIR/lima/raw/main/train.jsonl


echo "Downloading WizardLM dataset..."
wget -P data/raw_train/wizardlm/ https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_V2_196k/resolve/main/WizardLM_evol_instruct_V2_143k.json


echo "Downloading the OpenOrca dataset..."
wget -P data/raw_train/open_orca/ https://huggingface.co/datasets/Open-Orca/OpenOrca/resolve/main/1M-GPT4-Augmented.parquet
wget -P data/raw_train/open_orca/ https://huggingface.co/datasets/Open-Orca/OpenOrca/resolve/main/3_5M-GPT3_5-Augmented.parquet


echo "Downloading the UltraChat dataset..."
# wget -P data/raw_train/ultrachat https://cloud.tsinghua.edu.cn/f/0a27393192ad46a5a081/?dl=1
# wget -P data/raw_train/ultrachat https://cloud.tsinghua.edu.cn/f/57258a87846243218a9b/?dl=1
# wget -P data/raw_train/ultrachat https://cloud.tsinghua.edu.cn/f/099b4dd71b82448fb7fb/?dl=1
# wget -P data/raw_train/ultrachat https://cloud.tsinghua.edu.cn/f/1f7abdf2d2564cb4b338/?dl=1
wget -P data/raw_train/ultrachat/full https://huggingface.co/datasets/stingning/ultrachat/resolve/main/train_0.jsonl
wget -P data/raw_train/ultrachat/full https://huggingface.co/datasets/stingning/ultrachat/resolve/main/train_1.jsonl
wget -P data/raw_train/ultrachat/full https://huggingface.co/datasets/stingning/ultrachat/resolve/main/train_2.jsonl
wget -P data/raw_train/ultrachat/full https://huggingface.co/datasets/stingning/ultrachat/resolve/main/train_3.jsonl
wget -P data/raw_train/ultrachat/full https://huggingface.co/datasets/stingning/ultrachat/resolve/main/train_4.jsonl
wget -P data/raw_train/ultrachat/full https://huggingface.co/datasets/stingning/ultrachat/resolve/main/train_5.jsonl
wget -P data/raw_train/ultrachat/full https://huggingface.co/datasets/stingning/ultrachat/resolve/main/train_6.jsonl
wget -P data/raw_train/ultrachat/full https://huggingface.co/datasets/stingning/ultrachat/resolve/main/train_7.jsonl
wget -P data/raw_train/ultrachat/full https://huggingface.co/datasets/stingning/ultrachat/resolve/main/train_8.jsonl
wget -P data/raw_train/ultrachat/full https://huggingface.co/datasets/stingning/ultrachat/resolve/main/train_9.jsonl

# split long conversations
output_dir="data/raw_train/ultrachat/full_splitlongconv_2048/"
mkdir -p "$output_dir"
for input_file in data/raw_train/ultrachat/full/*.jsonl; do
    output_file="$output_dir$(basename "$input_file")"
    python scripts/split_ultrachat_conversations.py --in-file "$input_file" --out-file "$output_file" --model-name-or-path "results/baselines/huggyllama/llama-7b" --max-length 2048
done


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
    --in-files HuggingFaceH4/ultrachat_200k_train_sft \
    --out-file data/raw_train/ultrachat/ultrachat_200k_train_splitlongconv.json \
    --model-name-or-path results/baselines/huggyllama/llama-7b \
    --max-length 2048
python scripts/split_sharegpt_conversations.py \
    --in-files HuggingFaceH4/ultrachat_200k_test_sft \
    --out-file data/raw_train/ultrachat/ultrachat_200k_test_splitlongconv.json \
    --model-name-or-path results/baselines/huggyllama/llama-7b \
    --max-length 2048


wget -P data/raw_train/open_orca/ https://huggingface.co/datasets/Open-Orca/SlimOrca/resolve/main/oo-labeled_correct.gpt4.sharegpt.jsonl


echo "Downloading the Science Instructions dataset..."
wget -P data/raw_train/science https://beaker.org/api/v3/datasets/01HBS3G7TA8AT15C7RWTJAN66X/files/science_train.jsonl


echo "Downloading the HardCoded dataset..."
wget -P data/raw_train/hard_coded/ https://beaker.org/api/v3/datasets/01HBS14BBV16K45MMFSYJR86CA/files/hard_coded_examples.xlsx


echo "Downloading the Alpagasus dataset..."
wget -P data/raw_train/alpagasus https://raw.githubusercontent.com/gpt4life/alpagasus/main/data/filtered/chatgpt_9k.json


## preference dataset

echo "Downloading the UltraFeedback dataset..."
wget -P data/raw_train/ultrafeedback/ https://huggingface.co/datasets/allenai/ultrafeedback_binarized_cleaned/resolve/main/data/train_prefs-00000-of-00001.parquet?download=true -O data/raw_train/ultrafeedback/allenai_ultrafeedback_binarized_cleaned_train_prefs.parquet


echo "Downloading the openai/summarize_from_feedback dataset..."
mkdir -p data/raw_train/openai_sum/
wget -P data/raw_train/openai_sum/ https://huggingface.co/datasets/openai/summarize_from_feedback/resolve/refs%2Fconvert%2Fparquet/comparisons/train/0000.parquet?download=true -O data/raw_train/openai_sum/openai_summarize_from_feedback_train.parquet
wget -P data/raw_train/openai_sum/ https://huggingface.co/datasets/openai/summarize_from_feedback/resolve/refs%2Fconvert%2Fparquet/comparisons/validation/0000.parquet?download=true -O data/raw_train/openai_sum/openai_summarize_from_feedback_validation.parquet


echo "Downloading the HH-RLHF dataset..."
mkdir -p data/raw_train/hh_rlhf/
wget -P data/raw_train/hh_rlhf/ https://huggingface.co/datasets/PKU-Alignment/processed-hh-rlhf/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet?download=true -O data/raw_train/hh_rlhf/hh_rlhf_train.parquet
wget -P data/raw_train/hh_rlhf/ https://huggingface.co/datasets/PKU-Alignment/processed-hh-rlhf/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet?download=true -O data/raw_train/hh_rlhf/hh_rlhf_test.parquet


echo "Downloading SHP dataset..."
mkdir -p data/raw_train/shp/
wget -P data/raw_train/shp/ https://huggingface.co/datasets/stanfordnlp/SHP/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet?download=true -O data/raw_train/shp/shp_train_0.parquet
wget -P data/raw_train/shp/ https://huggingface.co/datasets/stanfordnlp/SHP/resolve/refs%2Fconvert%2Fparquet/default/train/0001.parquet?download=true -O data/raw_train/shp/shp_train_1.parquet

echo "Downloading alpagasus dataset..."
mkdir -p data/raw_train/alpagasus/
wget -P data/raw_train/alpagasus/ https://raw.githubusercontent.com/gpt4life/alpagasus/main/data/filtered/chatgpt_9k.json

echo "Downloading HelpSteer dataset..."
mkdir -p data/raw_train/HelpSteer/
wget -P data/raw_train/HelpSteer/ https://huggingface.co/datasets/nvidia/HelpSteer/blob/refs%2Fconvert%2Fparquet/default/train/0000.parquet?download=true -O data/raw_train/HelpSteer/HelpSteer_train.parquet
wget -p data/raw_train/HelpSteer/ https://huggingface.co/datasets/nvidia/HelpSteer/blob/refs%2Fconvert%2Fparquet/default/validation/0000.parquet?download=true -O data/raw_train/HelpSteer/HelpSteer_val.parquet

echo "Downloading orca_dpo_pairs dataset..."
mkdir -p data/raw_train/orca_dpo_pairs/
wget -P data/raw_train/orca_dpo_pairs/ https://huggingface.co/datasets/Intel/orca_dpo_pairs/blob/refs%2Fconvert%2Fparquet/default/train/0000.parquet?download=true -O data/raw_train/orca_dpo_pairs/orca_dpo_pairs_train.parquet

echo "Downloading DEITA6k/10k dataset..."
mkdir -p data/raw_train/DEITA6k/
wget -P data/raw_train/DEITA6k/ https://huggingface.co/datasets/hkust-nlp/deita-6k-v0/resolve/main/deita_6k.json

mkdir -p data/raw_train/DEITA10k/
wget -P data/raw_train/DEITA10k/ https://huggingface.co/datasets/hkust-nlp/deita-10k-v0/resolve/main/deita_10k.json

echo "Processing datasets..."
python open_instruct/reformat_datasets.py --raw_data_dir data/raw_train/ --output_dir data/processed/
