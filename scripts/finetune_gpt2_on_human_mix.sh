#!/bin/bash

: '
cd /dccstor/mit_fm/wpq/github/mitibm2023/external/open-instruct/scripts/ && \
		jbsub \
			-mem 64g \
			-cores 20+1 \
			-q x86_12h \
			-out finetune_gpt2_on_human_mix.txt \
			-name finetune_gpt2_on_human_mix \
			-require a100_40gb \
			bash finetune_gpt2_on_human_mix.sh
'

set -e # fail fully on first line failure

export OPENAI_API_KEY=$(cat ~/.openai_api_key)
export HF_HOME="/dccstor/mit_fm/wpq/hf_cache/"

source /dccstor/mit_fm/miniconda/bin/activate open-instruct

cd /dccstor/mit_fm/wpq/github/mitibm2023/external/open-instruct/

echo "Running on $(hostname)"
echo "accelerate launch     --mixed_precision bf16     --num_machines 1     --num_processes 1     open_instruct/finetune.py     --model_name_or_path gpt2-Large     --tokenizer_name gpt2-Large     --train_file data/processed/flanv2_cot_oasst1_dolly.jsonl     --max_seq_length 1024     --preprocessing_num_workers 16     --per_device_train_batch_size 2     --gradient_accumulation_steps 64     --learning_rate 2e-5     --lr_scheduler_type linear     --warmup_ratio 0.03     --weight_decay 0.     --num_train_epochs 2     --output_dir results/gpt2-Large_human_mix     --with_tracking     --report_to tensorboard     --logging_steps 1"

accelerate launch     --mixed_precision bf16     --num_machines 1     --num_processes 1     open_instruct/finetune.py     --model_name_or_path gpt2-Large     --tokenizer_name gpt2-Large     --train_file data/processed/flanv2_cot_oasst1_dolly.jsonl     --max_seq_length 1024     --preprocessing_num_workers 16     --per_device_train_batch_size 2     --gradient_accumulation_steps 64     --learning_rate 2e-5     --lr_scheduler_type linear     --warmup_ratio 0.03     --weight_decay 0.     --num_train_epochs 2     --output_dir results/gpt2-Large_human_mix     --with_tracking     --report_to tensorboard     --logging_steps 1