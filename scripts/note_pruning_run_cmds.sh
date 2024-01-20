set -e
set -x
CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset flan_v250k --sort_by random_s=0 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/flan_v250k

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset flan_v250k --sort_by random_s=1 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/flan_v250k

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset flan_v250k --sort_by random_s=2 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/flan_v250k

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset flan_v250k --sort_by log_prob --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/flan_v250k

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset flan_v250k --sort_by logit_margin --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/flan_v250k

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset flan_v250k --sort_by el2n_agg=mean --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/flan_v250k

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset flan_v250k --sort_by grad_loraB_l2n --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/flan_v250k

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset flan_v250k --sort_by numtoks --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/flan_v250k

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset stanford_alpaca50k --sort_by random_s=0 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/stanford_alpaca50k

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset stanford_alpaca50k --sort_by random_s=1 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/stanford_alpaca50k

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset stanford_alpaca50k --sort_by random_s=2 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/stanford_alpaca50k

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset stanford_alpaca50k --sort_by log_prob --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/stanford_alpaca50k

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset stanford_alpaca50k --sort_by logit_margin --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/stanford_alpaca50k

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset stanford_alpaca50k --sort_by el2n_agg=mean --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/stanford_alpaca50k

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset stanford_alpaca50k --sort_by grad_loraB_l2n --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/stanford_alpaca50k

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset stanford_alpaca50k --sort_by numtoks --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/stanford_alpaca50k

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset oasst2 --sort_by random_s=0 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/oasst2

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset oasst2 --sort_by random_s=1 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/oasst2

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset oasst2 --sort_by random_s=2 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/oasst2

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset oasst2 --sort_by log_prob --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/oasst2

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset oasst2 --sort_by logit_margin --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/oasst2

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset oasst2 --sort_by el2n_agg=mean --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/oasst2

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset oasst2 --sort_by grad_loraB_l2n --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/oasst2

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset oasst2 --sort_by numtoks --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/oasst2

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset wizardlm50k --sort_by random_s=0 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/wizardlm50k

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset wizardlm50k --sort_by random_s=1 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/wizardlm50k

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset wizardlm50k --sort_by random_s=2 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/wizardlm50k

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset wizardlm50k --sort_by log_prob --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/wizardlm50k

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset wizardlm50k --sort_by logit_margin --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/wizardlm50k

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset wizardlm50k --sort_by el2n_agg=mean --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/wizardlm50k

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset wizardlm50k --sort_by grad_loraB_l2n --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/wizardlm50k

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset wizardlm50k --sort_by numtoks --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/wizardlm50k

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset sharegpt50k --sort_by random_s=0 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/sharegpt50k

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset sharegpt50k --sort_by random_s=1 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/sharegpt50k

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset sharegpt50k --sort_by random_s=2 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/sharegpt50k

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset sharegpt50k --sort_by log_prob --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/sharegpt50k

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset sharegpt50k --sort_by logit_margin --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/sharegpt50k

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset sharegpt50k --sort_by el2n_agg=mean --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/sharegpt50k

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset sharegpt50k --sort_by grad_loraB_l2n --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/sharegpt50k

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset sharegpt50k --sort_by numtoks --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/sharegpt50k

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset ultrachat50k --sort_by random_s=0 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/ultrachat50k

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset ultrachat50k --sort_by random_s=1 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/ultrachat50k

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset ultrachat50k --sort_by random_s=2 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/ultrachat50k

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset ultrachat50k --sort_by log_prob --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/ultrachat50k

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset ultrachat50k --sort_by logit_margin --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/ultrachat50k

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset ultrachat50k --sort_by el2n_agg=mean --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/ultrachat50k

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset ultrachat50k --sort_by grad_loraB_l2n --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/ultrachat50k

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset ultrachat50k --sort_by numtoks --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/ultrachat50k