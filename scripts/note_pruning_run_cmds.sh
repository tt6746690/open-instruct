set -e
set -x
CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset mix_redundant --sort_by random_s=0 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/mix_redundant

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset mix_redundant --sort_by random_s=1 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/mix_redundant

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset mix_redundant --sort_by random_s=2 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/mix_redundant

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset mix_redundant --sort_by log_prob --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/mix_redundant

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset mix_redundant --sort_by logit_margin --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/mix_redundant

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset mix_redundant --sort_by el2n_agg=mean --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/mix_redundant

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset mix_redundant --sort_by grad_loraB_l2n --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/mix_redundant

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset mix_redundant --sort_by numtoks --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/mix_redundant

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset mix_diverse --sort_by random_s=0 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/mix_diverse

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset mix_diverse --sort_by random_s=1 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/mix_diverse

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset mix_diverse --sort_by random_s=2 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/mix_diverse

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset mix_diverse --sort_by log_prob --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/mix_diverse

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset mix_diverse --sort_by logit_margin --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/mix_diverse

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset mix_diverse --sort_by el2n_agg=mean --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/mix_diverse

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset mix_diverse --sort_by grad_loraB_l2n --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/mix_diverse

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset mix_diverse --sort_by numtoks --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/mix_diverse