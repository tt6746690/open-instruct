set -e
set -x
CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset dolly --sort_by log_prob --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/dolly

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset dolly --sort_by logit_margin --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/dolly

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset dolly --sort_by el2n_agg=mean --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/dolly

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset dolly --sort_by grad_loraB_l2n --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/dolly

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset dolly --sort_by numtoks --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/dolly

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset gpt4_alpaca50k --sort_by log_prob --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/gpt4_alpaca50k

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset gpt4_alpaca50k --sort_by logit_margin --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/gpt4_alpaca50k

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset gpt4_alpaca50k --sort_by el2n_agg=mean --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/gpt4_alpaca50k

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset gpt4_alpaca50k --sort_by grad_loraB_l2n --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/gpt4_alpaca50k

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset gpt4_alpaca50k --sort_by numtoks --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/gpt4_alpaca50k

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset self_instruct50k --sort_by log_prob --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/self_instruct50k

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset self_instruct50k --sort_by logit_margin --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/self_instruct50k

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset self_instruct50k --sort_by el2n_agg=mean --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/self_instruct50k

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset self_instruct50k --sort_by grad_loraB_l2n --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/self_instruct50k

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset self_instruct50k --sort_by numtoks --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/self_instruct50k