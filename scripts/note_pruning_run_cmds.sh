set -e
set -x
CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset oasst1 --sort_by random_s=0 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/oasst1

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset oasst1 --sort_by random_s=1 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/oasst1

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset oasst1 --sort_by random_s=2 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/oasst1

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset flan_v2 --sort_by random_s=0 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/flan_v2

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset flan_v2 --sort_by random_s=1 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/flan_v2

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset flan_v2 --sort_by random_s=2 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/flan_v2

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset dolly --sort_by random_s=0 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/dolly

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset dolly --sort_by random_s=1 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/dolly

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset dolly --sort_by random_s=2 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/dolly

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset ultrachat200kv2 --sort_by random_s=0 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/ultrachat200kv2

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset ultrachat200kv2 --sort_by random_s=1 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/ultrachat200kv2

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset ultrachat200kv2 --sort_by random_s=2 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/ultrachat200kv2