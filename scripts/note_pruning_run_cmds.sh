set -e
set -x
CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset stanford_alpaca --sort_by dedup_md=llama7br256p4096_emb=text+embedding --model_name llama-7b+lora:r=256:a=4096+proj=4096 --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=256:a=4096+proj=4096/stanford_alpaca