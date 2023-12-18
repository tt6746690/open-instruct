set -e
set -x
CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset tulu_v2 --sort_by numtoks --model_name llama-7b+lora:r=256:a=256 --encode_fn_type sft --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=256:a=256/tulu_v2