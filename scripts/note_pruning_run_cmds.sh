set -e
set -x
CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset wizardlmv2 --sort_by numtoks --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/wizardlmv2