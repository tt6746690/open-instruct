set -e
set -x
CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset open_orca_slim --sort_by dppmap_k=vmf_gamma=auto10000_kmd=llama7b_kemb=grad+rp+loraB --model_name llama-7b+lora:r=256:a=256 --encode_fn_type sft --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=256:a=256/open_orca_slim