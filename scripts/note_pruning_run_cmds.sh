set -e
set -x
CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset tulu_v1_mix --sort_by dppmap_k=rbf_gamma=0.0001_kmd=llama7b_kemb=grad+rp+loraB --model_name llama-7b+lora:r=256:a=256 --encode_fn_type sft --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=256:a=256/tulu_v1_mix