set -e
set -x
CUDA_VISIBLE_DEVICES=2 python note_pruning.py --dataset wizardlm --sort_by dppmap_k=rbf_gamma=0.031_kmd=llama7b_kemb=text+embedding --model_name llama-7b+lora:r=256:a=256 --encode_fn_type sft --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=256:a=256/wizardlm