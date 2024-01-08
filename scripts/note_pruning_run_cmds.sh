set -e
set -x
CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset wizardlm_alpaca --sort_by dppmap_k=vmf_gamma=1_kmd=llama7br512p4096_kemb=grad+rp+loraB+N50000 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/wizardlm_alpaca

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset wizardlm_alpaca --sort_by dppmap_k=rbf_gamma=1e-3_kmd=llama7br512p4096_kemb=text+embedding+N50000 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/wizardlm_alpaca