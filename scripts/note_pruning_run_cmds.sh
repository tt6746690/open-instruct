set -e
set -x
CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset mix_all50k --sort_by dppmap_k=vmf_gamma=1_kmd=llama7br512p4096_kemb=grad+rp+loraB --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/mix_all50k

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset mix_all50k --sort_by dppmap_k=vmf_gamma=10_kmd=llama7br512p4096_kemb=text+embedding --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/mix_all50k