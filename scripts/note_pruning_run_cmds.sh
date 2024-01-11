set -e
set -x
CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset ultrafeedback --sort_by dppmap_k=vmf_gamma=1_kmd=llama7b+sharegptv2ep2+r512p4096_kemb=grad+rp+loraB --model_name llama-7b+sharegptv2ep2+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+sharegptv2ep2+lora:r=512:a=11585+proj=4096/ultrafeedback

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset ultrafeedback --sort_by dppmap_k=rbf_gamma=1e-3_kmd=llama7b+sharegptv2ep2+r512p4096_kemb=text+embedding --model_name llama-7b+sharegptv2ep2+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+sharegptv2ep2+lora:r=512:a=11585+proj=4096/ultrafeedback

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset ultrafeedback --sort_by dppmap_k=vmf_gamma=1_kmd=mpnet_kemb=text+embedding --model_name all-mpnet-base-v2 --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/all-mpnet-base-v2/ultrafeedback