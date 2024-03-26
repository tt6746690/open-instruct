set -e
set -x
CUDA_VISIBLE_DEVICES=4 python note_pruning.py --dataset stanford_alpaca50k --sort_by dppmap_k=vmf_gamma=10_kmd=mistral7br512p4096_kemb=text+embedding --model_name mistral-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/mistral-7b+lora:r=512:a=11585+proj=4096/stanford_alpaca50k

CUDA_VISIBLE_DEVICES=4 python note_pruning.py --dataset stanford_alpaca50k --sort_by dppmap_k=rbf_gamma=1e-2_kmd=mistral7br512p4096_kemb=grad+rp+loraB --model_name mistral-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/mistral-7b+lora:r=512:a=11585+proj=4096/stanford_alpaca50k

CUDA_VISIBLE_DEVICES=4 python note_pruning.py --dataset stanford_alpaca50k --sort_by dppmap_k=vmf_gamma=1_theta=0.1_kmd=mistral7br512p4096_kemb=grad+rp+loraB_q=numtoks+output_qmd=mistral7br512p4096 --model_name mistral-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/mistral-7b+lora:r=512:a=11585+proj=4096/stanford_alpaca50k

CUDA_VISIBLE_DEVICES=4 python note_pruning.py --dataset stanford_alpaca50k --sort_by dppmap_k=vmf_gamma=1_theta=0.3_kmd=mistral7br512p4096_kemb=grad+rp+loraB_q=numtoks+output_qmd=mistral7br512p4096 --model_name mistral-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/mistral-7b+lora:r=512:a=11585+proj=4096/stanford_alpaca50k

CUDA_VISIBLE_DEVICES=4 python note_pruning.py --dataset stanford_alpaca50k --sort_by dppmap_k=vmf_gamma=1_theta=0.6_kmd=mistral7br512p4096_kemb=grad+rp+loraB_q=numtoks+output_qmd=mistral7br512p4096 --model_name mistral-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/mistral-7b+lora:r=512:a=11585+proj=4096/stanford_alpaca50k

CUDA_VISIBLE_DEVICES=4 python note_pruning.py --dataset stanford_alpaca50k --sort_by dppmap_k=vmf_gamma=1_theta=0.9_kmd=mistral7br512p4096_kemb=grad+rp+loraB_q=numtoks+output_qmd=mistral7br512p4096 --model_name mistral-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/mistral-7b+lora:r=512:a=11585+proj=4096/stanford_alpaca50k

CUDA_VISIBLE_DEVICES=4 python note_pruning.py --dataset ultrachat50k --sort_by dppmap_k=vmf_gamma=10_kmd=mistral7br512p4096_kemb=text+embedding --model_name mistral-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/mistral-7b+lora:r=512:a=11585+proj=4096/ultrachat50k

CUDA_VISIBLE_DEVICES=4 python note_pruning.py --dataset ultrachat50k --sort_by dppmap_k=rbf_gamma=1e-2_kmd=mistral7br512p4096_kemb=grad+rp+loraB --model_name mistral-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/mistral-7b+lora:r=512:a=11585+proj=4096/ultrachat50k

CUDA_VISIBLE_DEVICES=4 python note_pruning.py --dataset ultrachat50k --sort_by dppmap_k=vmf_gamma=1_theta=0.1_kmd=mistral7br512p4096_kemb=grad+rp+loraB_q=numtoks+output_qmd=mistral7br512p4096 --model_name mistral-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/mistral-7b+lora:r=512:a=11585+proj=4096/ultrachat50k

CUDA_VISIBLE_DEVICES=4 python note_pruning.py --dataset ultrachat50k --sort_by dppmap_k=vmf_gamma=1_theta=0.3_kmd=mistral7br512p4096_kemb=grad+rp+loraB_q=numtoks+output_qmd=mistral7br512p4096 --model_name mistral-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/mistral-7b+lora:r=512:a=11585+proj=4096/ultrachat50k

CUDA_VISIBLE_DEVICES=4 python note_pruning.py --dataset ultrachat50k --sort_by dppmap_k=vmf_gamma=1_theta=0.6_kmd=mistral7br512p4096_kemb=grad+rp+loraB_q=numtoks+output_qmd=mistral7br512p4096 --model_name mistral-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/mistral-7b+lora:r=512:a=11585+proj=4096/ultrachat50k

CUDA_VISIBLE_DEVICES=4 python note_pruning.py --dataset ultrachat50k --sort_by dppmap_k=vmf_gamma=1_theta=0.9_kmd=mistral7br512p4096_kemb=grad+rp+loraB_q=numtoks+output_qmd=mistral7br512p4096 --model_name mistral-7b+lora:r=512:a=11585+proj=4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/mistral-7b+lora:r=512:a=11585+proj=4096/ultrachat50k