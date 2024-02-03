set -e
set -x
CUDA_VISIBLE_DEVICES=0 python note_pruning.py --dataset stanford_alpaca50k --sort_by dppmap_k=vmf_gamma=1_theta=0.1_kmd=llama7br512p4096_kemb=grad+rp+loraB_q=log+prob_qmd=llama7br512p4096 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /dccstor/data-pruning/wpq/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/stanford_alpaca50k

CUDA_VISIBLE_DEVICES=0 python note_pruning.py --dataset stanford_alpaca50k --sort_by dppmap_k=vmf_gamma=1_theta=0.3_kmd=llama7br512p4096_kemb=grad+rp+loraB_q=log+prob_qmd=llama7br512p4096 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /dccstor/data-pruning/wpq/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/stanford_alpaca50k

CUDA_VISIBLE_DEVICES=0 python note_pruning.py --dataset stanford_alpaca50k --sort_by dppmap_k=vmf_gamma=1_theta=0.6_kmd=llama7br512p4096_kemb=grad+rp+loraB_q=log+prob_qmd=llama7br512p4096 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /dccstor/data-pruning/wpq/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/stanford_alpaca50k

CUDA_VISIBLE_DEVICES=0 python note_pruning.py --dataset stanford_alpaca50k --sort_by dppmap_k=vmf_gamma=1_theta=0.9_kmd=llama7br512p4096_kemb=grad+rp+loraB_q=log+prob_qmd=llama7br512p4096 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /dccstor/data-pruning/wpq/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/stanford_alpaca50k

CUDA_VISIBLE_DEVICES=0 python note_pruning.py --dataset ultrachat50k --sort_by dppmap_k=vmf_gamma=1_theta=0.1_kmd=llama7br512p4096_kemb=grad+rp+loraB_q=log+prob_qmd=llama7br512p4096 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /dccstor/data-pruning/wpq/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/ultrachat50k

CUDA_VISIBLE_DEVICES=0 python note_pruning.py --dataset ultrachat50k --sort_by dppmap_k=vmf_gamma=1_theta=0.3_kmd=llama7br512p4096_kemb=grad+rp+loraB_q=log+prob_qmd=llama7br512p4096 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /dccstor/data-pruning/wpq/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/ultrachat50k

CUDA_VISIBLE_DEVICES=0 python note_pruning.py --dataset ultrachat50k --sort_by dppmap_k=vmf_gamma=1_theta=0.6_kmd=llama7br512p4096_kemb=grad+rp+loraB_q=log+prob_qmd=llama7br512p4096 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /dccstor/data-pruning/wpq/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/ultrachat50k

CUDA_VISIBLE_DEVICES=0 python note_pruning.py --dataset ultrachat50k --sort_by dppmap_k=vmf_gamma=1_theta=0.9_kmd=llama7br512p4096_kemb=grad+rp+loraB_q=log+prob_qmd=llama7br512p4096 --model_name llama-7b+lora:r=512:a=11585+proj=4096 --save_dir /dccstor/data-pruning/wpq/github/mitibm2023/external/open-instruct/scripts/data_inds/llama-7b+lora:r=512:a=11585+proj=4096/ultrachat50k