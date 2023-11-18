set -e
set -x
CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset ultrachat15 --sort_by semdedup_cl=kmeansfaisscd_md=mistral7b_dist=cd_emb=text+embedding_nc=100 --model_name mistral-7b+lora:r=256:a=256 --encode_fn_type sft --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/mistral-7b+lora:r=256:a=256/ultrachat15

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset ultrachat15 --sort_by semdedup_cl=kmeansfaisscd_md=mistral7b_dist=cd_emb=grad+rp+loraB_nc=100 --model_name mistral-7b+lora:r=256:a=256 --encode_fn_type sft --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/mistral-7b+lora:r=256:a=256/ultrachat15

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset ultrachat15 --sort_by semdedup_cl=kmeansfaisscd_md=mistral7b_dist=cd_emb=text+embedding_nc=200 --model_name mistral-7b+lora:r=256:a=256 --encode_fn_type sft --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/mistral-7b+lora:r=256:a=256/ultrachat15

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset ultrachat15 --sort_by semdedup_cl=kmeansfaisscd_md=mistral7b_dist=cd_emb=grad+rp+loraB_nc=200 --model_name mistral-7b+lora:r=256:a=256 --encode_fn_type sft --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/mistral-7b+lora:r=256:a=256/ultrachat15

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset ultrachat15 --sort_by semdedup_cl=kmeansfaisscd_md=mistral7b_dist=cd_emb=text+embedding_nc=300 --model_name mistral-7b+lora:r=256:a=256 --encode_fn_type sft --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/mistral-7b+lora:r=256:a=256/ultrachat15

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset ultrachat15 --sort_by semdedup_cl=kmeansfaisscd_md=mistral7b_dist=cd_emb=grad+rp+loraB_nc=300 --model_name mistral-7b+lora:r=256:a=256 --encode_fn_type sft --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/mistral-7b+lora:r=256:a=256/ultrachat15

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset ultrachat15 --sort_by semdedup_cl=kmeansfaisscd_md=mistral7b_dist=cd_emb=text+embedding_nc=400 --model_name mistral-7b+lora:r=256:a=256 --encode_fn_type sft --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/mistral-7b+lora:r=256:a=256/ultrachat15

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset ultrachat15 --sort_by semdedup_cl=kmeansfaisscd_md=mistral7b_dist=cd_emb=grad+rp+loraB_nc=400 --model_name mistral-7b+lora:r=256:a=256 --encode_fn_type sft --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/mistral-7b+lora:r=256:a=256/ultrachat15

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset ultrachat15 --sort_by semdedup_cl=kmeansfaisscd_md=mistral7b_dist=cd_emb=text+embedding_nc=500 --model_name mistral-7b+lora:r=256:a=256 --encode_fn_type sft --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/mistral-7b+lora:r=256:a=256/ultrachat15

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset ultrachat15 --sort_by semdedup_cl=kmeansfaisscd_md=mistral7b_dist=cd_emb=grad+rp+loraB_nc=500 --model_name mistral-7b+lora:r=256:a=256 --encode_fn_type sft --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/mistral-7b+lora:r=256:a=256/ultrachat15

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset ultrachat15 --sort_by semdedup_cl=kmeansfaisscd_md=mistral7b_dist=cd_emb=text+embedding_nc=600 --model_name mistral-7b+lora:r=256:a=256 --encode_fn_type sft --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/mistral-7b+lora:r=256:a=256/ultrachat15

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset ultrachat15 --sort_by semdedup_cl=kmeansfaisscd_md=mistral7b_dist=cd_emb=grad+rp+loraB_nc=600 --model_name mistral-7b+lora:r=256:a=256 --encode_fn_type sft --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/mistral-7b+lora:r=256:a=256/ultrachat15