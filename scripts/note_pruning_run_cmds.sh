set -e
set -x
CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset wizardlm --sort_by semdedup_cl=kmeansfaisscd_md=bge_dist=cd_emb=text+embedding_nc=50 --model_name bge-large-en-v1.5 --encode_fn_type input --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/bge-large-en-v1.5/wizardlm

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset wizardlm --sort_by semdedup_cl=kmeansfaisscd_md=bge_dist=cd_emb=grad+rp+loraB_nc=50 --model_name bge-large-en-v1.5 --encode_fn_type input --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/bge-large-en-v1.5/wizardlm

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset wizardlm --sort_by semdedup_cl=kmeansfaisscd_md=bge_dist=cd_emb=text+embedding_nc=150 --model_name bge-large-en-v1.5 --encode_fn_type input --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/bge-large-en-v1.5/wizardlm

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset wizardlm --sort_by semdedup_cl=kmeansfaisscd_md=bge_dist=cd_emb=grad+rp+loraB_nc=150 --model_name bge-large-en-v1.5 --encode_fn_type input --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/bge-large-en-v1.5/wizardlm