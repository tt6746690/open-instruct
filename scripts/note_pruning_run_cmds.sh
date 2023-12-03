set -e
set -x
CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset wizardlm --sort_by dppmap_nc=200_theta=0.3_k=lin_kmd=mpnet_q=log+prob_qmd=llama7b --model_name all-mpnet-base-v2 --encode_fn_type input --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/all-mpnet-base-v2/wizardlm

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset wizardlm --sort_by dppmap_nc=200_theta=0.6_k=lin_kmd=mpnet_q=log+prob_qmd=llama7b --model_name all-mpnet-base-v2 --encode_fn_type input --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/all-mpnet-base-v2/wizardlm