set -e
set -x
CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset wizardlm --sort_by dppmapbd_nc=200_theta=0.3_k=lin_kmd=mpnet_q=prob_qmd=llama7b+lima --model_name all-mpnet-base-v2 --encode_fn_type input --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/all-mpnet-base-v2/wizardlm

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset wizardlm --sort_by dppmapbd_nc=200_theta=0.6_k=lin_kmd=mpnet_q=prob_qmd=llama7b+lima --model_name all-mpnet-base-v2 --encode_fn_type input --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/all-mpnet-base-v2/wizardlm