set -e
set -x
CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset ultrachat15 --sort_by dppmapbd_nc=200_theta=0.6_k=vmf_gamma=3.0_kmd=mpnet_q=prob_qmd=mistral7b --model_name all-mpnet-base-v2 --encode_fn_type input --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/all-mpnet-base-v2/ultrachat15

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset ultrachat15 --sort_by dppmapbd_nc=200_theta=0.9_k=vmf_gamma=3.0_kmd=mpnet_q=prob_qmd=mistral7b --model_name all-mpnet-base-v2 --encode_fn_type input --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/all-mpnet-base-v2/ultrachat15

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset ultrachat15 --sort_by dppmapbd_nc=200_theta=0.95_k=vmf_gamma=3.0_kmd=mpnet_q=prob_qmd=mistral7b --model_name all-mpnet-base-v2 --encode_fn_type input --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/all-mpnet-base-v2/ultrachat15