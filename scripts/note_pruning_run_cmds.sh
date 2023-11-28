set -e
set -x
CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset tulu_v1_mix --sort_by dppmapbd_nc=200_k=vmf_gamma=5e-05_kmd=mpnet --model_name all-mpnet-base-v2 --encode_fn_type input --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/all-mpnet-base-v2/tulu_v1_mix