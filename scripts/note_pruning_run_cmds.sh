set -e
set -x
CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset sharegptv2 --sort_by dppmap_k=vmf_gamma=0.03_kmd=mpnet --model_name all-mpnet-base-v2 --encode_fn_type input --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/all-mpnet-base-v2/sharegptv2