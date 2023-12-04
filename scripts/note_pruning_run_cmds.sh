set -e
set -x
CUDA_VISIBLE_DEVICES=3 python note_pruning.py --dataset wizardlm --sort_by dedup_md=mpnet_emb=text+embedding --model_name all-mpnet-base-v2 --encode_fn_type input --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/all-mpnet-base-v2/wizardlm