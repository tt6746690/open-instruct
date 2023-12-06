set -e
set -x
CUDA_VISIBLE_DEVICES=4 python note_pruning.py --dataset wizardlm --sort_by dppmap_k=lin_gamma=1.0_kmd=llama7b_kemb=grad+rp+loraB --model_name all-mpnet-base-v2 --encode_fn_type input --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/all-mpnet-base-v2/wizardlm