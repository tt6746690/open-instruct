set -e
set -x
CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset flan_v2 --sort_by dppmap_k=vmf_gamma=1e-3_kmd=randspherep4096_kemb=grad+rp+loraB --model_name randspherep4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/randspherep4096/flan_v2 --overwrite

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset flan_v2 --sort_by dppmap_k=vmf_gamma=1_kmd=randspherep4096_kemb=grad+rp+loraB --model_name randspherep4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/randspherep4096/flan_v2 --overwrite

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset flan_v2 --sort_by dppmap_k=vmf_gamma=10_kmd=randspherep4096_kemb=grad+rp+loraB --model_name randspherep4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/randspherep4096/flan_v2 --overwrite

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset flan_v2 --sort_by dppmap_k=vmf_gamma=1e-2_kmd=randspherep4096_kemb=grad+rp+loraB --model_name randspherep4096 --save_dir /gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/data_inds/randspherep4096/flan_v2 --overwrite