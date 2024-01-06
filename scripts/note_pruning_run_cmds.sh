set -e
set -x
CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset stanford_alpaca --sort_by dppmap_k=vmf_gamma=1_kmd=randspherep4096_kemb=grad+rp+loraB --model_name randspherep4096 --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/randspherep4096/stanford_alpaca

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset stanford_alpaca --sort_by dppmap_k=vmf_gamma=1_kmd=randspherep4096_kemb=grad+rp+loraB_ord=random:s@0 --model_name randspherep4096 --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/randspherep4096/stanford_alpaca

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset flan_v2 --sort_by dppmap_k=vmf_gamma=1_kmd=randspherep4096_kemb=grad+rp+loraB --model_name randspherep4096 --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/randspherep4096/flan_v2

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset flan_v2 --sort_by dppmap_k=vmf_gamma=1_kmd=randspherep4096_kemb=grad+rp+loraB_ord=random:s@0 --model_name randspherep4096 --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/randspherep4096/flan_v2

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset ultrachat200kv2 --sort_by dppmap_k=vmf_gamma=1_kmd=randspherep4096_kemb=grad+rp+loraB --model_name randspherep4096 --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/randspherep4096/ultrachat200kv2

CUDA_VISIBLE_DEVICES=1 python note_pruning.py --dataset ultrachat200kv2 --sort_by dppmap_k=vmf_gamma=1_kmd=randspherep4096_kemb=grad+rp+loraB_ord=random:s@0 --model_name randspherep4096 --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/randspherep4096/ultrachat200kv2