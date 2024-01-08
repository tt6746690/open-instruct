set -e
set -x
CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset gpt4_alpaca --sort_by dppmap_k=vmf_gamma=1_kmd=mpnet_kemb=text+embedding+N50000 --model_name all-mpnet-base-v2 --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/all-mpnet-base-v2/gpt4_alpaca

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset self_instruct --sort_by dppmap_k=vmf_gamma=1_kmd=mpnet_kemb=text+embedding+N50000 --model_name all-mpnet-base-v2 --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/all-mpnet-base-v2/self_instruct

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset unnatural_instructions --sort_by dppmap_k=vmf_gamma=1_kmd=mpnet_kemb=text+embedding+N50000 --model_name all-mpnet-base-v2 --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/all-mpnet-base-v2/unnatural_instructions

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset wizardlm_alpaca --sort_by dppmap_k=vmf_gamma=1_kmd=mpnet_kemb=text+embedding+N50000 --model_name all-mpnet-base-v2 --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/all-mpnet-base-v2/wizardlm_alpaca

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset stanford_alpaca --sort_by dppmap_k=vmf_gamma=1_kmd=mpnet_kemb=text+embedding+N50000 --model_name all-mpnet-base-v2 --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/all-mpnet-base-v2/stanford_alpaca

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset sharegptv2 --sort_by dppmap_k=vmf_gamma=1_kmd=mpnet_kemb=text+embedding+N50000 --model_name all-mpnet-base-v2 --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/all-mpnet-base-v2/sharegptv2

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset wizardlmv2 --sort_by dppmap_k=vmf_gamma=1_kmd=mpnet_kemb=text+embedding+N50000 --model_name all-mpnet-base-v2 --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/all-mpnet-base-v2/wizardlmv2

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset oasst1 --sort_by dppmap_k=vmf_gamma=1_kmd=mpnet_kemb=text+embedding+N50000 --model_name all-mpnet-base-v2 --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/all-mpnet-base-v2/oasst1

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset flan_v2 --sort_by dppmap_k=vmf_gamma=1_kmd=mpnet_kemb=text+embedding+N50000 --model_name all-mpnet-base-v2 --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/all-mpnet-base-v2/flan_v2

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset dolly --sort_by dppmap_k=vmf_gamma=1_kmd=mpnet_kemb=text+embedding+N50000 --model_name all-mpnet-base-v2 --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/all-mpnet-base-v2/dolly

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset ultrachat200kv2 --sort_by dppmap_k=vmf_gamma=1_kmd=mpnet_kemb=text+embedding+N50000 --model_name all-mpnet-base-v2 --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/all-mpnet-base-v2/ultrachat200kv2

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset lima --sort_by dppmap_k=vmf_gamma=1_kmd=mpnet_kemb=text+embedding+N50000 --model_name all-mpnet-base-v2 --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/all-mpnet-base-v2/lima