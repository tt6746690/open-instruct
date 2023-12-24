set -e
set -x
CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset starcoder_commentinstrv5 --sort_by random_s=0 --model_name codellama-7b+lora:r=256:a=256 --encode_fn_type sft --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/codellama-7b+lora:r=256:a=256/starcoder_commentinstrv5

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset starcoder_commentinstrv5 --sort_by random_s=1 --model_name codellama-7b+lora:r=256:a=256 --encode_fn_type sft --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/codellama-7b+lora:r=256:a=256/starcoder_commentinstrv5

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset starcoder_commentinstrv5 --sort_by random_s=2 --model_name codellama-7b+lora:r=256:a=256 --encode_fn_type sft --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/codellama-7b+lora:r=256:a=256/starcoder_commentinstrv5

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset starcoder_commentinstrv5 --sort_by log_prob --model_name codellama-7b+lora:r=256:a=256 --encode_fn_type sft --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/codellama-7b+lora:r=256:a=256/starcoder_commentinstrv5

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset starcoder_commentinstrv5 --sort_by logit_margin --model_name codellama-7b+lora:r=256:a=256 --encode_fn_type sft --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/codellama-7b+lora:r=256:a=256/starcoder_commentinstrv5

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset starcoder_commentinstrv5 --sort_by el2n_agg=mean --model_name codellama-7b+lora:r=256:a=256 --encode_fn_type sft --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/codellama-7b+lora:r=256:a=256/starcoder_commentinstrv5

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset starcoder_commentinstrv5 --sort_by grad_loraB_l2n --model_name codellama-7b+lora:r=256:a=256 --encode_fn_type sft --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/codellama-7b+lora:r=256:a=256/starcoder_commentinstrv5

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset starcoder_commentinstrv5 --sort_by ifd --model_name codellama-7b+lora:r=256:a=256 --encode_fn_type sft --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/codellama-7b+lora:r=256:a=256/starcoder_commentinstrv5

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset starcoder_commentinstrv5 --sort_by log_pmi --model_name codellama-7b+lora:r=256:a=256 --encode_fn_type sft --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/codellama-7b+lora:r=256:a=256/starcoder_commentinstrv5

CUDA_VISIBLE_DEVICES=5 python note_pruning.py --dataset starcoder_commentinstrv5 --sort_by numtoks --model_name codellama-7b+lora:r=256:a=256 --encode_fn_type sft --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/codellama-7b+lora:r=256:a=256/starcoder_commentinstrv5