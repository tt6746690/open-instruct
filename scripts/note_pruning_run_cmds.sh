set -e
set -x
CUDA_VISIBLE_DEVICES=4 python note_pruning.py --dataset starcoder_commentinstr --sort_by random_s=0 --model_name codellama-7b+lora:r=256:a=256 --encode_fn_type sft --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/codellama-7b+lora:r=256:a=256/starcoder_commentinstr

CUDA_VISIBLE_DEVICES=4 python note_pruning.py --dataset starcoder_commentinstr --sort_by random_s=1 --model_name codellama-7b+lora:r=256:a=256 --encode_fn_type sft --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/codellama-7b+lora:r=256:a=256/starcoder_commentinstr

CUDA_VISIBLE_DEVICES=4 python note_pruning.py --dataset starcoder_commentinstr --sort_by random_s=2 --model_name codellama-7b+lora:r=256:a=256 --encode_fn_type sft --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/codellama-7b+lora:r=256:a=256/starcoder_commentinstr

CUDA_VISIBLE_DEVICES=4 python note_pruning.py --dataset starcoder_commentinstr --sort_by log_prob --model_name codellama-7b+lora:r=256:a=256 --encode_fn_type sft --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/codellama-7b+lora:r=256:a=256/starcoder_commentinstr

CUDA_VISIBLE_DEVICES=4 python note_pruning.py --dataset starcoder_commentinstr --sort_by logit_margin --model_name codellama-7b+lora:r=256:a=256 --encode_fn_type sft --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/codellama-7b+lora:r=256:a=256/starcoder_commentinstr

CUDA_VISIBLE_DEVICES=4 python note_pruning.py --dataset starcoder_commentinstr --sort_by el2n_agg=mean --model_name codellama-7b+lora:r=256:a=256 --encode_fn_type sft --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/codellama-7b+lora:r=256:a=256/starcoder_commentinstr

CUDA_VISIBLE_DEVICES=4 python note_pruning.py --dataset starcoder_commentinstr --sort_by grad_loraB_l2n --model_name codellama-7b+lora:r=256:a=256 --encode_fn_type sft --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/codellama-7b+lora:r=256:a=256/starcoder_commentinstr

CUDA_VISIBLE_DEVICES=4 python note_pruning.py --dataset starcoder_commentinstr --sort_by ifd --model_name codellama-7b+lora:r=256:a=256 --encode_fn_type sft --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/codellama-7b+lora:r=256:a=256/starcoder_commentinstr

CUDA_VISIBLE_DEVICES=4 python note_pruning.py --dataset starcoder_commentinstr --sort_by log_pmi --model_name codellama-7b+lora:r=256:a=256 --encode_fn_type sft --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/codellama-7b+lora:r=256:a=256/starcoder_commentinstr

CUDA_VISIBLE_DEVICES=4 python note_pruning.py --dataset starcoder_commentinstr_cleaned --sort_by random_s=0 --model_name codellama-7b+lora:r=256:a=256 --encode_fn_type sft --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/codellama-7b+lora:r=256:a=256/starcoder_commentinstr_cleaned

CUDA_VISIBLE_DEVICES=4 python note_pruning.py --dataset starcoder_commentinstr_cleaned --sort_by random_s=1 --model_name codellama-7b+lora:r=256:a=256 --encode_fn_type sft --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/codellama-7b+lora:r=256:a=256/starcoder_commentinstr_cleaned

CUDA_VISIBLE_DEVICES=4 python note_pruning.py --dataset starcoder_commentinstr_cleaned --sort_by random_s=2 --model_name codellama-7b+lora:r=256:a=256 --encode_fn_type sft --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/codellama-7b+lora:r=256:a=256/starcoder_commentinstr_cleaned

CUDA_VISIBLE_DEVICES=4 python note_pruning.py --dataset starcoder_commentinstr_cleaned --sort_by log_prob --model_name codellama-7b+lora:r=256:a=256 --encode_fn_type sft --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/codellama-7b+lora:r=256:a=256/starcoder_commentinstr_cleaned

CUDA_VISIBLE_DEVICES=4 python note_pruning.py --dataset starcoder_commentinstr_cleaned --sort_by logit_margin --model_name codellama-7b+lora:r=256:a=256 --encode_fn_type sft --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/codellama-7b+lora:r=256:a=256/starcoder_commentinstr_cleaned

CUDA_VISIBLE_DEVICES=4 python note_pruning.py --dataset starcoder_commentinstr_cleaned --sort_by el2n_agg=mean --model_name codellama-7b+lora:r=256:a=256 --encode_fn_type sft --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/codellama-7b+lora:r=256:a=256/starcoder_commentinstr_cleaned

CUDA_VISIBLE_DEVICES=4 python note_pruning.py --dataset starcoder_commentinstr_cleaned --sort_by grad_loraB_l2n --model_name codellama-7b+lora:r=256:a=256 --encode_fn_type sft --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/codellama-7b+lora:r=256:a=256/starcoder_commentinstr_cleaned

CUDA_VISIBLE_DEVICES=4 python note_pruning.py --dataset starcoder_commentinstr_cleaned --sort_by ifd --model_name codellama-7b+lora:r=256:a=256 --encode_fn_type sft --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/codellama-7b+lora:r=256:a=256/starcoder_commentinstr_cleaned

CUDA_VISIBLE_DEVICES=4 python note_pruning.py --dataset starcoder_commentinstr_cleaned --sort_by log_pmi --model_name codellama-7b+lora:r=256:a=256 --encode_fn_type sft --save_dir /gpfs/u/home/PTFM/PTFMqngp/scratch/github/mitibm2023/external/open-instruct/scripts/data_inds/codellama-7b+lora:r=256:a=256/starcoder_commentinstr_cleaned