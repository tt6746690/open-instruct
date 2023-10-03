import os
import argparse
import random
import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flan_v2_data_dir", type=str, default="data/flan_v2")
    parser.add_argument("--total_num_samples", type=int, default=100000)
    parser.add_argument("--output_path", type=str, default="data/flan_v2/flan_v2_100k.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    # # The following portions are based on the flan_v2 code: https://github.com/google-research/FLAN/blob/main/flan/v2/run_example.py
    # portions = {
    #     "flan_zsopt": 0.1,
    #     "flan_fsopt": 0.1,
    #     "flan_zsnoopt": 0.1,
    #     "flan_fsnoopt": 0.1,
    #     "t0_zsopt": 0.08,
    #     "t0_fsopt": 0.08,
    #     "t0_zsnoopt": 0.08,
    #     "t0_fsnoopt": 0.08,
    #     "niv2_zsopt": 0.1,
    #     "niv2_fsopt": 0.1,
    #     "cot_zsopt": 0.025,
    #     "cot_fsopt": 0.025,
    #     "dialog_zsopt": 0.015,
    #     "dialog_fsopt": 0.015,
    # }

    # for flan2022_submix:
    # ('flan2021_submix', 0.4),  # mixing weight = 40%
    #       ('flan_zsopt', 1),      # mixing weight = 25%
    #       ('flan_fsopt', 1),      # mixing weight = 25%
    #       ('flan_zsnoopt', 1),    # mixing weight = 25%
    #       ('flan_fsnoopt', 1),    # mixing weight = 25%
    # ('t0_submix', 0.32),       # mixing weight = 32%
    #       ('t0_zsopt', 1),      # mixing weight = 25%
    #       ('t0_fsopt', 1),      # mixing weight = 25%
    #       ('t0_zsnoopt', 1),    # mixing weight = 25%
    #       ('t0_fsnoopt', 1),    # mixing weight = 25%
    # ('niv2_submix', 0.2),      # mixing weight = 20%
    #       ('niv2_zsopt', 1),    # mixing weight = 50%
    #       ('niv2_fsopt', 1),    # mixing weight = 50%
    # ('cot_submix', 0.05),      # mixing weight = 5%
    #       ('cot_zsopt', 1),    # mixing weight = 50%
    #       ('cot_fsopt', 1),    # mixing weight = 50%
    # ('dialog_submix', 0.03),   # mixing weight = 3%
    #       ('dialog_zsopt', 1),    # mixing weight = 50%
    #       ('dialog_fsopt', 1),    # mixing weight = 50%

    ## SirNeural/flan_v2
    portions = {
        'flan_zs_opt_train':   .4 * .25,
        'flan_fs_opt_train':   .4 * .25,
        'flan_zs_noopt_train': .4 * .25,
        'flan_fs_noopt_train': .4 * .25,
        't0_zs_opt_train':     .32 * .25,
        't0_zs_noopt_train':   .32 * .25,
        't0_fs_noopt_train':   .32 * .5, # missing fs_opt files, just sample from fs_noopt instead
        'niv2_zs_opt_train':   .2 * .5,
        'niv2_fs_opt_train':   .2 * .5,
        'cot_zs_opt_train':    .05 * .5,
        'cot_fs_opt_train':    .05 * .5,
        'dialog_zs_opt_train': .03 * .5,
        'dialog_fs_opt_train': .03 * .5,
    }

    num_lines_dict = {
        "flan_zs_opt_train": 62118170,
        "flan_fs_opt_train": 123010706,
        "flan_zs_noopt_train": 62118170,
        "flan_fs_noopt_train": 61521286,
        "t0_zs_opt_train": 42881000,
        "t0_zs_noopt_train": 42881000,
        "t0_fs_noopt_train": 42545443,
        "niv2_zs_opt_train": 5031430,
        "niv2_fs_opt_train": 10031790,
        "cot_zs_opt_train": 74730,
        "cot_fs_opt_train": 149490,
        "dialog_zs_opt_train": 11274820,
        "dialog_fs_opt_train": 22528240
    }

    assert sum(portions.values()) == 1.0
    assert all(os.path.isfile(os.path.join(args.flan_v2_data_dir, f'{x}.jsonl')) for x in portions.keys())

    num_samples = {k: int(v * args.total_num_samples) for k, v in portions.items()}

    with open(args.output_path, "w") as fout:
        for task_name, num_sample in num_samples.items():
            print(f"Sampling {num_sample} samples from {task_name}")
            task_data_path = os.path.join(args.flan_v2_data_dir, f"{task_name}.jsonl")
            # randomly sample num_sample lines from task_data_path, the data might be very large so we can't load it all into memory
            # we need to first count the total number of lines in the file and then only load the lines we need
            if task_name not in num_lines_dict:
                num_lines = 0
                with open(task_data_path, "r") as fin:
                    for line in tqdm.tqdm(fin, desc=f"Counting lines in {task_data_path}"):
                        num_lines += 1                    
                num_lines_dict[task_name] = num_lines
            else:
                num_lines = num_lines_dict[task_name]
            print(f"Sampling {num_sample} lines from {num_lines} lines")
            sampled_lines = random.sample(range(num_lines), num_sample)
            sampled_lines = set(sampled_lines)
            with open(task_data_path, "r") as fin:
                for i, line in tqdm.tqdm(enumerate(fin), desc=f"Reading the file to save the sampled lines"):
                    if i in sampled_lines:
                        fout.write(line)

    print('Number of lines for jsonl file: ', num_lines_dict)