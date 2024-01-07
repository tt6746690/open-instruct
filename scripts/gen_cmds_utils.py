import os
import uuid


def remove_all_symlinks(directory, verbose=False):
    for root, dirs, files in os.walk(directory):
        for name in files + dirs:
            path = os.path.join(root, name)
            if os.path.islink(path):
                os.unlink(path)
                if verbose:
                    print(f"Removed symlink: {path}")


def create_unique_symlinks(file_paths, verbose=False):
    """Create symlinks for each `file` in `files` in the same directory, with a unique name. """
    dirs = [os.path.dirname(x) for x in file_paths]

    symlink_path_dict = {}
    for directory, path in zip(dirs, file_paths):
        if os.path.isdir(path):
            symlink_name = f"symlink_{str(uuid.uuid4())[:8]}"  # Generate a unique symlink name
            symlink_path = os.path.join(directory, symlink_name)
            try:
                os.symlink(os.path.abspath(path), symlink_path)
                if verbose:
                    print(f"Created symlink: {symlink_path} -> {path}")
            except OSError as e:
                print(f"Failed to create symlink: {path}. Error: {e}")
            symlink_path_dict.update({path: symlink_path})
    return symlink_path_dict


def get_chat_formatting_function(model_name_or_path):
    if any(model_name_or_path.endswith(x) for x in [
        'NousResearch/Llama-2-7b-chat-hf',
        'codellama/CodeLlama-7b-Instruct-hf',
    ]):
        chat_formatting_function = 'eval.templates.create_prompt_with_llama2_chat_format'
    elif any(model_name_or_path.endswith(x) for x in [
        'HuggingFaceH4/mistral-7b-sft-alpha',
        'HuggingFaceH4/mistral-7b-sft-beta',
        'HuggingFaceH4/zephyr-7b-alpha',
        'HuggingFaceH4/zephyr-7b-beta',
    ]):
        chat_formatting_function = 'eval.templates.create_prompt_with_zephyr_chat_format'
    elif any(model_name_or_path.endswith(x) for x in [
        'mistralai/Mistral-7B-Instruct-v0.1'
    ]):
        chat_formatting_function = 'eval.templates.create_prompt_with_mistral_instruct_chat_format'
    else:
        chat_formatting_function = 'eval.templates.create_prompt_with_tulu_chat_format'
    return chat_formatting_function


def get_resource_for_task(task_name, model_name_or_path):
    model_name_or_path = model_name_or_path.lower()
    if any(x in model_name_or_path for x in ['gpt2-medium', 'pythia-160m']):
        return 50, 1
    if any(x in model_name_or_path for x in ['gpt-xl']):
        if any(x in task_name for x in ['bbh_s=3', 'mmlu_s=5', 'tydiqa_s=1_gp']):
            return 16, 1
        else:
            return 32, 1
    if any(x in model_name_or_path for x in ['llama', 'mistral', 'zephyr', 'pythia-1.4b', 'pythia-2.8b']):
        if any(x in task_name for x in ['bbh_s=3', 'mmlu_s=5', 'tydiqa_s=1_gp', 'alpacafarm']):
            return 5, 1
        else:
            return 10, 1
    if any(x in model_name_or_path for x in ['pythia-6.9b', 'dolly-v2-7b']):
        if any(x in task_name for x in ['bbh_s=3', 'mmlu_s=5', 'mmlu_s=0', 'tydiqa_s=1_gp', 'alpacafarm']):
            return 4, 1
        else:
            return 10, 1
    return 10, 1
