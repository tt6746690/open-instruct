import os
import uuid


# from fastchat.model.model_adapter
OPENAI_MODEL_LIST = (
    "gpt-3.5-turbo", "gpt-3.5-turbo-0301", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106",
    "gpt-4", "gpt-4-0314", "gpt-4-0613", "gpt-4-turbo",)


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
    """Returns batch_size, job_duration. """
    from llm.submit import get_host_info
    arch = get_host_info()['arch']
    model_name_or_path = model_name_or_path.lower()

    models_7b = ['llama', 'mistral', 'zephyr']

    if arch == 'x86_64': # ccc use vllm
        if any(x in model_name_or_path for x in models_7b):
            batch_size, job_duration = 10, 1
            if any(x in task_name for x in ['alpacafarm']):
                job_duration = 6
    else:
        if any(x in model_name_or_path for x in models_7b):
            if any(x in task_name for x in ['bbh_s=3', 'mmlu_s=5', 'tydiqa_s=1_gp', 'alpacafarm']):
                batch_size, job_duration = 5, 1
            else:
                batch_size, job_duration = 10, 1
    return batch_size, job_duration
