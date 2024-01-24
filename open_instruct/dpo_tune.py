#!/usr/bin/env python
# coding=utf-8
'''
DPO tuning script. Adapted from our finetuning script.
'''

import glob
import argparse
import logging
import math
import os
import json
import pickle
import random
from collections import defaultdict
from copy import deepcopy
import pyarrow # add before datasets/torch
import datasets
import torch
from functools import partial
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm.auto import tqdm
import deepspeed
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    CodeLlamaTokenizerFast,
    SchedulerType,
    get_scheduler,
    GPTNeoXTokenizerFast,
    GPT2Tokenizer,
    OPTForCausalLM,
    BitsAndBytesConfig,
)
from transformers.utils import WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_INDEX_NAME
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from dpo_utils import dpo_loss, concatenated_forward, DataCollatorForSeq2SeqDPO

logger = get_logger(__name__)


def parse_args(cmd=None):
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="If passed, will use LORA (low-rank parameter-efficient training) to train the model.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="The rank of lora.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=16,
        help="The alpha parameter of lora.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="The dropout rate of lora modules.",
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="If passed, will use flash attention to train the model.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="The maximum total sequence length (prompt+completion) of each training example.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0, help="Ratio of total training steps used for warmup."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=None,
        help="Log the training loss and learning rate every logging_steps steps.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help=(
            "Turn on gradient checkpointing. Saves memory but slows training."
        ),
    )
    parser.add_argument(
        "--use_qlora",
        action="store_true",
        help=(
            "Use qLoRA training - main thing is initialising model in quantised form. Not compatible with deepspeed."
        ),
    )
    parser.add_argument(
        '--clip_grad_norm',
        type=float,
        default=-1,
        help='Clip gradient norm. Not compatible with deepspeed (use deepspeed config instead).',
    )
    parser.add_argument(
        '--use_8bit_optimizer',
        action='store_true',
        help='Use 8bit optimizer from bitsandbytes. Not compatible with deepspeed (use deepspeed config instead).',
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=0.1,
        help='Beta parameter for DPO loss. Default is 0.1.',
    )
    parser.add_argument(
        '--use_paged_optimizer',
        action='store_true',
        help='Use paged optimizer from bitsandbytes. Not compatible with deepspeed (use deepspeed config instead).',
    )
    parser.add_argument(
        '--subsample_inds_file',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--dataloader_sampler',
        type=str,
        choices=['RandomSampler', 'SequentialSampler'],
        default='RandomSampler',
    )

    if cmd is not None:
        from rosemary import jpt_parse_args
        args = jpt_parse_args(parser, cmd)
    else:
        args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None:
        raise ValueError("Need either a dataset name or a training file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["json", "jsonl"], "`train_file` should be a json/jsonl file."
    return args

def encode_with_messages_format(example, tokenizer, max_seq_length):
    '''
    Here we assume each example has a rejected and chosen field, both of which are a list of messages.
    Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    We assume only the last message is different, and the prompt is contained in the list of messages.
    '''
    chosen_messages = example['chosen']
    rejected_messages = example['rejected']
    if len(chosen_messages) == 0:
        raise ValueError('chosen messages field is empty.')
    if len(rejected_messages) == 0:
        raise ValueError('rejected messages field is empty.')
    
    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text
        
    def encode_messages(messages):
        example_text = _concat_messages(messages).strip()
        tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
        input_ids = tokenized_example.input_ids
        labels = input_ids.clone()

        # mask the non-assistant part for avoiding loss
        for message_idx, message in enumerate(messages):
            if message["role"] != "assistant":
                if message_idx == 0:
                    message_start_idx = 0
                else:
                    message_start_idx = tokenizer(
                        _concat_messages(messages[:message_idx]), return_tensors='pt', max_length=max_seq_length, truncation=True
                    ).input_ids.shape[1]
                if message_idx < len(messages) - 1 and messages[message_idx+1]["role"] == "assistant":
                    # here we also ignore the role of the assistant
                    messages_so_far = _concat_messages(messages[:message_idx+1]) + "<|assistant|>\n"
                else:
                    messages_so_far = _concat_messages(messages[:message_idx+1])
                message_end_idx = tokenizer(
                    messages_so_far,
                    return_tensors='pt', 
                    max_length=max_seq_length, 
                    truncation=True
                ).input_ids.shape[1]
                labels[:, message_start_idx:message_end_idx] = -100
                
                if message_end_idx >= max_seq_length:
                    break

        attention_mask = torch.ones_like(input_ids)
        return {
            'input_ids': input_ids.flatten(),
            'labels': labels.flatten(),
            'attention_mask': attention_mask.flatten(),
        }
    chosen_encoded = encode_messages(chosen_messages)
    rejected_encoded = encode_messages(rejected_messages)
    # labels are useful for working out where the loss is valid.
    return {
        'chosen_input_ids': chosen_encoded['input_ids'],
        'chosen_labels': chosen_encoded['labels'],
        'chosen_attention_mask': chosen_encoded['attention_mask'],
        'rejected_input_ids': rejected_encoded['input_ids'],
        'rejected_labels': rejected_encoded['labels'],
        'rejected_attention_mask': rejected_encoded['attention_mask'],
    }



def remove_state_dict_file_if_model_is_sharded(folder):
    from transformers.utils import WEIGHTS_NAME, WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_INDEX_NAME
    from transformers.utils import is_safetensors_available

    # Load the index
    weights_file = os.path.join(folder, WEIGHTS_NAME)
    index_file = os.path.join(folder, WEIGHTS_INDEX_NAME)
    safe_index_file = os.path.join(folder, SAFE_WEIGHTS_INDEX_NAME)
    weights_present = os.path.isfile(weights_file)
    index_present = os.path.isfile(index_file)
    safe_index_present = os.path.isfile(safe_index_file)
    is_sharded = index_present or (safe_index_present and is_safetensors_available())
    if weights_present and is_sharded:
        print(f'both {WEIGHTS_NAME} and model shards exists under {folder}.\n'
              f'Remove {WEIGHTS_NAME}')
        os.remove(os.path.join(folder, WEIGHTS_NAME))



def save_with_accelerate(accelerator, model, tokenizer, output_dir, args):
    ## wpq: reference https://github.com/huggingface/accelerate/blob/v0.21.0/examples/by_feature/deepspeed_with_config_support.py#L707
    # 
    if output_dir is None:
        return

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
    unwrapped_model = accelerator.unwrap_model(model)
    # When doing multi-gpu training, we need to use accelerator.get_state_dict(model) to get the state_dict.
    # Otherwise, sometimes the model will be saved with only part of the parameters.
    # Also, accelerator needs to use the wrapped model to get the state_dict.
    state_dict = accelerator.get_state_dict(model)
    if args.use_lora:
        # When using lora, the unwrapped model is a PeftModel, which doesn't support the is_main_process 
        # and has its own save_pretrained function for only saving lora modules.
        # We have to manually specify the is_main_process outside the save_pretrained function.
        if accelerator.is_main_process:
            unwrapped_model.save_pretrained(output_dir, state_dict=state_dict)
    else:
        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=False # errors with safetensors...
        )
    # wpq: both `pytorch_model.bin` (without containing actual weights) and its sharded are present, rename `pytorch_model.bin` so that `from_pretrained` can load the model properly from model shards.
    if accelerator.is_main_process:
        remove_state_dict_file_if_model_is_sharded(output_dir)


# from trl, we have to prep the ref model separately.
def prepare_deepspeed(accelerator, model):
        deepspeed_plugin = accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model
        

def clean_checkpoints(accelerator, output_dir, keep_n=1):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        import shutil
        dirs = glob.glob(os.path.join(output_dir, 'step_*')) + \
            glob.glob(os.path.join(output_dir, 'epoch_*'))
        dirs = [x for x in dirs if os.path.isdir(x)]
        dirs.sort(key=os.path.getctime) # Sorts folders by date modified
        dirs_remove = dirs[:len(dirs)-keep_n] if keep_n < len(dirs) else []
        dirs_keep = dirs[len(dirs)-keep_n:] if keep_n < len(dirs) else dirs
        logger.info(f'Cleaning checkpoints. Keeping {dirs_keep} and removing {dirs_remove}')
        for d in dirs_remove:
            shutil.rmtree(d)
        return dirs_keep
    else:
        return None
    


def main():
    args = parse_args()


    if os.path.isdir(args.output_dir) and \
            (os.path.isfile(os.path.join(args.output_dir, 'pytorch_model.bin')) or os.path.isfile(os.path.join(args.output_dir, SAFE_WEIGHTS_INDEX_NAME)) or os.path.isfile(os.path.join(args.output_dir, WEIGHTS_INDEX_NAME))):
        print('Finished training already. Exit now.')
        return


    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    logger.info(f'[wpq] SLURM_PROCID: {os.environ.get("SLURM_PROCID", None)}')

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    accelerator.wait_for_everyone()

    ## wpq: save args
    if accelerator.is_main_process:
        args_dict_path = args.output_dir+'.args.json'
        with open(args_dict_path, 'w') as f:
            json.dump(vars(args), f, indent=4)
        logger.info(f'Saving args dict to {args_dict_path}')
    ## 

    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
        )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train_prefs"] = args.train_file
        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
            # download_mode=datasets.DownloadMode.FORCE_REDOWNLOAD if args.overwrite_cache else datasets.DownloadMode.REUSE_DATASET_IF_EXISTS,
            **dataset_args,
        )

    # Load pretrained model and tokenizer
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    def load_model():
        if args.model_name_or_path:
            if args.use_qlora:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                device_index = accelerator.local_process_index
                device_map = {"": device_index} # force data-parallel training.
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                    load_in_4bit=True,
                    quantization_config=bnb_config,
                    device_map=device_map,
                    torch_dtype=torch.bfloat16,
                    use_flash_attention_2=True if args.use_flash_attn else False,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                    low_cpu_mem_usage=args.low_cpu_mem_usage,
                    use_flash_attention_2=True if args.use_flash_attn else False,
                )
        else:
            logger.info("Training new model from scratch")
            model = AutoModelForCausalLM.from_config(config)
        return model

    model = load_model()
    if not args.use_lora:
        reference_model = load_model()
    else:
        reference_model = model


    # no default pad token for llama!
    # here we add all special tokens again, because the default ones are not in the special_tokens_map 
    if isinstance(tokenizer, (LlamaTokenizer, LlamaTokenizerFast, CodeLlamaTokenizerFast)):
        from transformers import AddedToken
        num_added_tokens = tokenizer.add_special_tokens({
            "bos_token": AddedToken("<s>", normalized=False, special=True),
            "eos_token": AddedToken("</s>", normalized=False, special=True),
            "unk_token": AddedToken("<unk>", normalized=False, special=True),
            "pad_token": AddedToken("<pad>", normalized=False, special=True),
        })
        ## wpq: for `huggyllama`/`NousResearch/Llama-2-7b-hf`, `LlamaTokenizerFast` tokenizer config not properly implemented and cannot tokenize special tokens like eos_token corretly. 
        # Need the following workaround. More details: https://github.com/huggingface/transformers/issues/23833
        def check_tokenizer_pad_properly(tokenizer):
            tokenizer_handles_pad_ok = True
            for s, s_tokenized in [
                ("Hi<s>Hey</s>sir<unk>what<pad><pad>", 
                ['▁Hi', '<s>', '▁Hey', '</s>', '▁sir', '<unk>', '▁what', '<pad>', '<pad>']),
            ]:
                if tokenizer.tokenize(s, add_special_tokens=False)!=s_tokenized:
                    tokenizer_handles_pad_ok = False
            return tokenizer_handles_pad_ok
        if not check_tokenizer_pad_properly(tokenizer):
            if os.path.isdir(args.model_name_or_path):
                tmp_tok_path = os.path.join(
                    os.path.dirname(args.model_name_or_path),
                    os.path.basename(args.model_name_or_path)+'_fixtok')
                if not os.path.isdir(tmp_tok_path):
                    tokenizer.save_pretrained(tmp_tok_path)
            else:
                from secrets import token_hex
                tmp_tok_path = f'/tmp/wpq_tok_{token_hex(16)}'
                tokenizer.save_pretrained(tmp_tok_path)
            tokenizer = AutoTokenizer.from_pretrained(tmp_tok_path, use_fast=not args.use_slow_tokenizer)
        assert(check_tokenizer_pad_properly(tokenizer))
    elif isinstance(tokenizer, GPTNeoXTokenizerFast):
        num_added_tokens = tokenizer.add_special_tokens({
            "pad_token": "<pad>",
        })
        assert num_added_tokens == 1, "GPTNeoXTokenizer should only add one special token - the pad_token."
    elif isinstance(tokenizer, GPT2Tokenizer) and isinstance(model, OPTForCausalLM):
        num_added_tokens = tokenizer.add_special_tokens({'unk_token': '<unk>'})

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
        if not args.use_lora:
            reference_model.resize_token_embeddings(len(tokenizer))

    if args.use_lora:
        if args.use_qlora:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

        logger.info("Initializing LORA model...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=args.lora_rank, 
            lora_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "o_proj", "v_proj", "k_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # wpq: `use_cache=True` is incompatible with gradient checkpointing
    model.config.use_cache = True if not args.gradient_checkpointing else False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Preprocessing the datasets.
    if "prompt" in raw_datasets["train_prefs"].column_names and "completion" in raw_datasets["train_prefs"].column_names:
        raise ValueError("Sorry, prompt-completion format is not supported for DPO training.")
    elif "chosen" in raw_datasets["train_prefs"].column_names and "rejected" in raw_datasets["train_prefs"].column_names:
        encode_function = partial(
            encode_with_messages_format,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
        )
    else:
        raise ValueError("You need to have 'chosen' and 'rejected in your column names.")
    
    with accelerator.main_process_first():
        lm_datasets = raw_datasets["train_prefs"].map(
            encode_function,
            batched=False,
            num_proc=args.preprocessing_num_workers,
            remove_columns=[name for name in raw_datasets["train_prefs"].column_names if name not in ["chosen_input_ids", "chosen_labels", "chosen_attention_mask", "rejected_input_ids", "rejected_labels", "rejected_attention_mask"]],
            load_from_cache_file=not args.overwrite_cache,
            desc="Tokenizing and reformatting instruction data",
        )
        lm_datasets.set_format(type="pt")
        # wpq: if enforce max_seq_length properly, below should do nothing, so just remove.
        # # our thresholding mighta meant some examples have no labels, remove.
        # lm_datasets = lm_datasets.filter(lambda example: (example['chosen_labels'] != -100).any(), num_proc=8)
        # lm_datasets = lm_datasets.filter(lambda example: (example['rejected_labels'] != -100).any(), num_proc=8)

    train_dataset = lm_datasets

    if args.subsample_inds_file is not None:
        logger.info(f'Subsample dataset according to indices: {args.subsample_inds_file}')
        with open(args.subsample_inds_file, 'rb') as f:
            inds = pickle.load(f)['inds']
        logger.info(f'subsample_inds_file has {len(inds)} indices.')
        train_dataset = train_dataset.select(inds)


    # wpq: add option to change sampler to allow for non-shuffled ordering of data
    if args.dataloader_sampler == 'RandomSampler':
        sampler = RandomSampler(train_dataset)
    elif args.dataloader_sampler == 'SequentialSampler':
        sampler = SequentialSampler(train_dataset)
    else:
        raise ValueError(f'{args.dataloader_sampler} not supported.')

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, 
        collate_fn=DataCollatorForSeq2SeqDPO(tokenizer=tokenizer, model=model, padding="longest"),
        batch_size=args.per_device_train_batch_size,
        sampler=sampler,
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    if args.use_qlora or args.use_paged_optimizer:
        from bitsandbytes.optim import AdamW
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            optim_bits=8 if args.use_8bit_optimizer else 32,
            is_paged=True
        )
    else:
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True


    # Create the learning rate scheduler.
    # Note: the current accelerator.step() calls the .step() of the real scheduler for the `num_processes` times. This is because they assume 
    # the user initialize the scheduler with the entire training set. In the case of data parallel training, each process only
    # sees a subset (1/num_processes) of the training set. So each time the process needs to update the lr multiple times so that the total 
    # number of updates in the end matches the num_training_steps here.
    # Here we need to set the num_training_steps to either using the entire training set (when epochs is specified) or we need to multiply the 
    # num_training_steps by num_processes so that the total number of updates matches the num_training_steps.
    num_training_steps_for_scheduler = args.max_train_steps if overrode_max_train_steps else args.max_train_steps * accelerator.num_processes
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
    )
    logger.info(f"[wpq]  num_update_steps_per_epoch = {num_update_steps_per_epoch}")
    logger.info(f"[wpq]  overrode_max_train_steps = {overrode_max_train_steps}")
    logger.info(f"[wpq]  args.max_train_steps = {args.max_train_steps}")
    logger.info(f"[wpq]  num_training_steps_for_scheduler = {num_training_steps_for_scheduler}")
    logger.info(f"[wpq]  accelerator.num_processes = {accelerator.num_processes}")

    # Prepare everything with `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    if not args.use_lora:
        reference_model = prepare_deepspeed(accelerator, reference_model)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("mitibm", experiment_config, init_kwargs={
                'wandb': {
                    'name': args.output_dir.replace('results/', ''),
                }
            })

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
 
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # wpq: instead of using `from_pretrained`, use accelerate's `load_state` to save model weights, optimizer state etc. reference: https://github.com/huggingface/accelerate/blob/v0.25.0/examples/by_feature/deepspeed_with_config_support.py

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        dirs = glob.glob(os.path.join(args.output_dir, 'step_*')) + \
            glob.glob(os.path.join(args.output_dir, 'epoch_*'))
        dirs = [x for x in dirs if os.path.isdir(x)]
        dirs.sort(key=os.path.getctime) # Sorts folders by date modified
        if len(dirs) == 0:
            args.resume_from_checkpoint = False
            logger.info(f"No checkpoint found in {args.output_dir}. Training from scratch.")
        else:
            # if previous training interrupted during `accelerator.save_state`, there will be two checkpoints, 
            # with the newer checkpoint being incomplete. We want to load the older checkpoint.
            if len(dirs) == 2:
                checkpoint_path = dirs[0]
            else:
                checkpoint_path = dirs[-1]
            accelerator.load_state(checkpoint_path)
            accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
            # Extract `epoch_{i}` or `step_{i}`
            training_difference = os.path.splitext(os.path.basename(checkpoint_path))[0]

            if "epoch" in training_difference:
                starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                resume_step = None
                completed_steps = starting_epoch * num_update_steps_per_epoch
            else:
                # need to multiply `gradient_accumulation_steps` to reflect real steps
                resume_step = (
                    int(training_difference.replace("step_", ""))
                    * args.gradient_accumulation_steps
                )
                starting_epoch = resume_step // len(train_dataloader)
                completed_steps = resume_step // args.gradient_accumulation_steps
                resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0
        if (
            args.resume_from_checkpoint
            and epoch == starting_epoch
            and resume_step is not None
        ):
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_step
            )
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            logps = defaultdict(int)
            # dpo forward pass & loss
            with accelerator.accumulate(model):
                policy_chosen_logps, policy_rejected_logps = concatenated_forward(model, batch)
                with torch.no_grad():
                    if args.use_lora:
                        with accelerator.unwrap_model(model).disable_adapter():
                            reference_chosen_logps, reference_rejected_logps = concatenated_forward(model, batch)
                    else:
                        reference_chosen_logps, reference_rejected_logps = concatenated_forward(reference_model, batch)
                losses, _, _ = dpo_loss(
                    policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta=args.beta)
                # TODO: metric logging          
                loss = losses.mean()
                # We keep track of the loss at each logged step
                total_loss += loss.detach().float()
                logps['logps_policy_chosen'] += policy_chosen_logps.detach().float()
                logps['logps_policy_rejected'] += policy_rejected_logps.detach().float()
                logps['logps_reference_chosen'] += reference_chosen_logps.detach().float()
                logps['logps_reference_rejected'] += reference_rejected_logps.detach().float()
                accelerator.backward(loss)
                # clip gradient norm. don't do this with deepspeed
                if accelerator.sync_gradients and args.clip_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()       

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    avg_loss = accelerator.gather(total_loss).mean().item() / args.gradient_accumulation_steps / args.logging_steps
                    for k in list(logps.keys()):
                        logps[k] = accelerator.gather(logps[k]).mean().item() / args.gradient_accumulation_steps
                    logger.info(f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}")
                    if args.with_tracking:
                        accelerator.log(
                            {
                                "learning_rate": lr_scheduler.get_last_lr()[0],
                                "train_loss": avg_loss,
                                "step": completed_steps,
                                "train/global_step": completed_steps,
                                'logps_policy_chosen': logps['logps_policy_chosen'],
                                'logps_policy_rejected': logps['logps_policy_rejected'],
                                'logps_reference_chosen': logps['logps_reference_chosen'],
                                'logps_reference_rejected': logps['logps_reference_rejected'],
                            },
                            step=completed_steps,
                        )
                    total_loss = 0
                    
                # We save the model, optimizer, lr_scheduler, and seed states by calling `save_state`
                # These are saved to folders named `step_{overall_step}`
                # Will contain files: "pytorch_model.bin", "optimizer.bin", "scheduler.bin", and "random_states.pkl"
                # If mixed precision was used, will also save a "scalar.bin" file
                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir)
                        clean_checkpoints(accelerator, args.output_dir, keep_n=1)

                if completed_steps >= args.max_train_steps:
                    break

        # We save the model, optimizer, lr_scheduler, and seed states by calling `save_state`
        # These are saved to folders named `epoch_{epoch}`
        # Will contain files: "pytorch_model.bin", "optimizer.bin", "scheduler.bin", and "random_states.pkl"
        # If mixed precision was used, will also save a "scalar.bin" file
        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
            clean_checkpoints(accelerator, args.output_dir, keep_n=1)

    if args.with_tracking:
        accelerator.end_training()

    save_with_accelerate(accelerator, model, tokenizer, args.output_dir, args)
    clean_checkpoints(accelerator, args.output_dir, keep_n=1)

    ## wpq: save args
    if accelerator.is_main_process:
        args_dict_path = args.output_dir+'.args.json'
        if os.path.isfile(args_dict_path) and os.path.isdir(args.output_dir):
            os.rename(args_dict_path, os.path.join(args.output_dir, 'ft_args.json'))
    ##


if __name__ == "__main__":
    main()
