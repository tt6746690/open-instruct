#!/usr/bin/env python
# coding=utf-8
"""
This file is modified from the huggingface example for finetuning language models
[run_clm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py)
"""
import os
os.environ['TORCHELASTIC_ERROR_FILE'] = '/gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/scripts/error_file'

## wpq: https cannot download huggingface's model related files
#  - https://github.com/huggingface/transformers/issues/17611
#  - downgrade  requests==2.29.0 to 2.27.1 (sideeffect: charset-normalizer==2.0.12)
os.environ['CURL_CA_BUNDLE'] = ''
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
##
import pickle
from collections import Counter
import logging
import os
import sys
import json
import re
import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional, List
from functools import partial

import pyarrow
import datasets
import torch
import numpy as np
from datasets import load_dataset
from torch.distributed.elastic.multiprocessing.errors import record

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    CodeLlamaTokenizerFast,
    HfArgumentParser,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
    GPTNeoXTokenizerFast,
    GPT2Tokenizer,
    GPT2TokenizerFast, 
    OPTForCausalLM,
    Trainer,
)
from peft import LoraConfig, TaskType, get_peft_model
from transformers.trainer_utils import get_last_checkpoint


# ##### wpq: sub-class `get_train_dataloader` to add option to not shuffle the dataset

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers.trainer_pt_utils import LengthGroupedSampler
from transformers.trainer_utils import has_length
from transformers.utils import is_datasets_available
from transformers.optimization import get_scheduler

class MyTrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )
        else:
            # wpq: add option to change sampler to allow for non-shuffled ordering of data
            if self.args.dataloader_sampler == 'RandomSampler':
                return RandomSampler(self.train_dataset)
            elif self.args.dataloader_sampler == 'SequentialSampler':
                return SequentialSampler(self.train_dataset)
            else:
                raise ValueError(f'{self.args.dataloader_sampler} not supported.')
            
    ## wpq: for debugging llama2-7b train/loss issue.
    # def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
    #     """
    #     Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
    #     passed as an argument.

    #     Args:
    #         num_training_steps (int): The number of training steps to do.
    #     """
    #     if self.lr_scheduler is None:
    #         self.lr_scheduler = get_scheduler(
    #             self.args.lr_scheduler_type,
    #             optimizer=self.optimizer if optimizer is None else optimizer,
    #             num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
    #             num_training_steps=num_training_steps,
    #         )
    #         self._created_lr_scheduler = True

    #         print(f'create_scheduler: num_training_steps={num_training_steps}, optimizer={optimizer}')
    #         print(f'create_scheduler: lr_scheduler={self.lr_scheduler}')
    #         d = {
    #             'lr_scheduler_type': self.args.lr_scheduler_type,
    #             'optimizer': self.optimizer if optimizer is None else optimizer,
    #             'num_warmup_steps': self.args.get_warmup_steps(num_training_steps),
    #             'num_training_steps': num_training_steps,
    #         }
    #         print(f'create_scheduler kwargs = {d}')

    #         s = self.lr_scheduler
    #         lr_lambda = s.lr_lambdas[0]
    #         print(f'create_scheduler: lr_lambda={lr_lambda}')

    #         xs = list(range(num_training_steps))
    #         ys = [lr_lambda(x) for x in xs]
    #         print(f'create_scheduler xs:\n', xs)
    #         print(f'create_scheduler ys:\n', ys)


    #     return self.lr_scheduler
        
# wpq: having `open_instruct` not found error.
# from open_instruct.finetune import encode_with_prompt_completion_format, encode_with_messages_format


def encode_with_prompt_completion_format(example, tokenizer, max_seq_length):
    '''
    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated 
    and it doesn't make sense to follow directly with the completion.
    '''
    # if prompt doesn't end with space and completion doesn't start with space, add space
    if not example['prompt'].endswith((' ', '\n', '\t')) and not example['completion'].startswith((' ', '\n', '\t')):
        example_text = example['prompt'] + ' ' + example['completion']
    else:
        example_text = example['prompt'] + example['completion']
    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    ## wpq: `rstrip()` due to the space between prompt & completion, is a stand-alone token when tokenizing prompt, but not prompt+completion.
    tokenized_prompt = tokenizer(example['prompt'].rstrip(' '), return_tensors='pt', max_length=max_seq_length, truncation=True)
    # mask the prompt part for avoiding loss
    labels[:, :tokenized_prompt.input_ids.shape[1]] = -100
    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }


def encode_with_messages_format(example, tokenizer, max_seq_length):
    '''
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    '''
    messages = example['messages']
    if len(messages) == 0:
        raise ValueError('messages field is empty.')
    
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


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether to use flash attention in the model training"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in Transformers v4.34. Please use `token`."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default="auto",
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    use_lora: bool = field(
        default=False,
    )
    lora_rank: int = field(
        default=64,
    )
    lora_alpha: float = field(
        default=16,
    )
    lora_dropout: float = field(
        default=0.1,
    )
    load_in_8bit: bool = field(
        default=False,
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a json/jsonl file)."})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": ("The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,")
        },
    )
    subsample_mixture: Optional[str] = field(
        default=None,
    )
    subsample_inds_file: Optional[str] = field(
        default=None,
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None:
            raise ValueError("Need either a dataset name or a training file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["json", "jsonl"], "`train_file` should be a json or a jsonl file."


# see https://github.com/huggingface/transformers/blob/main/src/transformers/training_args_seq2seq.py for more details
from transformers import GenerationConfig

@dataclass
class TrainingArguments(transformers.training_args.TrainingArguments):

    dataloader_sampler: Optional[str] = field(
        default='RandomSampler',
        metadata={"help": "Whether to shuffle the dataset."},
    )

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values and `GenerationConfig` by dictionaries (for JSON
        serialization support). It obfuscates the token values by removing their value.
        """
        # filter out fields that are defined as field(init=False)
        d = super().to_dict()
        for k, v in d.items():
            if isinstance(v, GenerationConfig):
                d[k] = v.to_dict()
        return d



# wpq: decorate for debugging distributed runs.
# @record
def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir and os.path.isfile(os.path.join(training_args.output_dir, 'pytorch_model.bin')):
        print('Finished training already. Exit now.')
        return

    if model_args.use_auth_token is not None:
        warnings.warn("The `use_auth_token` argument is deprecated and will be removed in Transformers v4.34.", FutureWarning)
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # wpq: convert str to dict
    if data_args.subsample_mixture is not None:
        data_args.subsample_mixture = re.compile('(?<!\\\\)\'').sub('\"', data_args.subsample_mixture)
        data_args.subsample_mixture = json.loads(data_args.subsample_mixture)
        print('subsample mixture:')
        print(data_args.subsample_mixture)

    if data_args.subsample_mixture is not None and data_args.subsample_inds_file is not None:
        raise ValueError('Either use mixture proportion or exact subset indices, but not both.')
    # if data_args.subsample_inds_file is not None:
    #     if 'flan_v2' not in data_args.train_file:
    #         raise ValueError('subset indices only support flan_v2 for now.')

    # wpq: save args to a json file
    with training_args.main_process_first(local=False, desc=f"Saving args to `{training_args.output_dir+'.args.json'}`"):
        args_dict = {
            'model_args': asdict(model_args),
            'data_args': asdict(data_args),
            'training_args': asdict(training_args),
        }
        args_dict_path = training_args.output_dir+'.args.json'
        with open(args_dict_path, 'w') as f:
            json.dump(args_dict, f, indent=4)
        print(f'Saving args dict to {args_dict_path}')

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            # wpq: since I do write `.json` file to output_dir, raise if there are more files.
            if len(os.listdir(training_args.output_dir))>1:
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            streaming=data_args.streaming,
            download_mode=datasets.GenerateMode.FORCE_REDOWNLOAD if data_args.overwrite_cache else datasets.GenerateMode.REUSE_DATASET_IF_EXISTS,
        )
    else: 
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if 'ultrachat' in data_args.train_file:
            data_files['test'] = (
                '/gpfs/u/scratch/PTFM/PTFMqngp/github/mitibm2023/external/open-instruct/'
                'data/processed/ultrachat/ultrachat200k_test_data.jsonl')
        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            **dataset_args,
        )
        if 'ultrachat' in data_args.train_file:
            raw_datasets['test'] = raw_datasets['test'].select(range(1000))


    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
        # wpq: add support for mpt models.
        # flash attention only works on a100 gpus! so just disable.
        # if 'mpt' in model_args.model_name_or_path:
        #     config.attn_config['attn_impl'] = 'triton'
        #     config.init_device = 'cuda' # For fast initialization directly on GPU!
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this finetuning script."
        )

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this finetuning script."
        )

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
            use_flash_attention_2=True if model_args.use_flash_attn else False,
        )
    else:
        logger.warning("No pretrained model_name_or_path is given. Training new model from scratch.")
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=model_args.trust_remote_code)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # wpq: `use_cache=True` is incompatible with gradient checkpointing
    model.config.use_cache = True if not training_args.gradient_checkpointing else False

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
        ## wpq: for `huggyllama`/`NousResearch/Llama-2-7b-hf`, `LlamaTokenizerFast` tokenizer config not properly implemented and cannot tokenize special tokens like eos_token corretly. Need the following workaround. More details: https://github.com/huggingface/transformers/issues/23833
        if isinstance(tokenizer, LlamaTokenizerFast):
            if os.path.isdir(model_args.model_name_or_path):
                tmp_tok_path = os.path.join(
                    os.path.dirname(model_args.model_name_or_path),
                    os.path.basename(model_args.model_name_or_path)+'_fixtok')
                if not os.path.isdir(tmp_tok_path):
                    raise ValueError(f'Not valid fixtok path: {tmp_tok_path}')
            else:
                from secrets import token_hex
                tmp_tok_path = f'/tmp/wpq_tok_{token_hex(16)}'
                tokenizer.save_pretrained(tmp_tok_path)
            tokenizer = AutoTokenizer.from_pretrained(tmp_tok_path, **tokenizer_kwargs)
        for s, s_tokenized in [
            ("Hi<s>Hey</s>sir<unk>what<pad><pad>", 
            ['▁Hi', '<s>', '▁Hey', '</s>', '▁sir', '<unk>', '▁what', '<pad>', '<pad>']),
        ]:
            assert(tokenizer.tokenize(s, add_special_tokens=False)==s_tokenized)
    elif isinstance(tokenizer, GPTNeoXTokenizerFast):
        num_added_tokens = tokenizer.add_special_tokens({
            "pad_token": "<pad>",
        })
        assert num_added_tokens == 1, "GPTNeoXTokenizer should only add one special token - the pad_token."
    elif isinstance(tokenizer, (GPT2Tokenizer, GPT2TokenizerFast)) and isinstance(model, OPTForCausalLM):
        num_added_tokens = tokenizer.add_special_tokens({'unk_token': '<unk>'})
    elif isinstance(tokenizer, (GPT2Tokenizer, GPT2TokenizerFast)):
        num_added_tokens = tokenizer.add_special_tokens({
            "pad_token": "<pad>",
        })
        assert num_added_tokens == 1, "GPT2Tokenizer should only add one special token - the pad_token."

    # resize embeddings if needed (e.g. for LlamaTokenizer)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # wpq: use int8 training
    # wpq: put this after resize embedding!
    if model_args.load_in_8bit:
        from peft import prepare_model_for_int8_training
        model = prepare_model_for_int8_training(model)

    # wpq: add peft to finetune_trainer.py
    if model_args.use_lora:
        logger.info("Initializing LORA model...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            bias='none',
            r=model_args.lora_rank, 
            lora_alpha=model_args.lora_alpha, 
            lora_dropout=model_args.lora_dropout,
            target_modules=['q_proj','k_proj','v_proj','o_proj'],
        )
        # wpq: the following fixes `element 0 of tensors does not require grad and does not have a grad_fn` 
        # https://github.com/huggingface/peft/issues/137
        # https://github.com/huggingface/peft/issues/522
        if hasattr(training_args, 'gradient_checkpointing'):
            if training_args.gradient_checkpointing:
                model.enable_input_require_grads()

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Preprocessing the datasets.
    if "prompt" in raw_datasets["train"].column_names and "completion" in raw_datasets["train"].column_names:
        encode_function = partial(
            encode_with_prompt_completion_format,
            tokenizer=tokenizer,
            max_seq_length=data_args.max_seq_length,
        )
    elif "messages" in raw_datasets["train"].column_names:
        encode_function = partial(
            encode_with_messages_format,
            tokenizer=tokenizer,
            max_seq_length=data_args.max_seq_length,
        )
    else:
        raise ValueError("You need to have either 'prompt'&'completion' or 'messages' in your column names.")
    

    # To speed up this part, we use multiprocessing.
    # wpq: set `local=False` in multi-node environment so that the main process
    # of the first node will do the processing, given the filesystem is shared by nodes.
    with training_args.main_process_first(local=False, desc="Processing instruction data"):
        if not data_args.streaming:
            lm_datasets = raw_datasets.map(
                encode_function,
                batched=False,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Tokenizing and reformatting instruction data",
            )
        else:
            lm_datasets = raw_datasets.map(
                encode_function,
                batched=False,
            )
        lm_datasets.set_format(type="pt")
        # ## retain data points with at least one label not equal to -100
        # # however, this messes up data ordering. for now just comment this out.
        # lm_datasets = lm_datasets.filter(lambda example: (example['labels'] != -100).any())

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        ## wpq: subsample `dataset` according to `data_args.subsample_mixture`.
        # assumes dataset in `train_file` ordered, 
        # e.g., ['cot', 'cot', 'flan_v2', 'flan_v2', ...]
        # note `counts.items()` is ordered as well!
        if data_args.subsample_mixture is not None:
            logger.info('Subsample dataset according to mixture proportions: {data_args.subsample_mixture}')
            counts = Counter(train_dataset['dataset'])
            inds = []
            cum = 0
            for k, N in counts.items():
                n = int(data_args.subsample_mixture.get(k, 0))
                replace = True if n>N else False
                replace = True # always sample with replacement, for fairer comparison.
                inds += list(np.random.choice(N, size=n, replace=replace) + cum)
                cum += N
            train_dataset = train_dataset.select(inds)

        if data_args.subsample_inds_file is not None:
            logger.info(f'Subsample dataset according to indices: {data_args.subsample_inds_file}')
            with open(data_args.subsample_inds_file, 'rb') as f:
                inds = pickle.load(f)['inds']
            logger.info(f'subsample_inds_file has {len(inds)} indices.')
            train_dataset = train_dataset.select(inds)
        ## 
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))


    logger.info(f"model.dtype = {model.dtype}")

    # initalize a trainer
    # here we use a custom trainer that moves the model to CPU when saving the checkpoint in FSDP mode
    # we can switch to the default trainer after moving to deepspeed (let's don't change too much for now)
    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=lm_datasets['test'] if (training_args.do_eval and 'test' in lm_datasets) else None,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    with training_args.main_process_first(local=False, desc=f"Moving args to `{training_args.output_dir}`"):
        args_dict_path = training_args.output_dir+'.args.json'
        if os.path.isfile(args_dict_path) and os.path.isdir(training_args.output_dir):
            os.rename(args_dict_path, os.path.join(training_args.output_dir, 'ft_args.json'))


if __name__ == "__main__":
    main()