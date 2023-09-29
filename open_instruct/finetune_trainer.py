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
    HfArgumentParser,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
    GPTNeoXTokenizerFast,
    GPT2Tokenizer,
    GPT2TokenizerFast, 
    OPTForCausalLM,
)
from peft import LoraConfig, TaskType, get_peft_model
from transformers.trainer_utils import get_last_checkpoint
from transformers import Trainer

##### wpq: avoid import error
# from safe_save_trainer import SafeSaveTrainer
import os
from pathlib import Path
from packaging import version
from transformers import Trainer, is_torch_tpu_available
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_sagemaker_mp_enabled, WEIGHTS_NAME
from transformers.utils import logging as hf_logging
from transformers.trainer_utils import ShardedDDPOption
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
from typing import Optional

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


# ### wpq: import for fix resuming PeftModel checkpoints in Trainer
# # https://github.com/huggingface/transformers/pull/24274/files
# from transformers.trainer import (
#     CONFIG_NAME, 
#     ADAPTER_WEIGHTS_NAME, 
#     ADAPTER_SAFE_WEIGHTS_NAME, 
#     WEIGHTS_INDEX_NAME,
#     SAFE_WEIGHTS_NAME,
#     SAFE_WEIGHTS_INDEX_NAME)
# from transformers import PretrainedConfig
# from transformers import __version__
# from transformers.utils import is_safetensors_available, is_peft_available
# if is_safetensors_available():
#     import safetensors.torch
# if is_peft_available():
#     from peft import PeftModel
# from transformers.modeling_utils import load_sharded_checkpoint
# ###

logger = hf_logging.get_logger(__name__)

class SafeSaveTrainer(Trainer):

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Will save the model, so you can reload it using `from_pretrained()`.
        Will only save from the main process.
        """

        if output_dir is None:
            output_dir = self.args.output_dir

        if is_torch_tpu_available():
            self._save_tpu(output_dir)
        elif is_sagemaker_mp_enabled():
            # Calling the state_dict needs to be done on the wrapped model and on all processes.
            os.makedirs(output_dir, exist_ok=True)
            state_dict = self.model_wrapped.state_dict()
            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
            if IS_SAGEMAKER_MP_POST_1_10:
                # 'user_content.pt' indicates model state_dict saved with smp >= 1.10
                Path(os.path.join(output_dir, "user_content.pt")).touch()
        elif (
            ShardedDDPOption.ZERO_DP_2 in self.args.sharded_ddp
            or ShardedDDPOption.ZERO_DP_3 in self.args.sharded_ddp
            or self.fsdp is not None
        ):
            full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                state_dict = self.model.state_dict()

            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
        # wpq: fix `SafeSaveTrainer` has no `deepspeed` attribute
        elif hasattr(self, 'deepspeed') and self.deepspeed:
            # this takes care of everything as long as we aren't under zero3
            if self.args.should_save:
                self._save(output_dir)

            if is_deepspeed_zero3_enabled():
                # It's too complicated to try to override different places where the weights dump gets
                # saved, so since under zero3 the file is bogus, simply delete it. The user should
                # either user deepspeed checkpoint to resume or to recover full weights use
                # zero_to_fp32.py stored in the checkpoint.
                if self.args.should_save:
                    file = os.path.join(output_dir, WEIGHTS_NAME)
                    if os.path.isfile(file):
                        # logger.info(f"deepspeed zero3: removing {file}, see zero_to_fp32.py to recover weights")
                        os.remove(file)

                # now save the real model if stage3_gather_16bit_weights_on_model_save=True
                # if false it will not be saved.
                # This must be called on all ranks
                if not self.deepspeed.save_16bit_model(output_dir, WEIGHTS_NAME):
                    logger.warning(
                        "deepspeed.save_16bit_model didn't save the model, since"
                        " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                        " zero_to_fp32.py to recover weights"
                    )
                    self.deepspeed.save_checkpoint(output_dir)

        elif self.args.should_save:
            self._save(output_dir)

        # Push to the Hub when `save_model` is called by the user.
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")

#     # fix resuming peftmodel checkpoint in Trainer `https://github.com/huggingface/transformers/pull/24274`
#     # merged to main: Jun 20. latest v4.31.0 has the change
#     # uncomment if use <v.4.31.0
#     def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
#         if model is None:
#             model = self.model

#         config_file = os.path.join(resume_from_checkpoint, CONFIG_NAME)
#         adapter_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_WEIGHTS_NAME)
#         adapter_safe_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_SAFE_WEIGHTS_NAME)
#         weights_file = os.path.join(resume_from_checkpoint, WEIGHTS_NAME)
#         weights_index_file = os.path.join(resume_from_checkpoint, WEIGHTS_INDEX_NAME)
#         safe_weights_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_NAME)
#         safe_weights_index_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_INDEX_NAME)

#         if not any(
#             os.path.isfile(f)
#             for f in [
#                 weights_file,
#                 safe_weights_file,
#                 weights_index_file,
#                 safe_weights_index_file,
#                 adapter_weights_file,
#                 adapter_safe_weights_file,
#             ]
#         ):
#             raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

#         logger.info(f"Loading model from {resume_from_checkpoint}.")

#         if os.path.isfile(config_file):
#             config = PretrainedConfig.from_json_file(config_file)
#             checkpoint_version = config.transformers_version
#             if checkpoint_version is not None and checkpoint_version != __version__:
#                 logger.warning(
#                     f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
#                     f"Transformers but your current version is {__version__}. This is not recommended and could "
#                     "yield to errors or unwanted behaviors."
#                 )

#         if os.path.isfile(weights_file) or os.path.isfile(safe_weights_file):
#             # If the model is on the GPU, it still works!
#             if is_sagemaker_mp_enabled():
#                 if os.path.isfile(os.path.join(resume_from_checkpoint, "user_content.pt")):
#                     # If the 'user_content.pt' file exists, load with the new smp api.
#                     # Checkpoint must have been saved with the new smp api.
#                     smp.resume_from_checkpoint(
#                         path=resume_from_checkpoint, tag=WEIGHTS_NAME, partial=False, load_optimizer=False
#                     )
#                 else:
#                     # If the 'user_content.pt' file does NOT exist, load with the old smp api.
#                     # Checkpoint must have been saved with the old smp api.
#                     if hasattr(self.args, "fp16") and self.args.fp16 is True:
#                         logger.warning(
#                             "Enabling FP16 and loading from smp < 1.10 checkpoint together is not suppported."
#                         )
#                     state_dict = torch.load(weights_file, map_location="cpu")
#                     # Required for smp to not auto-translate state_dict from hf to smp (is already smp).
#                     state_dict["_smp_is_partial"] = False
#                     load_result = model.load_state_dict(state_dict, strict=True)
#                     # release memory
#                     del state_dict
#             elif self.is_fsdp_enabled:
#                 from accelerate.utils import load_fsdp_model
#                 load_fsdp_model(self.accelerator.state.fsdp_plugin, self.accelerator, model, resume_from_checkpoint)
#             else:
#                 # We load the model state dict on the CPU to avoid an OOM error.
#                 if self.args.save_safetensors and os.path.isfile(safe_weights_file):
#                     state_dict = safetensors.torch.load_file(safe_weights_file, device="cpu")
#                 else:
#                     state_dict = torch.load(weights_file, map_location="cpu")

#                 # workaround for FSDP bug https://github.com/pytorch/pytorch/issues/82963
#                 # which takes *args instead of **kwargs
#                 load_result = model.load_state_dict(state_dict, False)
#                 # release memory
#                 del state_dict
#                 self._issue_warnings_after_load(load_result)

#         # Load adapters following PR # 24096
#         elif is_peft_available() and isinstance(model, PeftModel):
#             # If train a model using PEFT & LoRA, assume that adapter have been saved properly.
#             if hasattr(model, "active_adapter") and hasattr(model, "load_adapter"):
#                 if os.path.exists(resume_from_checkpoint):
#                     model.load_adapter(resume_from_checkpoint, model.active_adapter)
#                 else:
#                     logger.warning(
#                         "The intermediate checkpoints of PEFT may not be saved correctly, "
#                         f"consider using a custom callback to save {ADAPTER_WEIGHTS_NAME} in corresponding saving folders. "
#                         "Check some examples here: https://github.com/huggingface/peft/issues/96"
#                     )
#             else:
#                 logger.warning("Could not load adapter model, make sure to have `peft>=0.3.0` installed")
#         else:
#             # We load the sharded checkpoint
#             load_result = load_sharded_checkpoint(
#                 model, resume_from_checkpoint, strict=is_sagemaker_mp_enabled(), prefer_safe=self.args.save_safetensors
#             )
#             if not is_sagemaker_mp_enabled():
#                 self._issue_warnings_after_load(load_result)
# #####



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
    example_text = example_text + tokenizer.eos_token
    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    tokenized_prompt = tokenizer(example['prompt'], return_tensors='pt', max_length=max_seq_length, truncation=True)
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
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
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

# wpq: decorate for debugging distributed runs.
# @record
def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # wpq: convert str to dict
    if data_args.subsample_mixture is not None:
        data_args.subsample_mixture = re.compile('(?<!\\\\)\'').sub('\"', data_args.subsample_mixture)
        data_args.subsample_mixture = json.loads(data_args.subsample_mixture)
        print('subsample mixture:')
        print(data_args.subsample_mixture)

    if data_args.subsample_mixture is not None and data_args.subsample_inds_file is not None:
        raise ValueError('Either use mixture proportion or exact subset indices, but not both.')
    if data_args.subsample_inds_file is not None:
        if 'flan_v2' not in data_args.train_file:
            raise ValueError('subset indices only support flan_v2 for now.')

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
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
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
            use_auth_token=True if model_args.use_auth_token else None,
            streaming=data_args.streaming,
        )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            **dataset_args,
        )

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "trust_remote_code": True if 'mpt' in model_args.model_name_or_path else False,
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
        "use_auth_token": True if model_args.use_auth_token else None,
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
            use_auth_token=True if model_args.use_auth_token else None,
            torch_dtype=torch_dtype,
            # wpq: 8bit training
            # load_in_8bit=model_args.load_in_8bit,
            trust_remote_code=bool('mpt' in model_args.model_name_or_path),
        )
    else:
        logger.warning("No pretrained model_name_or_path is given. Training new model from scratch.")
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # wpq: `use_cache=True` is incompatible with gradient checkpointing
    model.config.use_cache = True if not training_args.gradient_checkpointing else False

    # no default pad token for llama!
    # here we add all special tokens again, because the default ones are not in the special_tokens_map 
    if isinstance(tokenizer, (LlamaTokenizer, LlamaTokenizerFast)):
        num_added_tokens = tokenizer.add_special_tokens({
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
        })
        assert num_added_tokens in [0, 1], "LlamaTokenizer/ LlamaTokenizerFast should only add one special token - the pad_token, or no tokens if pad token present."
    elif isinstance(tokenizer, GPTNeoXTokenizerFast):
        num_added_tokens = tokenizer.add_special_tokens({
            "pad_token": "<pad>",
        })
        assert num_added_tokens == 1, "GPTNeoXTokenizer should only add one special token - the pad_token."
    elif isinstance(tokenizer, (GPT2Tokenizer, GPT2TokenizerFast)) and isinstance(model, OPTForCausalLM):
        num_added_tokens = tokenizer.add_special_tokens({'unk_token': '<unk>'})
    ## wpq: add support for gpt2 tokenizer.
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
            logger.info('Subsample dataset according to indices: {data_args.subsample_inds_file}')
            with open(data_args.subsample_inds_file, 'rb') as f:
                inds = pickle.load(f)['inds']
            logger.info(f'Using subsample_inds_file: {data_args.subsample_inds_file}')
            logger.info(f'subsample_inds_file has {len(inds)} indices.')
            train_dataset = train_dataset.select(inds)
        ## 
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    # initalize a trainer
    # here we use a custom trainer that moves the model to CPU when saving the checkpoint in FSDP mode
    # we can switch to the default trainer after moving to deepspeed (let's don't change too much for now)
    # trainer = Trainer(
    trainer = SafeSaveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
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