import argparse
import os
import re
import json
import tqdm
import glob
import torch
import random
import evaluate
from eval.utils import (
    load_hf_lm_and_tokenizer,
    generate_completions,
    query_openai_chat_model,
    dynamic_import_function,
)


exact_match = evaluate.load("exact_match")
    

def eval_hf_model(args, model, tokenizer, examples, task_prompt, save_path=None):
    targets = [example["target"] for example in examples]
    if save_path:
        fout = open(save_path, "w")

    prompts = []
    chat_formatting_function = dynamic_import_function(args.chat_formatting_function) if args.use_chat_format else None
    for example in examples:
        prompt = task_prompt.strip() + "\n\nQ: " + example["input"]
        if args.use_chat_format:
            messages = [{"role": "user", "content": prompt}]
            prompt = chat_formatting_function(messages, add_bos=False)
            if prompt[-1] in ["\n", " "]:
                prompt += "A:"
            else:
                prompt += " A:"
        else:
            prompt += "\nA:"
        prompts.append(prompt)

    if args.no_cot:
        # get the last token because the tokenizer may add space tokens at the start.
        # wpq: t5 tokenizer strips `\n`. don't use `\n` as stop sequence. just generate to max length or encounters <\s>.
        stop_id_sequences = tokenizer.encode("\n\n", add_special_tokens=False)
        stop_id_sequences = [stop_id_sequences[-2:]] if stop_id_sequences else None
    else:
        # let's not use the stop sequence for cot now since it's too inefficient when the generation is long. 
        # instead, we'll do some post-processing to extract the answer.
        stop_id_sequences = None

    from transformers import GPT2LMHeadModel
    if isinstance(model, GPT2LMHeadModel):
        # wpq: for gpt-2 model, need to enforce `max_length` constraints to avoid `position_id` index errors.
        generation_kwargs = {'max_length': model.config.max_position_embeddings} # 1024
    else:
        # wpq: modify `max_new_tokens=512` to `128` for faster generation.
        # for non-cot multiple choice answers, e.g., ' (G).' requires just 5 tokens
        generation_kwargs = {'max_new_tokens': 10 if args.no_cot else 128}

    outputs = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        batch_size=args.eval_batch_size if args.eval_batch_size else 1,
        stop_id_sequences=stop_id_sequences,
        **generation_kwargs,
    )

    predictions = []
    for example, output in zip(examples, outputs):
        example["raw_output"] = output
        
        # extract the first answer after `So the answer is` and before the next period.
        # if there is no such answer, we will just use the raw output.
        results = re.search(r"So the answer is (.*?)\.", output)
        if results:
            prediction = results.group(1).strip()
        else:
            # only keep the first part of the output - this is mainly for vanilla language models.
            output = output.strip().split("\n\n")[0].strip()
            prediction = output.strip()

        example["prediction"] = prediction
        predictions.append(prediction)
        if save_path:
            fout.write(json.dumps(example) + "\n")        

    assert len(predictions) == len(targets), "number of predictions and targets are not the same."
    return exact_match.compute(predictions=predictions, references=targets, ignore_case=True, ignore_punctuation=True)["exact_match"]


def eval_openai_chat_engine(args, examples, task_prompt, save_path=None):
    targets = [example["target"] for example in examples]
    instances = []
    for i, example in enumerate(examples):
        prompt = task_prompt.strip() + "\n\nQ: " + example["input"] + "\nA:"
        instances.append({
            "id": example["id"] if "id" in example else i,
            "prompt": prompt,
        })

    if save_path:
        openai_result_save_path = os.path.join(os.path.dirname(save_path), os.path.basename(save_path).split(".")[0] + "_openai_results.jsonl")
    
    results = query_openai_chat_model(
        engine=args.openai_engine,
        instances=instances,
        batch_size=args.eval_batch_size if args.eval_batch_size else 10,
        output_path=openai_result_save_path if save_path else None,
    )

    outputs = [result["output"] for result in results]
    assert len(outputs) == len(targets), "number of predictions and targets are not the same."

    if save_path:
        fout = open(save_path, "w")

    predictions = []
    for example, output in zip(examples, outputs):
        example["raw_output"] = output
        # extract the first answer after `So the answer is` and before the next period.
        # if there is no such answer, we will just use the raw output.
        results = re.search(r"So the answer is (.*?)\.", output)
        if results:
            prediction = results.group(1).strip()
        else:
            prediction = output.strip()
        example["prediction"] = prediction
        predictions.append(prediction)
        if save_path:
            fout.write(json.dumps(example) + "\n")        

    assert len(predictions) == len(targets), "number of predictions and targets are not the same."
    return exact_match.compute(predictions=predictions, references=targets, ignore_case=True, ignore_punctuation=True)["exact_match"]


def main(args):
    random.seed(42)

    all_tasks = {}
    task_files = glob.glob(os.path.join(args.data_dir, "bbh", "*.json"))
    for task_file in tqdm.tqdm(task_files, desc="Loading tasks"):
        with open(task_file, "r") as f:
            task_name = os.path.basename(task_file).split(".")[0]
            all_tasks[task_name] = json.load(f)["examples"]
            if args.max_num_examples_per_task:
                all_tasks[task_name] = random.sample(all_tasks[task_name], args.max_num_examples_per_task)

    all_prompts = {}
    cot_prompt_files = glob.glob(os.path.join(args.data_dir, "cot-prompts", "*.txt"))
    for cot_prompt_file in tqdm.tqdm(cot_prompt_files, desc="Loading prompts"):
        with open(cot_prompt_file, "r") as f:
            task_name = os.path.basename(cot_prompt_file).split(".")[0]
            task_prompt = "".join(f.readlines()[2:])
            if args.no_cot:
                prompt_fields = task_prompt.split("\n\n")
                new_prompt_fields = []
                for prompt_field in prompt_fields:
                    if prompt_field.startswith("Q:"):
                        assert "So the answer is" in prompt_field, f"`So the answer is` not found in prompt field of {task_name}.txt."
                        assert "\nA:" in prompt_field, "`\nA:` not found in prompt field."
                        answer = prompt_field.split("So the answer is")[-1].strip()
                        question = prompt_field.split("\nA:")[0].strip()
                        new_prompt_fields.append(question + "\nA: " + answer)
                    else:
                        new_prompt_fields.append(prompt_field)
                task_prompt = "\n\n".join(new_prompt_fields)
            all_prompts[task_name] = task_prompt

    assert set(all_tasks.keys()) == set(all_prompts.keys()), "task names in task data and task prompts are not the same."

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "predictions"), exist_ok=True)

    if args.model_name_or_path:
        print("Loading model and tokenizer...")
        model, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path, 
            tokenizer_name_or_path=args.tokenizer_name_or_path, 
            load_in_8bit=args.load_in_8bit, 
            device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
            gptq_model=args.gptq,
            use_fast_tokenizer=not args.use_slow_tokenizer,
        )

    performance = {}
    for task_name in tqdm.tqdm(all_tasks.keys(), desc="Evaluating"):
        task_examples = all_tasks[task_name]
        prompt = all_prompts[task_name]
        if args.model_name_or_path:
            task_perf = eval_hf_model(
                args, 
                model, 
                tokenizer, 
                task_examples, 
                prompt, 
                save_path=os.path.join(args.save_dir, "predictions", f"{task_name}.jsonl")
            )
        else:
            task_perf = eval_openai_chat_engine(
                args,
                task_examples,
                prompt,
                save_path=os.path.join(args.save_dir, "predictions", f"{task_name}.jsonl")
            )
        performance[task_name] = task_perf
        print(f"Task {task_name} - EM: {task_perf}")

    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
        performance["average_exact_match"] = sum(performance.values()) / len(performance)
        print(f"Average EM: {performance['average_exact_match']}")
        json.dump(performance, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="data/bbh"
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="results/bbh"
    )
    parser.add_argument(
        "--model_name_or_path", 
        type=str, 
        default=None, 
        help="if specified, we will load the model to generate the predictions."
    )
    parser.add_argument(
        "--tokenizer_name_or_path", 
        type=str, 
        default=None, 
        help="if specified, we will load the tokenizer from here."
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--openai_engine", 
        type=str, 
        default=None, 
        help="if specified, we will use the OpenAI API to generate the predictions."
    )
    parser.add_argument(
        "--no_cot", 
        action="store_true", 
        help="if specified, chain of thoughts will be removed from the prompts."
    )
    parser.add_argument(
        "--max_num_examples_per_task", 
        type=int, 
        default=None, 
        help="maximum number of examples to evaluate per task."
    )
    parser.add_argument(
        "--eval_batch_size", 
        type=int, 
        default=1, 
        help="batch size for evaluation."
    )
    parser.add_argument(
        "--load_in_8bit", 
        action="store_true", 
        help="load model in 8bit mode, which will reduce memory and speed up inference."
    )
    parser.add_argument(
        "--gptq", 
        action="store_true", 
        help="If given, we're evaluating a 4-bit quantized GPTQ model."
    )
    parser.add_argument(
        "--use_chat_format", 
        action="store_true", 
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function", 
        type=str, 
        default="eval.templates.create_prompt_with_tulu_chat_format", 
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    args = parser.parse_args()

    # model_name_or_path and openai_engine cannot be both None or both not None.
    assert (args.model_name_or_path is None) != (args.openai_engine is None), "Either model_name_or_path or openai_engine should be specified."
    main(args)
