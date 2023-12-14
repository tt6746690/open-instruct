import argparse
import os
import re
import json
import random
import pyarrow # wpq: added to prevent GLIBCXX not found error on aimos, put before `evaluate`, `torch`, `datasets`
import torch
import evaluate
from transformers import GPT2LMHeadModel
from eval.utils import (
    generate_completions,
    load_hf_lm_and_tokenizer,
    query_openai_chat_model,
    dynamic_import_function,
)
from eval.gsm.examplars import EXAMPLARS as GSM_EXAMPLARS


def main(args):
    random.seed(42)

    print("Loading data...")
    test_data = []
    with open(os.path.join(args.data_dir, f"test.jsonl")) as fin:
        for line in fin:
            example = json.loads(line)
            test_data.append({
                "question": example["question"],
                "answer": example["answer"].split("####")[1].strip()
            })
        
    # some numbers are in the `x,xxx` format, and we want to remove the comma
    for example in test_data:
        example["answer"] = re.sub(r"(\d),(\d)", r"\1\2", example["answer"])
        assert float(example["answer"]), f"answer is not a valid number: {example['answer']}"

    if args.max_num_examples and len(test_data) > args.max_num_examples:
        test_data = random.sample(test_data, args.max_num_examples)
        
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)


    def get_prompt_prefix(n_shot):
        global GSM_EXAMPLARS
        if n_shot:
            if len(GSM_EXAMPLARS) > n_shot:
                examples = random.sample(GSM_EXAMPLARS, n_shot)
            else:
                examples = GSM_EXAMPLARS
            demonstrations = []
            for example in examples:
                if args.no_cot:
                    demonstrations.append(
                        "Quesion: " + example["question"] + "\n" + "Answer: " + example["short_answer"]
                    )
                else:
                    demonstrations.append(
                        "Question: " + example["question"] + "\n" + "Answer: " + example["cot_answer"]
                    )
            prompt_prefix = "Answer the following questions.\n\n" + "\n\n".join(demonstrations) + "\n\n"
        else:
            prompt_prefix = "Answer the following question.\n\n"
        return prompt_prefix

    if args.model_name_or_path:
        print("Loading model and tokenizer...")
        if args.use_vllm:
            import vllm
            # prepare prompts
            prompt_prefix = get_prompt_prefix(args.n_shot)
            if args.use_chat_format:
                prompts = []
                chat_formatting_function = dynamic_import_function(args.chat_formatting_function)
                for example in test_data:
                    messages = [{"role": "user", "content": prompt_prefix + "Question: " + example["question"].strip()}]
                    prompt = chat_formatting_function(messages, add_bos=False)
                    prompt += "Answer:" if prompt[-1] in ["\n", " "] else " Answer:"
                    prompts.append(prompt)
            else:
                prompts = [prompt_prefix + "Question: " + example["question"].strip() + "\nAnswer:" for example in test_data]

            model = vllm.LLM(
                model=args.model_name_or_path,
                tokenizer=args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path,
                tokenizer_mode="slow" if args.use_slow_tokenizer else "auto",
                tensor_parallel_size=torch.cuda.device_count(),
            )
            sampling_params = vllm.SamplingParams(
                temperature=0,
                max_tokens=512,
                stop=["\n"] if not args.use_chat_format else None, # we only use stop token for non-chat format (usually applied to vanilla pretrained language models). For chat format, we will rely on the model knows when to stop.
            )
            # We need to remap the outputs to the prompts because vllm might not return outputs for some prompts (e.g., if the prompt is too long)
            generations = model.generate(prompts, sampling_params)
            prompt_to_output = {
                g.prompt: g.outputs[0].text for g in generations
            }
            outputs = [prompt_to_output[prompt] if prompt in prompt_to_output else "" for prompt in prompts]
        else:
            model, tokenizer = load_hf_lm_and_tokenizer(
                model_name_or_path=args.model_name_or_path, 
                tokenizer_name_or_path=args.tokenizer_name_or_path, 
                load_in_8bit=args.load_in_8bit, 
                device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
                gptq_model=args.gptq,
                use_fast_tokenizer=not args.use_slow_tokenizer,
            )

            ## prepare prompts, reduce number of icl examples if exceeds `max_input_seq_len`.
            # wpq: for gpt-2 model, need to enforce `max_length` constraints to avoid `position_id` index errors.
            if isinstance(model, GPT2LMHeadModel):
                max_input_seq_len = model.config.max_position_embeddings - args.max_new_tokens
            else:
                max_input_seq_len = 2048 - args.max_new_tokens
            prompts = []
            chat_formatting_function = dynamic_import_function(args.chat_formatting_function) if args.use_chat_format else None
            for example in test_data:
                ## wpq: Use <n_shot prompt if exceeds `max_input_seq_len`.
                for n_shot in list(range(args.n_shot+1)[::-1]):
                    prompt_prefix = get_prompt_prefix(n_shot)
                    prompt = prompt_prefix + "Question: " + example["question"].strip()
                    if args.use_chat_format:
                        messages = [{"role": "user", "content": prompt}]
                        prompt = chat_formatting_function(messages, add_bos=False)
                        prompt += "Answer:" if prompt[-1] in ["\n", " "] else " Answer:"
                    else:
                        prompt += "\nAnswer:"
                    tokenized_prompt_len = len(tokenizer(prompt, add_special_tokens=False)['input_ids'])
                    if tokenized_prompt_len < max_input_seq_len:
                        break
                if n_shot != args.n_shot:
                    print(f'n_shot: {args.n_shot} -> {n_shot}')
                prompts.append(prompt)

            new_line_token = tokenizer.encode("\n", add_special_tokens=False)[-1] # get the last token because the tokenizer may add space tokens at the start.
            outputs = generate_completions(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.eval_batch_size,
                stop_id_sequences=[[new_line_token]],
                ## wpq: this hurts baseline instruct models (e.g., mistral-7b-instruct, llama2-7b-chat). For now use stop tokens whenever possible.
                # stop_id_sequences=[[new_line_token]] if not args.use_chat_format else None,  # we only use stop token for non-chat format (usually applied to vanilla pretrained language models). For chat format, we will rely on the model knows when to stop.
                do_sample=False,
            )

    else:
        instances = [{"id": prompt, "prompt": prompt} for _, prompt in enumerate(prompts)]
        results = query_openai_chat_model(
            engine=args.openai_engine,
            instances=instances,
            batch_size=args.eval_batch_size if args.eval_batch_size else 10,
            output_path=os.path.join(args.save_dir, f"openai_results.jsonl"),
        )
        outputs = [result["output"] for result in results]

    predictions = []
    for output in outputs:
        # replace numbers like `x,xxx` with `xxxx`
        output = re.sub(r"(\d),(\d)", r"\1\2", output)
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
        if numbers:
            predictions.append(numbers[-1])
        else:
            predictions.append(output)
        
    print("Calculating accuracy...")
    targets = [example["answer"] for example in test_data]

    em_score = args.exact_match.compute(predictions=predictions, references=targets, ignore_case=True, ignore_punctuation=True)["exact_match"]
    print(f"Exact match : {em_score}")

    predictions = [{
        "question": example["question"],
        "answer": example["answer"],
        "model_output": output,
        "prediction": pred
    } for example, output, pred in zip(test_data, outputs, predictions)]

    with open(os.path.join(args.save_dir, f"predictions.jsonl"), "w") as fout:
        for prediction in predictions:
            fout.write(json.dumps(prediction) + "\n") 
    
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
        json.dump({
            "exact_match": em_score
        }, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/mgsm")
    parser.add_argument("--max_num_examples", type=int, default=None, help="maximum number of examples to evaluate.")
    parser.add_argument("--save_dir", type=str, default="results/mgsm")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None, help="if specified, we will load the tokenizer from here.")
    parser.add_argument("--use_slow_tokenizer", action="store_true", help="If given, we will use the slow tokenizer.")
    parser.add_argument("--openai_engine", type=str, default=None, help="if specified, we will use the OpenAI API to generate the predictions.")
    parser.add_argument("--n_shot", type=int, default=8, help="max number of examples to use for demonstration.")
    parser.add_argument("--no_cot", action="store_true", help="If given, we're evaluating a model without chain-of-thought.")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")
    parser.add_argument("--load_in_8bit", action="store_true", help="load model in 8bit mode, which will reduce memory and speed up inference.")
    parser.add_argument("--gptq", action="store_true", help="If given, we're evaluating a 4-bit quantized GPTQ model.")
    parser.add_argument("--use_vllm", action="store_true", help="If given, we will use the vllm library, which will likely increase the inference throughput.")
    parser.add_argument("--use_chat_format", action="store_true", help="If given, the prompt will be encoded as a chat format with the roles in prompt.")
    parser.add_argument("--chat_formatting_function", type=str, default="eval.templates.create_prompt_with_tulu_chat_format", help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`.")
    parser.add_argument("--max_new_tokens", type=int, default=512)

    args = parser.parse_args()

    # wpq: prevents the following error.
    # `ValueError: Error in finalize: another evaluation module instance is already using the local cache file. Please specify an experiment_id to avoid collision between distributed evaluation module instances.`
    args.exact_match = evaluate.load("exact_match", experiment_id=args.save_dir, keep_in_memory=True)

    # model_name_or_path and openai_engine cannot be both None or both not None.
    assert (args.model_name_or_path is None) != (args.openai_engine is None), "Either model_name_or_path or openai_engine should be specified."
    main(args)
