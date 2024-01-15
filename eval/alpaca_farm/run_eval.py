import os
import json
import argparse
import logging
import random
import numpy as np
import pyarrow # wpq: added to prevent GLIBCXX not found error on aimos, put before `evaluate`, `torch`, `datasets`
import torch
import datasets
from alpaca_eval import evaluate as alpaca_farm_evaluate
from eval.utils import query_openai_chat_model, query_openai_model, generate_completions, dynamic_import_function, load_hf_lm_and_tokenizer


def main(args):
    random.seed(42)
    os.makedirs(args.save_dir, exist_ok=True)

    logging.info("loading data and model...")
    alpaca_eval_data = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]

    if args.max_num_examples and len(alpaca_eval_data) > args.max_num_examples:
        inds = random.sample(range(len(alpaca_eval_data)), args.max_num_examples)
        print('Selecting data subsets: ', inds)
        alpaca_eval_data = alpaca_eval_data.select(inds)

    prompts = []
    chat_formatting_function = dynamic_import_function(args.chat_formatting_function) if args.use_chat_format else None
    for example in alpaca_eval_data:
        prompt = example["instruction"]
        if args.use_chat_format:
            messages = [{"role": "user", "content": prompt}]
            prompt = chat_formatting_function(messages, add_bos=False)
        prompts.append(prompt)

    if args.model_name_or_path is not None:
        if args.use_vllm:
            import vllm
            model = vllm.LLM(
                model=args.model_name_or_path,
                tokenizer=args.tokenizer_name_or_path if args.tokenizer_name_or_path is not None else args.model_name_or_path,
                # tokenizer_mode="slow",
                tensor_parallel_size=torch.cuda.device_count(),
                dtype=getattr(torch, args.torch_dtype),
            )
            sampling_params = vllm.SamplingParams(
                temperature=0,  # greedy decoding
                max_tokens=args.max_new_tokens,
            )
            outputs = model.generate(prompts, sampling_params)
            outputs = [it.outputs[0].text for it in outputs]
        else:
            model, tokenizer = load_hf_lm_and_tokenizer(
                model_name_or_path=args.model_name_or_path,
                tokenizer_name_or_path=args.tokenizer_name_or_path if args.tokenizer_name_or_path is not None else args.model_name_or_path,
                load_in_8bit=args.load_in_8bit,
                device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
                gptq_model=args.gptq,
                use_fast_tokenizer=not args.use_slow_tokenizer,
                torch_dtype=getattr(torch, args.torch_dtype),
            )
            outputs = generate_completions( 
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=0,
                batch_size=args.eval_batch_size if args.eval_batch_size else 1,
            )
    else:
        openai_query_cache_path = os.path.join(args.save_dir, "openai_query_cache.jsonl")
        openai_func = query_openai_model if args.openai_engine == "text-davinci-003" else query_openai_chat_model
        results = openai_func(
            engine=args.openai_engine,
            instances=[{"id": str(i), "prompt": prompt} for i, prompt in enumerate(prompts)],
            batch_size=args.eval_batch_size if args.eval_batch_size else 10,
            output_path=openai_query_cache_path,
            max_tokens=args.max_new_tokens,
            temperature=0,
            reuse_existing_outputs=True,
        )
        outputs = [result["output"] for result in results]

    model_name = os.path.basename(os.path.normpath(args.model_name_or_path)) if args.model_name_or_path is not None else args.openai_engine
    model_results = []
    with open(os.path.join(args.save_dir, f"{model_name}-greedy-long-output.json"), "w") as fout:
        for example, output in zip(alpaca_eval_data, outputs):
            example["output"] = output
            example["generator"] = f"{model_name}-greedy-long"
            fout.write(json.dumps(example) + "\n")
            model_results.append(example)

    df_leaderboard, annotations = alpaca_farm_evaluate(
        model_outputs=model_results,
        reference_outputs=args.reference_path if args.reference_path != 'alpaca_eval_data' else alpaca_eval_data,
        annotators_config=args.annotators_config,
        output_path=args.save_dir,
        is_return_instead_of_print=True,
        caching_path=os.path.join(args.save_dir, "alpaca_eval_annotator_cache.json"),
        precomputed_leaderboard=None,
        is_cache_leaderboard=False
    )

    prices = np.array([x['price_per_example'] for x in annotations], dtype=np.float32)
    print(f'Price (per-example / total) = {np.nanmean(prices):.4f} / {np.nansum(prices):.2f}')

    times = np.array([x['time_per_example'] for x in annotations], dtype=np.float32)
    print(f'Time  (per-example / total) = {np.nanmean(times):.4f} / {np.nansum(times):.2f}')

    df_leaderboard['avg_output_tok_length'] = np.mean(
        [len(tokenizer(x['output'])['input_ids']) for x in model_results])
    df_leaderboard['price'] = np.nansum(prices)
    df_leaderboard.insert(0, 'model', df_leaderboard.index)
    df_leaderboard = df_leaderboard.reset_index(drop=True)

    print(df_leaderboard.to_string(float_format="%.2f"))

    # save to json
    with open(os.path.join(args.save_dir, f"metrics.json"), "w") as fout:
        json.dump(df_leaderboard.iloc[0].to_dict(), fout)

    ## wpq: update `metrics.json` with repetitiveness adjusted win_rate.
    try:
        from scripts.note_pruning_analysis import update_metrics_with_highly_repeated_chars
        update_metrics_with_highly_repeated_chars(args.save_dir, update_metrics_file=True)
    except:
        print('Running `update_metrics_with_highly_repeated_chars` failed.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_path", type=str, default="data/eval/alpaca_farm/davinci_003_outputs_2048_token.json", help=
                        "Path to the reference outputs. "
                        "Alpaca_eval leaderboard use text-davinci-003 to generate the reference outputs, "
                        "but they limit the max_tokens to 300, which is a bit unfair for text-davinci-003. "
                        "Here we keep this default setup to make numbers comparable to their leaderboard. "
                        "But you can also use the regenerated reference outputs with max_tokens=2048 "
                        "hosted at https://huggingface.co/datasets/hamishivi/alpaca-farm-davinci-003-2048-token.",)
    parser.add_argument("--save_dir", type=str, default="results/alpaca_farm")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="If specified, we will load the model to generate the predictions.")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None, help="If specified, we will load the tokenizer from here.")
    parser.add_argument("--use_slow_tokenizer", action="store_true", help="If given, we will use the slow tokenizer.")
    parser.add_argument("--openai_engine", type=str, default=None, help="If specified, we will use the OpenAI API to generate the predictions.")
    parser.add_argument("--max_new_tokens", type=int, default=8192, help="Maximum number of tokens to generate.")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Batch size for evaluation.")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8bit mode, which will reduce memory and speed up inference.")
    parser.add_argument("--gptq", action="store_true", help="If given, we're evaluating a 4-bit quantized GPTQ model.")
    parser.add_argument("--use_chat_format", action="store_true", help="If given, we will use the chat format for the prompts.")
    parser.add_argument("--chat_formatting_function", type=str, default="eval.templates.create_prompt_with_tulu_chat_format", help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`.")
    parser.add_argument("--use_vllm", action="store_true", help="If given, we will use vLLM to generate the predictions - much faster.")
    parser.add_argument("--annotators_config", type=str, default="alpaca_eval_gpt4_0314")
    parser.add_argument("--max_num_examples", type=int, default=8192, help="maximum number of examples to evaluate.")
    parser.add_argument("--torch_dtype", type=str, default='float16', choices=['float16', 'bfloat16'])

    args = parser.parse_args()

    # model_name_or_path and openai_engine cannot be both None or both not None.
    assert (args.model_name_or_path is None) != (args.openai_engine is None), "Either model_name_or_path or openai_engine should be specified."
    main(args)