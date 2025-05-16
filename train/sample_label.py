import argparse
import re
import sys
import inspect
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from .dataloader import SFTDataLoader
from .utils import set_offline_if_needed, StreamingJSONWriter
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
)
from . import data as data_module

from simpleverify.verify_generic import verify_generic_cached
from simpleverify.verify_math import verify_math_cached


def get_available_datasets():
    """Get list of available datasets by finding all get_* functions in dataloader.py"""
    return [name[4:] for name, _ in inspect.getmembers(data_module, inspect.isfunction) 
            if name.startswith('get_')]


def validate_datasets(datasets):
    """Validate that all requested datasets have corresponding get_* functions"""
    available_datasets = get_available_datasets()
    invalid_datasets = [d for d in datasets if d not in available_datasets]
    
    if invalid_datasets:
        available_str = "\n- ".join(available_datasets)
        raise ValueError(
            f"The following datasets are not available: {invalid_datasets}\n"
            f"Available datasets must have a corresponding get_* function in train.data\n"
            f"Currently available datasets are:\n- {available_str}"
        )


def main(args):
    validate_datasets(args.datasets)
    set_offline_if_needed()

    # Load the model and tokenizer
    print(f"Loading model and tokenizer from {args.model_path}")
    llm = LLM(model=args.model_path, tensor_parallel_size=args.gpu_count)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # tokenizer.chat_template = open('config/template.jinja').read()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=[args.stop_token],
        n=args.num_samples_per_prompt
    )

    # Open the output file and create a streaming writer
    with open(args.output_file, 'w') as f:
        writer = StreamingJSONWriter(f)

        prompts_left = args.num_prompts
        num_skip = args.num_skip
        while prompts_left:
            n_examples = prompts_left * args.oversample if args.oversample is not None else prompts_left
            dataloader = SFTDataLoader(
                dataset_names=args.datasets,
                tokenizer=tokenizer,
                split=args.split,
                max_prompt_length=args.max_prompt_length,
                n_epochs=args.num_epochs,
                seed=args.seed,
                microbatch_size=n_examples,
                n_examples=n_examples, 
                num_skip=num_skip,
            )
            num_skip += n_examples # won't be 100% accurate if oversampling and early breaking but close enough

            # Collect all prompts, original prompts, and dataset names
            all_prompt_texts = []
            all_original_prompts = []
            all_dataset_names = []
            all_targets = []

            print("Collecting all prompts...")
            for batch in dataloader:
                all_prompt_texts.extend(batch['prompt_text'])
                all_original_prompts.extend(batch['prompt'])
                all_dataset_names.extend(batch['dataset_name'])
                if "target" in batch:
                    all_targets.extend(batch['target'])
                else:
                    all_targets.extend([None] * len(batch['prompt_text']))

            
            print(f"Num duplicates: {len(all_prompt_texts) - len(set(all_prompt_texts))}")
            # Generate all responses at once; around 4x faster than generating per batch (39.84 toks/s -> 138.40 toks/s)
            print(f"Generating responses for {len(all_prompt_texts)} prompts...")
            # 31:50 min for 12K MATH questions with 4096 seq len
            all_responses = llm.generate(all_prompt_texts, sampling_params)
            # Filter out responses that are too long; filters 12K -> 10966 for MATH; Done in loop now instead
            # all_responses = [r for r in all_responses if all([o.finish_reason != "length" for o in r.outputs])]
            # print(f"Left with {len(all_responses)} responses that are not too long.")

            acc = []
            skipped_len = 0
            skipped_verify = 0
            # Process and write each output
            for prompt_idx, (prompt, response, dataset_name, target) in enumerate(
                zip(all_original_prompts, all_responses, all_dataset_names, all_targets)
            ):
                if any([o.finish_reason == "length" for o in response.outputs]):
                    skipped_len += 1
                    continue
                if args.verifyfn == "math":
                    rewards = [r[0] for r in verify_math_cached(
                        [r.text for r in response.outputs],
                        target[0]['content'],
                        sep="</think>",
                        na_to_zero=True,
                    )]
                    # rewards = [verify_math_cached([r.outputs[0].text], t[0]['content'], sep="</think>", na_to_zero=True)[0] for r,t in zip(all_responses_1_l, all_targets)]

                elif args.verifyfn == "generic":
                    rewards = [r[0] for r in verify_generic_cached(
                        [r.text for r in response.outputs],
                        target[0]['content'],
                        sep="</think>",
                        na_to_zero=True,
                    )]
                    # print(rewards)
                    # import pdb; pdb.set_trace()
                # Skip if all rewards are the same as no signal in GRPO
                # Filters 12K -> 7089 for MATH
                # acc.append(rewards)
                if len(set(rewards)) == 1:
                    skipped_verify += 1
                    continue
                for sample_idx, sample in enumerate(response.outputs):
                    output = {
                        "prompt": prompt,
                        "output": [{"role": "assistant", "content": re.sub(r"<?\|(im_start|im_end)\|>?", "", sample.text.strip())}],
                        "generator": args.model_path,
                        "dataset": f"{dataset_name}_{args.split}",
                        "prompt_id": args.num_prompts - prompts_left,
                        "sample_id": sample_idx,
                        "type": "binary_feedback",
                        "answer": target[0]['content'],
                        "label": rewards[sample_idx],
                        "reward": rewards[sample_idx],
                    }
                    writer.write_item(output)
                prompts_left -= 1
                if prompts_left == 0: break
            print("Skipped len:", skipped_len)
            print("Skipped verify:", skipped_verify)
            print("Prompts left:", prompts_left)
            # import pdb; pdb.set_trace()
            if prompts_left == 0: break
        writer.close()

    # print("Accuracy:", sum(acc) / len(acc))
    # [x for x in acc if sum(x) not in (0, len(x))]
    # import pdb; pdb.set_trace()
    # llm.engine.shutdown()
    destroy_model_parallel()
    destroy_distributed_environment()
    # del llm
    # import gc
    # gc.collect()
    # import torch
    # torch.cuda.empty_cache()
    # import pdb; pdb.set_trace()
    # import os
    # os._exit(0)

    # sys.exit(0)

    # import gc
    # import contextlib
    # import ray
    # import torch
    # del llm
    # with contextlib.suppress(AssertionError):
    #     torch.distributed.destroy_process_group()
    # gc.collect()
    # torch.cuda.empty_cache()
    # ray.shutdown()

    print("NUMSKIP=", num_skip) # Capture for future runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample from a local model using vllm for AlpacaEval")
    parser.add_argument("model_path", type=str, help="Path to the local model folder or the Huggingface repo")
    parser.add_argument("--datasets", nargs="+", default=["alpacaeval"], help="List of datasets to sample from (space-separated)")
    parser.add_argument("--verifyfn", type=str, default="math", help="math/generic")    
    parser.add_argument("--output_file", type=str, default="outputs.json", help="Path to save the output JSON file")
    parser.add_argument("--gpu_count", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling parameter")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum number of tokens to generate")
    parser.add_argument("--max_prompt_length", type=int, default=512, help="Maximum length of prompt (in tokens)")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for processing datasets")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use (train/test)")
    parser.add_argument("--num_samples_per_prompt", type=int, default=1, help="Number of samples to generate per input")
    parser.add_argument("--stop_token", type=str, default='<|im_end|>', help="Stop token")
    parser.add_argument("--mode", type=str, default="alpacaeval", help="mode")
    parser.add_argument("--num_prompts", type=int, default=None, help="number of prompts to sample from")
    parser.add_argument("--oversample", type=float, default=None, help="oversample in first round of generation to speed things up")
    parser.add_argument("--num_skip", type=int, default=0, help="number of prompts to skip at the beginning")
    parser.add_argument("--num_epochs", type=int, default=None, help="number of times to pass through the data (in order)")

    args = parser.parse_args()
    main(args)

# Upgrade vLLM?
# Check eval with same seq len; maybe increase seq len

# 3235 (25880/25880) -> Left with 2949 responses that are not too long; losing ~10%
