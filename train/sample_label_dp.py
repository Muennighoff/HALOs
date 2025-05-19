import argparse
import re
import sys
import inspect
from typing import List
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

def undistribute(iterable):
    """
    Undoes https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.distribute .

    Re-interleaves results that have been split using more_itertools.distribute:
        >>> group_1, group_2 = distribute(2, [1, 2, 3, 4, 5, 6])
        >>> list(group_1)
        [1, 3, 5]
        >>> list(group_2)
        [2, 4, 6]
        >>> undistribute([group_1, group_2])
        [1, 2, 3, 4, 5, 6]

    Handles non-uniform component lengths:

        >>> children = distribute(3, [1, 2, 3, 4, 5, 6, 7])
        >>> [list(c) for c in children]
        [[1, 4, 7], [2, 5], [3, 6]]
        >>> undistribute(children)
        [1, 2, 3, 4, 5, 6, 7]

    Also handles when some iterables are empty:

        >>> children = distribute(5, [1, 2, 3])
        >>> [list(c) for c in children]
        [[1], [2], [3], [], []]
        >>> undistribute(children)
        [1, 2, 3]

    """
    import itertools
    return [
        x
        for x in itertools.chain.from_iterable(
            itertools.zip_longest(*[list(x) for x in iterable])
        )
        if x is not None
    ]

def main(args):
    validate_datasets(args.datasets)
    set_offline_if_needed()

    # Load the model and tokenizer
    print(f"Loading model and tokenizer from {args.model_path}")
    if args.engine == "vllm":
        llm = LLM(model=args.model_path, tensor_parallel_size=args.tp)
    elif args.engine == "sgl":
        import sglang as sgl
        llm = sgl.Engine(model_path=args.model_path, tp_size=args.tp, dp_size=args.dp)#, mem_fraction_static=0.7)
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
        prompts_set = set()
        while prompts_left:
            n_examples = int(prompts_left * args.oversample) if args.oversample is not None else prompts_left
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


            unique_prompts = set(all_prompt_texts)
            num_dups = len(all_prompt_texts) - len(unique_prompts)
            if num_dups > 0:
                print(f"Found {num_dups} duplicate prompts. Removing...")
                new_prompts, new_original_prompts, new_dataset_names, new_targets = [], [], [], []
                for prompt, original_prompt, dataset_name, target in zip(all_prompt_texts, all_original_prompts, all_dataset_names, all_targets):
                    if prompt not in unique_prompts: continue
                    unique_prompts.remove(prompt)
                    new_prompts.append(prompt)
                    new_original_prompts.append(original_prompt)
                    new_dataset_names.append(dataset_name)
                    new_targets.append(target)
                all_prompt_texts, all_original_prompts, all_dataset_names, all_targets = new_prompts, new_original_prompts, new_dataset_names, new_targets


            print(f"Num duplicates: {len(all_prompt_texts) - len(set(all_prompt_texts))}")
            # Generate all responses at once; around 4x faster than generating per batch (39.84 toks/s -> 138.40 toks/s)
            print(f"Generating responses for {len(all_prompt_texts)} prompts...")
            if args.engine == "vllm":
                if args.dp > 1:
                    import ray
                    from more_itertools import distribute
                    # vLLM hangs if resources are set in ray.remote
                    # also seems to only work with decorator and not with ray.remote() fn
                    # see https://github.com/vllm-project/vllm/issues/973
                    @ray.remote
                    def run_inference_one_model(
                        model_args: dict,
                        sampling_params: SamplingParams,
                        requests: List[List[int]],
                    ):
                        llm = LLM(**model_args)
                        return llm.generate(
                            prompt_token_ids=requests,
                            sampling_params=sampling_params,
                        )
                    # dispatch requests to all self.data_parallel_size workers, in interleaved fashion
                    # interleaved important to balance context lengths across workers
                    requests = [list(x) for x in distribute(args.dp, all_prompt_texts)]
                    inputs = (
                        (
                            dict(model=args.model_path, tensor_parallel_size=args.tp, distributed_executor_backend="ray"),
                            sampling_params,
                            req,
                        )
                        for req in requests
                    )
                    object_refs = [run_inference_one_model.remote(*x) for x in inputs]
                    results = ray.get(object_refs)
                    # Invoke ray.shutdown() to prevent hang-ups if subsequent calls required.
                    ray.shutdown()
                    # flatten results
                    all_responses = undistribute(results)
                else:
                    # 31:50 min for 12K MATH questions with 4096 seq len
                    all_responses = llm.generate(all_prompt_texts, sampling_params)
                    # Filter out responses that are too long; filters 12K -> 10966 for MATH; Done in loop now instead
                    # all_responses = [r for r in all_responses if all([o.finish_reason != "length" for o in r.outputs])]
                    # print(f"Left with {len(all_responses)} responses that are not too long.")
            elif args.engine == "sgl":
                # SGL engine
                all_responses = llm.generate(
                    all_prompt_texts, 
                    sampling_params={
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_new_tokens": args.max_tokens,
                    "stop": [args.stop_token],
                    "n": args.num_samples_per_prompt,
                })
                all_responses = [all_responses[i:i + args.num_samples_per_prompt] for i in range(0, len(all_responses), args.num_samples_per_prompt)]

            # acc = []
            skipped_len, skipped_verify, skipped_dup = 0, 0, 0
            # Process and write each output
            for prompt_idx, (prompt, response, dataset_name, target) in enumerate(
                zip(all_original_prompts, all_responses, all_dataset_names, all_targets)
            ):
                if (args.engine == "vllm" and any([o.finish_reason == "length" for o in response.outputs])) or \
                   (args.engine == "sgl" and any([o['meta_info']['finish_reason'] == "length" for o in response])):
                    skipped_len += 1
                    continue
                if prompt in prompts_set:
                    skipped_dup += 1
                    continue
                prompts_set.add(prompt)
                txts = [o.text for o in response.outputs] if args.engine == "vllm" else [o['text'] for o in response]
                if args.verifyfn == "math":
                    rewards = [r[0] for r in verify_math_cached(
                        txts,
                        target[0]['content'],
                        sep="</think>",
                        na_to_zero=True,
                    )]
                elif args.verifyfn == "generic":
                    rewards = [r[0] for r in verify_generic_cached(
                        txts,
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
                iterate = response.outputs if args.engine == "vllm" else response
                for sample_idx, sample in enumerate(iterate):
                    output = {
                        "prompt": prompt,
                        "generator": args.model_path,
                        "dataset": f"{dataset_name}_{args.split}",
                        "prompt_id": args.num_prompts - prompts_left,
                        "sample_id": sample_idx,
                        "type": "binary_feedback",
                        "answer": target[0]['content'],
                        "label": rewards[sample_idx],
                        "reward": rewards[sample_idx],
                    }
                    if args.engine == "vllm":
                        output["output"] = [{"role": "assistant", "content": re.sub(r"<?\|(im_start|im_end)\|>?", "", sample.text.strip())}]
                    elif args.engine == "sgl":
                        output["output"] = [{"role": "assistant", "content": re.sub(r"<?\|(im_start|im_end)\|>?", "", sample['text'].strip())}]
                    writer.write_item(output)
                prompts_left -= 1
                if prompts_left == 0: break
            print("Skipped len:", skipped_len)
            print("Skipped verify:", skipped_verify)
            print("Prompts left:", prompts_left)
            # import pdb; pdb.set_trace()
            if prompts_left == 0: break
        writer.close()

    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample from a local model using vllm for AlpacaEval")
    parser.add_argument("model_path", type=str, help="Path to the local model folder or the Huggingface repo")
    parser.add_argument("--datasets", nargs="+", default=["alpacaeval"], help="List of datasets to sample from (space-separated)")
    parser.add_argument("--verifyfn", type=str, default="math", help="math/generic")    
    parser.add_argument("--output_file", type=str, default="outputs.json", help="Path to save the output JSON file")
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
    parser.add_argument("--tp", type=int, default=1, help="Number of tensor parallel gpus")
    parser.add_argument("--dp", type=int, default=1, help="number of data parallel workers to use; sometimes hangs")
    parser.add_argument("--engine", type=str, default="vllm", help="vllm/sgl")

    args = parser.parse_args()
    main(args)
