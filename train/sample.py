"""
A script for sampling from LLMs. It should be run like this:

python -m train.sample /models/llama3-8B-sft/FINAL --output_file outputs.json \ 
    --gpu_count 4 --datasets alpacaeval --num_samples_per_prompt 4

The resulting JSON file with have items with the following fields:

- prompt: a list of key-value pairs with speaker and content
- instruction: only for one-turn eval datasets like alpacaeval
- output: clean output, without the chat template
- generator: path to either local model dir or Huggingface repo
- dataset: specific dataset that the prompt is from 
- split: either 'train' or 'test'
- prompt_id: unique integer for the prompt
- sample_id: integer from 0 to k - 1 for one of the k samples produced per prompt_id
- type: set to "sample"

The (prompt_id, sample_id) pair uniquely identifies each entry.
An exit code of 1 is returned if not all the data has been processed; 0 otherwise.

Note that the keys 'instruction', 'output' are necessary to run Alpacaeval on the samples.
"""
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
    tokenizer.chat_template = open('config/template.jinja').read()
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

        # Initialize the SFTDataLoader
        dataloader = SFTDataLoader(
            dataset_names=args.datasets,
            tokenizer=tokenizer,
            split=args.split,
            max_prompt_length=args.max_prompt_length,
            n_epochs=args.num_epochs,
            seed=args.seed,
            microbatch_size=(args.num_prompts if args.num_prompts else args.batch_size),
            n_examples=args.num_prompts, 
            num_skip=args.num_skip,
        )
        
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
        
        # Generate all responses at once; around 4x faster than generating per batch (39.84 toks/s -> 138.40 toks/s)
        print(f"Generating responses for {len(all_prompt_texts)} prompts...")
        all_responses = llm.generate(all_prompt_texts, sampling_params)
        # Filter out all responses where model generated too many tokens
        import pdb; pdb.set_trace()
        group_size = 8
        filtered_respsonses = [
            r for r in all_responses if all([o.finish_reason != "length" for o in r.outputs])
        ]



        for i in range(0, len(all_responses), group_size):
            group = all_responses[i:i + group_size]
            # filtered_group = [r for r in group if len(r.outputs) > 0 and len(r.outputs[0].text) < args.max_tokens]
            # filtered_respsonses.extend(filtered_group)
            # Use Stop reason instead
            filtered_group = [r for r in group if r.outputs[0].finish_reason != "length"]
            if len(filtered_group) != len(group): continue
            filtered_respsonses.extend(filtered_group)
        all_responses = filtered_respsonses
        print(f"Generated {len(all_responses)} valid responses.")
        
        # Process and write each output
        for prompt_idx, (prompt, response, dataset_name, target) in enumerate(
            zip(all_original_prompts, all_responses, all_dataset_names, all_targets)
        ):
            for sample_idx, sample in enumerate(response.outputs):
                output = {
                    "output": re.sub(r"<?\|(im_start|im_end)\|>?", "", sample.text.strip()),
                    "generator": args.model_path,
                    "dataset": f"{dataset_name}_{args.split}",
                    "prompt_id": prompt_idx,
                    "sample_id": sample_idx,
                    "type": "sample",
                    "answer": target[0]['content'] if target else None,
                }

                # for eval with alpacaeval
                if args.mode == "alpacaeval":
                    output["instruction"] = prompt[0]["content"]
                else:
                    output["prompt"] = prompt

                writer.write_item(output)
        
        writer.close()

    destroy_model_parallel()
    destroy_distributed_environment()

    if prompt_idx == 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample from a local model using vllm for AlpacaEval")
    parser.add_argument("model_path", type=str, help="Path to the local model folder or the Huggingface repo")
    parser.add_argument("--datasets", nargs="+", default=["alpacaeval"], help="List of datasets to sample from (space-separated)")
    parser.add_argument("--output_file", type=str, default="outputs.json", help="Path to save the output JSON file")
    parser.add_argument("--gpu_count", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum number of tokens to generate")
    parser.add_argument("--max_prompt_length", type=int, default=512, help="Maximum length of prompt (in tokens)")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for processing datasets")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use (train/test)")
    parser.add_argument("--num_samples_per_prompt", type=int, default=1, help="Number of samples to generate per input")
    parser.add_argument("--stop_token", type=str, default='<|im_end|>', help="Stop token")
    parser.add_argument("--mode", type=str, default="alpacaeval", help="mode")
    parser.add_argument("--num_prompts", type=int, default=None, help="number of prompts to sample from")
    parser.add_argument("--num_skip", type=int, default=0, help="number of prompts to skip at the beginning")
    parser.add_argument("--num_epochs", type=int, default=1, help="number of times to pass through the data (in order)")

    args = parser.parse_args()
    main(args)