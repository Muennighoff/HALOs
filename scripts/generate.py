from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "/data/niklas/models/qwen-32b-kto-humanline-pos/FINAL"
model_name = "/data/niklas/models/qwen-32b-kto-humanline-mathstepdpo/FINAL"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "How many r in raspberry"
messages = [
    {"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# text += "<|im_start|>think\n"
#text += "Let's think step by step."

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


### vLLM version
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model = LLM(
    model_name,
    tensor_parallel_size=1,
    max_model_len=10000,
)
tok = AutoTokenizer.from_pretrained(model_name)

stop_token_ids = tok("<|im_end|>")["input_ids"]

sampling_params = SamplingParams(
    max_tokens=32768,
    min_tokens=0,
    stop_token_ids=stop_token_ids,
)

prompt = "How many r in raspberry"
prompt = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
text += "Let's think step by step."

o = model.generate(prompt, sampling_params=sampling_params)
print(o[0].outputs[0].text)