import json
from transformers import AutoTokenizer

paths = [
    # "/data/niklas/HALOs/models/qwen15-online-grpo-filter-bs1024-kl-0.0-lr-1e-6-clip-0.2-deepscaler/R1_samples.json",
    # "/data/niklas/HALOs/models/qwen15-online-grpo-filter-bs1024-kl-0.0-lr-1e-6-clip-0.2-deepscaler/R2_samples.json",
    # "/data/niklas/HALOs/models/qwen15-online-grpo-filter-bs1024-kl-0.0-lr-1e-6-clip-0.2-deepscaler/R3_samples.json",
    # "/data/niklas/HALOs/models/qwen15-online-grpo-filter-bs1024-kl-0.0-lr-1e-6-clip-0.2-deepscaler/R4_samples.json",
    # "/data/niklas/HALOs/models/qwen15-online-grpo-filter-bs1024-kl-0.0-lr-1e-6-clip-0.2-deepscaler/R5_samples.json",
    # "/data/niklas/HALOs/models/qwen15-online-grpo-filter-bs1024-kl-0.0-lr-1e-6-clip-0.2-deepscaler/R6_samples.json",
    # "/data/niklas/HALOs/models/qwen15-online-grpo-filter-bs1024-kl-0.0-lr-1e-6-clip-0.2-deepscaler/R7_samples.json",
    "/data/niklas/HALOs/models/qwen15-online-grpo-filter-bs1024-kl-0.0-lr-1e-6-clip-0.2-deepscaler/R8_samples.json",
    "/data/niklas/HALOs/models/qwen15-online-grpo-filter-bs1024-kl-0.0-lr-1e-6-clip-0.2-deepscaler/R9_samples.json",
    "/data/niklas/HALOs/models/qwen15-online-grpo-filter-bs1024-kl-0.0-lr-1e-6-clip-0.2-deepscaler/R10_samples.json",
    "/data/niklas/HALOs/models/qwen15-online-grpo-filter-bs1024-kl-0.0-lr-1e-6-clip-0.2-deepscaler/R11_samples.json",
]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

for p in paths:
    print(p)
    with open(p, "r") as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"Error loading JSON: {e}")
            continue
    lens = [len(tok.tokenize(d["output"][0]['content'])) for d in data]
    print("max", max(lens))
    print("min", min(lens))
    print("mean", sum(lens) / len(lens))

