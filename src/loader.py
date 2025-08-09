import os
from huggingface_hub import snapshot_download
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer

# -------------------------
# Configuration
# -------------------------

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN environment variable not set. Please export your Hugging Face token before running.")

MODEL_REPO = "ibm-granite/granite-8b-code-instruct-128k"
OUTPUT_DIR = "./granite_8b_q5"
BITS = 8
GROUP_SIZE = 128
DESC_ACT = False

# -------------------------
# Download or reuse local
# -------------------------
CACHE_DIR = "./hf_cache"
local_model_path = os.path.join(CACHE_DIR, MODEL_REPO.replace("/", "_"))

if not os.path.exists(local_model_path):
    print(f"Downloading model {MODEL_REPO}...")
    local_model_path = snapshot_download(
        repo_id=MODEL_REPO,
        cache_dir=CACHE_DIR,
        token=HF_TOKEN
    )
else:
    print(f"Using cached model at {local_model_path}")

# -------------------------
# Load tokenizer from model dir
# -------------------------
tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder", token=HF_TOKEN, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer loaded successfully.")

# -------------------------
# Quantization
# -------------------------
quant_cfg = BaseQuantizeConfig(bits=BITS, group_size=GROUP_SIZE, desc_act=DESC_ACT)

model = AutoGPTQForCausalLM.from_pretrained(
    local_model_path,
    quantize_config=quant_cfg,
    device_map="auto",
    offload_folder="./offload",
)

# Prepare example inputs for quantization calibration
example_texts = [
    "def example_function():\n    return 42",
    "print('Hello, world!')"
]

examples = tokenizer(example_texts, padding=True, truncation=True, return_tensors=None)
examples_list = [
    {"input_ids": input_id, "attention_mask": attn_mask}
    for input_id, attn_mask in zip(examples["input_ids"], examples["attention_mask"])
]

print("Quantizing...")
model.quantize(examples=examples_list)

os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_quantized(OUTPUT_DIR, use_safetensors=True)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Quantization complete. Saved to: {OUTPUT_DIR}")
