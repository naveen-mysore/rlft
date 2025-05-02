# activate the same venv you're using for lm‑evaluation‑harne
from transformers import AutoModelForCausalLM, AutoTokenizer
import shutil, os, pathlib, torch

src = "/path/to/ppo_trained_model"
dst = "/path/to/save/model_without_value_head"

# 1. load – HF will happily ignore the unexpected v_head.* keys
model = AutoModelForCausalLM.from_pretrained(
        src,
        trust_remote_code=True,          # in case you used custom code
        torch_dtype="auto", 
        device_map="cpu")               # keep it on CPU while saving

# 2. save only the causal‑LM weights (HF won't write the v_head back)
model.save_pretrained(dst, safe_serialization=True)

# 3. copy the tokenizer & configs
for f in ["tokenizer.json","tokenizer_config.json",
          "special_tokens_map.json","config.json",
          "generation_config.json"]:
    if (pathlib.Path(src)/f).exists():
        shutil.copy(pathlib.Path(src)/f, dst)
