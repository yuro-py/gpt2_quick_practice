import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
# optional -> from bench import measure_perplexity, measure_generate, print_report

print("Loading tokenizer and reference model...")
#MODEL = "openai-community/gpt2" # USE official model name from huggingface
# tok = AutoTokenizer.from_pretrained(MODEL)
#device = "cuda" if torch.cuda.is_available() else "cpu"

# We load in fp16 because run_baseline.py generated ground_logits.pt in fp16
# hf_model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(device)
# sd = hf_model.state_dict()

#import torch
#import torch.nn.functional as F
#from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "gpt2"
device = "cuda" if torch.cuda.is_available() else "cpu"

hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(device).eval()
tok = AutoTokenizer.from_pretrained(MODEL)

W = hf.state_dict()          # all weights, keyed exactly like weights.json


# ---------- primitives ----------

def linear(x, w, b=None):        # nn.Linear convention: y = x @ w.T + b
    return x @ w.T if b is None else x @ w.T + b

def conv1d(x, w, b):             # HF Conv1D convention: y = x @ w + b
    return x @ w + b

def layernorm(x, g, b, eps=1e-5):
    mu  = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    return (x - mu) / (var + eps).sqrt() * g + b

def gelu(x):
    return F.gelu(x)             # NewGELU ≈ tanh-approx GELU; F.gelu is fine

def softmax(x):
    return F.softmax(x, dim=-1)


# ---------- attention ----------

def attention(x, prefix):
    B, T, C = x.shape

    # fused QKV:  [B,T,768] @ [768,2304] + [2304]
    qkv = conv1d(x, W[f"{prefix}.c_attn.weight"], W[f"{prefix}.c_attn.bias"])
    q, k, v = qkv.split(C, dim=-1)                       # each [B,T,768]

    # reshape to heads: 12 heads × 64 dim
    q = q.view(B, T, 12, 64).transpose(1, 2)             # [B,12,T,64]
    k = k.view(B, T, 12, 64).transpose(1, 2)
    v = v.view(B, T, 12, 64).transpose(1, 2)

    # scaled dot-product + causal mask
    att = (q @ k.transpose(-2, -1)) / (64 ** 0.5)        # [B,12,T,T]
    mask = W[f"{prefix}.bias"][:, :, :T, :T]             # [1,1,T,T] causal
    att = att + mask
    att = softmax(att)

    # weighted sum + merge heads
    y = att @ v                                          # [B,12,T,64]
    y = y.transpose(1, 2).contiguous().view(B, T, C)     # [B,T,768]

    # output projection
    return conv1d(y, W[f"{prefix}.c_proj.weight"], W[f"{prefix}.c_proj.bias"])


# ---------- mlp ----------

def mlp(x, prefix):
    x = conv1d(x, W[f"{prefix}.c_fc.weight"],   W[f"{prefix}.c_fc.bias"])   # 768 -> 3072
    x = gelu(x)
    x = conv1d(x, W[f"{prefix}.c_proj.weight"], W[f"{prefix}.c_proj.bias"]) # 3072 -> 768
    return x


# ---------- transformer block ----------

def block(x, n):
    p = f"transformer.h.{n}"
    x = x + attention(layernorm(x, W[f"{p}.ln_1.weight"], W[f"{p}.ln_1.bias"]), f"{p}.attn")
    x = x + mlp(      layernorm(x, W[f"{p}.ln_2.weight"], W[f"{p}.ln_2.bias"]), f"{p}.mlp")
    return x


# ---------- full forward pass ----------

@torch.no_grad()
def gpt2_forward(input_ids):
    B, T = input_ids.shape

    # embeddings
    x = W["transformer.wte.weight"][input_ids]                     # [B,T,768]
    x = x + W["transformer.wpe.weight"][:T]                        # [T,768] broadcast

    # 12 transformer blocks
    for n in range(12):
        x = block(x, n)

    # final layernorm
    x = layernorm(x, W["transformer.ln_f.weight"], W["transformer.ln_f.bias"])

    # LM head (weight-tied to wte)
    logits = x @ W["transformer.wte.weight"].T                     # [B,T,50257]
    return logits


# ---------- sanity check vs HF ----------

ids = tok("Hello, I'm a language model,", return_tensors="pt").input_ids.to(device)
mine = gpt2_forward(ids)
theirs = hf(ids).logits
print("max abs diff:", (mine - theirs).abs().max().item())
