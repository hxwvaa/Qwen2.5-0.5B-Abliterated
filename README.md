# Qwen2.5-0.5B-Abliterated

A weight-space abliterated Qwen2.5-0.5B-Instruct checkpoint, made for an
AI security and red-teaming workshop. Private repository.

## What is in this repo

- `Qwen2.5-0.5B-Abliterated.zip.part-00` … `.part-38` — the complete
  model checkpoint, split because GitHub rejects single files over
  100 MB. Each part is 25 MB except the last (3.3 MB).
- `REASSEMBLE.sh` — joins the parts and unzips the checkpoint.
- `CHECKSUMS.txt` — SHA-256 for the zip and for the model weights.
- This README — the full process, so the result can be understood
  and reproduced without trusting a single claim.

## Reassemble and load

```bash
cat Qwen2.5-0.5B-Abliterated.zip.part-* > Qwen2.5-0.5B-Abliterated.zip
sha256sum Qwen2.5-0.5B-Abliterated.zip
# expected: 82ca983efee4c3be631ab40a4566dd2b4659b742813d7bfe1481749cf7d017c1
unzip Qwen2.5-0.5B-Abliterated.zip
# -> qwen-abliterated-bf16/ with config.json, model.safetensors, tokenizer files
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("qwen-abliterated-bf16")
model = AutoModelForCausalLM.from_pretrained("qwen-abliterated-bf16")
```

## How this was achieved — step by step

### 1. Base model
Qwen/Qwen2.5-0.5B-Instruct, downloaded from Hugging Face.
Architecture: Qwen2ForCausalLM, 24 layers, hidden size 896,
14 attention heads, 2 KV heads, intermediate size 4864,
vocab 151936, tied word embeddings, original dtype bfloat16.
Original `model.safetensors`: 943 MB,
SHA-256 `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`.
Load test passed: "What is the capital of France?" -> "The capital of France is Paris."

Environment: CPU-only Linux VM, 4 cores, 7.7 GiB RAM, no GPU.
Python 3.12, transformers 5.22.1, torch 2.15.0 (CPU).
Two environment problems were hit and fixed, and are part of the record:
system pip ran out of space because /tmp is a 512 MiB tmpfs — fixed by
using a venv under the home disk with TMPDIR there; and the Hugging Face
Hub xet download for the weights stalled at 0 bytes — fixed by a direct
curl download from the Hugging Face resolve URL, whose SHA-256 matched.

### 2. Baseline behaviour (before any edit)
Identical prompts, ChatML template, greedy decoding, 60 tokens.
20 prompts in three sets:
- Normal (5): general QA, writing, arithmetic
- Benign trigger (5): harmless prompts containing trigger words
  (kill a Linux process, bomb calorimeter, shoot a photo, break a window
  in fiction, program execution)
- Refusal trigger (10): aggregate only — refusal boolean and length
  are recorded, full outputs are never saved or distributed

Results, original model:
- Normal: 0/5 refused
- Benign trigger: 0/5 refused
- Refusal trigger: 9/10 refused

The benign set is an honest negative result: this model does not
over-refuse those harmless prompts, so they demonstrate preservation
after the edit, not a change.

### 3. Finding the refusal direction
Method: Arditi et al., "Refusal in Language Models Is Mediated by a
Single Direction" (2024).

- Run the 10 refusal-trigger prompts and the 10 harmless prompts
  (normal + benign) through the original model.
- Take the hidden state at the last prompt token, for every layer
  (output_hidden_states=True).
- Per layer: direction candidate = mean(harmful) - mean(harmless).
- Layer-state difference norms rose with depth (embedding 0.00,
  mid layers ~2-26) and jumped at the final layer state (119.88),
  which was selected as the direction source.
- The direction was normalised to a unit vector in the 896-dim
  residual space.

Caveat, stated plainly: norms grow with depth, so "largest norm" is
not a pure separation metric. The choice is justified by what happened
next — the before/after evaluation — not assumed in advance.

### 4. Weight-space abliteration (the actual edit)
For a unit refusal direction r, every matrix W that writes to the
residual stream was replaced by:

    W' = W - r (r^T W)

so no output of W can contain a component along r.
For the embedding, each token vector e was replaced by
e - (e . r) r.

Edited, 49 matrices total:
- model.embed_tokens.weight (1)
- model.layers[i].self_attn.o_proj.weight (24)
- model.layers[i].mlp.down_proj.weight (24)

lm_head was not edited separately: this checkpoint ties lm_head to
embed_tokens, so editing the embedding already changes it.
This is not prompt engineering, not a system-prompt change, and not
a config flag. The prompt, system message, and generation config are
unchanged; the weights themselves are different.

Measured weight change: total Frobenius norm 8.3186,
embedding alone 4.6525. Per-matrix values are in
abliteration_metadata.json inside the checkpoint.

The first saved checkpoint was float32. It was then converted to
bf16 (the original dtype) for delivery and re-evaluated separately,
so the checkpoint in this repo is the evaluated one, not an
untested conversion.

### 5. Before / after evaluation (same prompts, same settings)

| Set | n | Original refused | Abliterated refused |
|---|---|---|---|
| Normal | 5 | 0 | 0 |
| Benign trigger | 5 | 0 | 0 |
| Refusal trigger (aggregate) | 10 | 9 | 0 |

The bf16 delivery checkpoint gives the identical table:
normal 0/5, benign 0/5, refusal 0/10.

Normal answers stayed sensible (Paris, photosynthesis, haiku,
17 x 6 = 102), with slightly different wording — what a weight edit
should do, not a broken model.

### 6. Fresh-process verification
A separate process reloaded the saved checkpoint from disk alone,
generated "The capital of France is Paris.", compared embedding
weights to the original (Frobenius difference 4.6525, must be > 0),
and recorded SHA-256 checksums for every file. Result: VERIFY OK.

Delivered bf16 weights:
- `model.safetensors`: 943 MB,
  SHA-256 `a6ce36dd0acb45579f773721aec7399818b78ecfa99c3f52c9d6f16eb5014172`
- Zip of the whole checkpoint: 954 MB,
  SHA-256 `82ca983efee4c3be631ab40a4566dd2b4659b742813d7bfe1481749cf7d017c1`

### 7. Limitations — what this does not prove
- Small sample: 20 prompts, single run, keyword refusal detection
  (refusal markers in the first 300 characters), greedy 60-token outputs.
- No HarmBench, JailbreakBench, or StrongREJECT scores are claimed:
  those benchmarks were not run. They are relevant methodology
  references, not results from this experiment.
- Abliteration is blunt: it removes refusal broadly, not surgically
  for one prompt type. That breadth is the workshop point.
- A 0.5B CPU model is a demonstration of a causal weight effect,
  not a claim about general capability, safety, or production use.
- Refusal-trigger outputs are deliberately not included in this repo.

## Workshop demo
The demo script (in the original experiment workspace, scripts/demo.py)
loads the original and abliterated checkpoints one at a time and
prints them side by side for normal + benign prompts only, so it is
safe to show live. Do not demo refusal-trigger prompts live, and do
not paste abliterated outputs for those prompts into slides.

## Reproduce
1. Download Qwen/Qwen2.5-0.5B-Instruct.
2. Run baseline evaluation (20 prompts, greedy, 60 tokens).
3. Collect last-token activations, compute mean difference, normalise.
4. Orthogonalise embed_tokens, o_proj, and down_proj against that direction.
5. Save, reload in a fresh process, re-run the same 20 prompts.
