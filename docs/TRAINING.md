# Training and export workflow

All training and export code lives in two Kaggle notebooks. You do not need a local GPU.

| Notebook | Link |
|---|---|
| Training | https://www.kaggle.com/code/haidarnasirodin/mini-tritty-training |
| Export | https://www.kaggle.com/code/haidarnasirodin/mini-trits-export-models |

## Model

A small Llama-style decoder transformer.

| Setting | Value |
|---|---|
| Parameters | 5,350,272 (ternary version; the fp16 baseline has 5,344,512, the difference being the extra norms) |
| Layers | 6 |
| Width (`dim`) | 256 |
| Heads | 8 (head size 32) |
| MLP | SwiGLU, hidden size 704 |
| Normalization | RMSNorm (computed in fp32) |
| Positions | RoPE, `theta = 10000`, "split-half" pairing |
| Context | 256 tokens |
| Vocabulary | 2048 (byte-level BPE) |
| Embedding | Tied input and output |

### Ternary layers (BitNet b1.58 style)

Every weight matrix in the transformer blocks is a `BitLinear` (42 matrices: 7 per layer). The embedding and the norms stay higher precision.

- **Weights:** `gamma = mean(|W|)`, `W_q = clamp(round(W / gamma), -1, 1)`. One scale per matrix.
- **Activations:** per-token absmax scaling to int8 (`s = 127 / max|x|`).
- **Training:** the optimizer updates latent fp32 weights; the forward pass uses the quantized values and the backward pass uses a straight-through estimator.
- **Extra norms:** an RMSNorm in front of the `wo` and `w_down` projections (12 inner norms in total), which stabilizes ternary training.

## Data and tokenizer

- **Dataset:** [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories). About 2.1M training stories for the long runs; the first 300k stories for quick experiments.
- **Tokenizer:** byte-level BPE trained on the first 300k stories, vocabulary 2048, with `<|endoftext|>` appended after every story. The tokenizer is saved once and then loaded (never retrained), so it always matches saved checkpoints.
- **Token files:** stories are tokenized into flat `uint16` files (`train.bin`, `train_full.bin`, `val.bin`) and sampled at random offsets.
- **Validation set:** the first 5,000 validation stories.

## Training recipe

| Setting | Value |
|---|---|
| Batch | 64 sequences x 256 tokens = 16,384 tokens per step |
| Optimizer | AdamW, betas (0.9, 0.95), weight decay 0.1 (matrices only) |
| Schedule | 200 warmup steps, then cosine decay to `lr / 10` |
| Gradient clipping | 1.0 |
| Precision | fp16 autocast with a gradient scaler |
| Init | normal, std 0.02; residual output projections scaled by `1 / sqrt(2 * layers)` |
| Long runs | 30,000 steps on all data (about 491.5M tokens seen) |
| Hardware | Kaggle GPU (the notebook's time estimates assume a T4) |

`run_experiment` sets the data file, layer type and learning rate itself, checkpoints at every evaluation and **resumes automatically** after a Kaggle disconnect. Do not change `steps` for a run you want to resume, because it changes the schedule.

### Workflow in the training notebook

1. Install, set a seed, optionally restore a backup.
2. Load data, train or load the tokenizer, tokenize to `.bin` files, sanity-check them.
3. Build the model piece by piece (RMSNorm, RoPE, BitLinear, attention, MLP, block), each with a small test.
4. **Smoke tests:** 100 steps for each variant to catch bugs cheaply (a loss stuck near 7.6 or `nan` means a bug).
5. **Short comparison** (300k stories, 6,000 steps): baseline best val loss 1.667, ternary 1.849.
6. Compare sampling settings (temperature and repetition penalty) and learning rates (1e-3 vs 2e-3).
7. **Long runs** on all data, 30,000 steps each, then plot and back up everything.

## Results

| Run | Data | Steps | Best val loss |
|---|---|---|---|
| fp16 baseline | 300k stories | 6,000 | 1.667 |
| Ternary | 300k stories | 6,000 | 1.849 |
| **fp16 baseline** | all stories | 30,000 | **1.4895** |
| **Ternary** | all stories | 30,000 | **1.7033** |

At 6,000 steps the ternary gap is 0.182, and at 30,000 steps it is 0.214, so the gap did not shrink with more training in this setup.

## Export notebook

The export notebook does **not** train anything. It takes `ternary_full_best.pt` and `tokenizer.json` (attach them as a Kaggle Dataset) and:

1. **Tests the model:** validation loss on 40 random validation batches (1.712 in `models/export_results.json`, a different sample from the best-validation figure in the results table), the effect of an int8 embedding table (+0.0002), and the share of ternary weights that are zero (37.4%).
2. **Exports** `bitlm_ternary.bin` (format in [MODEL_FORMAT.md](MODEL_FORMAT.md)).
3. **Cross-checks the file** with `RefModel`, a pure-NumPy engine that reads only the `.bin` and uses integer math and a KV cache, one token at a time. It is compared against PyTorch on 48 real tokens (max and mean logit difference, argmax agreement; the saved run reports 100% argmax agreement, max difference 0.26 and mean 0.026). This is the specification the Rust engine follows.
4. **Generates sample text** with the same sampler the firmware uses.
5. **Saves 16 test vectors** (`test_vectors.bin`) and bundles `bitlm_ternary.bin`, `test_vectors.bin`, `tokenizer.json` and a results JSON into `export_bundle.zip`.

## From notebook output to firmware

1. Download `export_bundle.zip` from the notebook output.
2. Copy `bitlm_ternary.bin` and `test_vectors.bin` into `models/`.
3. Put `tokenizer.json` in `models/` and run `python make_tokenizer.py` to produce `models/tokenizer.bin` (the firmware embeds `tokenizer.bin` and `test_vectors.bin` at build time).
4. Write the model to the `model` partition (`espflash write-bin 0x410000 models/bitlm_ternary.bin`), then `cargo run --release`.
5. At the `esp32>` prompt run `test`. You should see full argmax agreement and a tiny logit difference (about 1e-5).

## Tips

- Run the notebook cells one at a time, not "Run All": the long runs take hours.
- Download the backup zip after every big step. Kaggle sessions end and `/kaggle/working` is cleared.
- If you change the model definition, change it in **both** notebooks and in `engine.rs` / `runner.rs`, or the checkpoint and the exporter will disagree. The export notebook's file reader asserts that no bytes are left unread, which catches most layout mistakes.
