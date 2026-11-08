# Mini-Trits — Tiny Ternary Model

A tiny **BitNet b1.58-style language model** (ternary weights `{-1, 0, +1}`, int8 activations) trained from scratch on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) and run **entirely on an ESP32-S3** by an inference engine written in Rust. No PC, no cloud.

You can talk to it two ways:

- **Serial console:** type `gen Once upon a time` and the chip writes the story.
- **Wi-Fi booth mode:** the board creates its own open network (`Mini-Trits`) with a captive portal. A phone joins, a page opens, and everyone connected watches the same story being written live. It runs standalone from a USB charger.

**At a glance**

- 5.35M parameters, 6 layers, dim 256, 8 heads, 2048-token BPE vocabulary, 256-token context
- **1.77 MB** model file (about 6x smaller than the same model in fp16)
- On-device logits match PyTorch: **16/16 argmax agreement**, max logit difference 0.00002
- Hand-written byte-level BPE tokenizer, KV cache, sampler and dual-core integer kernels in Rust (`std` on ESP-IDF, no ML libraries)

> Built as a portfolio project: I'm comfortable with software engineering and was new to ML, so the training notebooks are heavily commented and tested step by step.

## Results

### Model quality (full TinyStories, 30,000 steps)

| Model | Best validation loss |
|---|---|
| fp16 baseline | **1.4895** |
| Ternary (this project) | **1.7033** |

Same architecture, same data, same 30,000 steps (batch 64 x 256 tokens, about 491M tokens seen). The ternary model gives up about 0.21 nats (roughly 14%) in exchange for 2-bit weights and integer arithmetic. About 37% of the ternary weights are zero, and storing the embedding table as int8 costs about 0.0002 in loss.

### On-device (ESP32-S3, `test` command)

| Metric | Value |
|---|---|
| Tokens checked | 16 |
| Time | 2727 ms (**170 ms/token**) |
| Argmax agreement with PyTorch | **16/16** |
| Max \|logit diff\| | 0.00002 |
| Mean \|logit diff\| | 0.000002 |

The `test` run covers the first 16 positions, where attention is cheap. Longer stories are slower per token; the web code comments estimate roughly 4.6 tokens/s during generation. `gen` prints its own measured speed after every story.

### Size

| Format | Size |
|---|---|
| fp32 | about 21.4 MB |
| fp16 | about 10.7 MB |
| **Ternary `.bin` (2-bit blocks + int8 embedding)** | **1,773,288 bytes (1.77 MB)** |

## How it works

```
Kaggle (GPU)                     Kaggle (CPU)                      ESP32-S3
┌──────────────────┐   .pt       ┌──────────────────────┐  .bin    ┌─────────────────────────┐
│ train fp16 and   │ ──────────▶ │ test, pack to 2 bits,│ ───────▶ │ Rust engine reads the   │
│ ternary models   │             │ check NumPy engine,  │          │ file from flash and     │
│ (PyTorch, QAT)   │             │ save test vectors    │          │ generates text          │
└──────────────────┘             └──────────────────────┘          └─────────────────────────┘
```

1. **Train** with quantization-aware training: latent fp32 weights are rounded to ternary (absmean scale) and activations to int8 (per-token absmax) in the forward pass, with a straight-through estimator for gradients.
2. **Export** packs each ternary weight into 2 bits (4 per byte), quantizes the tied embedding table to int8, and writes one `bitlm_ternary.bin`. A pure-NumPy engine that reads only that file is checked against PyTorch, and reference logits are saved as test vectors.
3. **Run** on the ESP32-S3: the model sits in a 2 MB flash partition and is copied once into PSRAM. A Rust engine processes one token at a time with a KV cache. Matrix multiplies accumulate `{-1,0,+1} x int8` in `i32` with a single float rescale, and the two Xtensa cores each take half of every heavy loop.

More detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/MODEL_FORMAT.md](docs/MODEL_FORMAT.md), [docs/TRAINING.md](docs/TRAINING.md).

## Hardware

- **ESP32-S3 N16R8** board (16 MB flash, 8 MB octal PSRAM, 240 MHz)
- Use the USB-C port labeled **COM** for flashing and the serial console

## Quick start

The model, tokenizer and test vectors are committed in `models/`, so you do not need to train anything.

### 1. Prerequisites

- Rust with the ESP toolchain (`rust-toolchain.toml` selects the `esp` channel), installed with [`espup`](https://github.com/esp-rs/espup)
- [`espflash`](https://github.com/esp-rs/espflash) and `ldproxy` (`cargo install espflash ldproxy`)
- The usual ESP-IDF build requirements (Python 3, git, cmake, ninja). `embuild` downloads ESP-IDF v5.5.3 into the project on the first build, which takes a while.

### 2. Clone

```bash
git clone https://github.com/Ingenium-Society/ESP32-Rust-Firmware.git
cd ESP32-Rust-Firmware
```

### 3. Write the model to its flash partition (once)

The firmware reads the weights from a dedicated 2 MB partition named `model` (see `partitions.csv`). `cargo run` flashes the app and the partition table but **not** the model, so write it separately:

```bash
espflash write-bin 0x410000 models/bitlm_ternary.bin
```

Run this again only if you export a new model. (`tokenizer.bin` and `test_vectors.bin` are compiled into the app, so they need no separate step.)

### 4. Build, flash and open the monitor

```bash
cargo run --release
```

This builds the firmware, flashes it with the custom partition table, and opens the serial monitor.

### 5. Talk to it

At the `esp32>` prompt:

```
esp32> test
tokens: 16  time: 2727 ms (170 ms/token)
argmax agreement: 16/16
max |diff|: 0.00002  mean |diff|: 0.000002

esp32> gen Once upon a time, there was a little girl named
```

| Command | What it does |
|---|---|
| `help` | List commands |
| `ping` | Connection check |
| `info` | Device information |
| `model` | Print the model header (vocab, context, layers, ...) |
| `load` | Parse the model and dump some fields (checks the file layout) |
| `mem` | Show free internal RAM and PSRAM |
| `test` | Run the 16 reference tokens and compare logits with PyTorch |
| `gen <prompt>` | Generate a story (empty prompt starts a fresh one) |
| `seed <n>` / `seed random` | Fix the sampling seed to repeat a run exactly, or go back to a fresh random seed |
| `wifi` | Show access point status (starts it if boot-time start failed) |

`gen` streams the text as it is produced, then prints a line such as `[prompt N tokens in X ms | generated N tokens in X ms = X tok/s | context N/256 | seed S]`. Sampling uses temperature 0.7, top-k 40 and a repetition penalty of 1.15 over the last 64 tokens (the same sampler as the export notebook). A story ends at the end-of-text token or when the 256-token context is full. Serial lines are limited to 128 characters.

### 6. Booth mode (Wi-Fi)

At boot the board starts an open access point:

1. Join the Wi-Fi network **`Mini-Trits`** (no password) from a phone.
2. The captive portal should offer to open the page. If not, browse to the address printed in the serial log (`page: http://...`).
3. Type a prompt and press **Ask**. The story streams in live for everyone connected. Up to two further prompts can wait in the queue; after that the page asks you to try again.

The network is open and unauthenticated by design, so use it for demos only.

## Reproducing the model

Everything runs on free Kaggle notebooks (copies are also in this repo):

| Notebook | Purpose |
|---|---|
| [mini-tritty-training](https://www.kaggle.com/code/haidarnasirodin/mini-tritty-training) · [local copy](mini-tritty-training.ipynb) | Tokenizer, data, fp16 baseline and ternary training, comparison plots |
| [mini-trits-export-models](https://www.kaggle.com/code/haidarnasirodin/mini-trits-export-models) · [local copy](mini-tritty-export_models.ipynb) | Tests the trained model, exports `bitlm_ternary.bin`, checks it with a NumPy engine, writes `test_vectors.bin` |

To rebuild the tokenizer file used by the firmware from `models/tokenizer.json`:

```bash
python make_tokenizer.py
```

Full walkthrough: [docs/TRAINING.md](docs/TRAINING.md).

## Project layout

```
.
├── src/
│   ├── main.rs        boot banner; starts the generation task, Wi-Fi AP and console
│   ├── console.rs     serial line editor and read loop
│   ├── commands.rs    serial command dispatch (help, gen, seed, test, mem, ...)
│   ├── job.rs         the single generation task and the shared story all front ends use
│   ├── web.rs         Wi-Fi access point, captive portal DNS, HTTP routes
│   ├── page.html      the phone-facing page (embedded in the firmware)
│   ├── flash.rs       model partition mapping; embedded tokenizer and test vectors
│   ├── engine.rs      zero-copy parser for the BTLM model file
│   ├── runner.rs      forward pass, KV cache, dual-core kernels
│   ├── generate.rs    sampler (temperature, top-k, repetition penalty) and token loop
│   ├── tokenizer.rs   byte-level BPE (GPT-2 style) reading tokenizer.bin
│   └── selfcheck.rs   the `test` command: engine logits vs PyTorch
├── models/
│   ├── bitlm_ternary.bin   packed model (1,773,288 bytes)
│   ├── tokenizer.json      Hugging Face tokenizer (source for tokenizer.bin)
│   ├── tokenizer.bin       compact tokenizer embedded in the firmware
│   ├── test_vectors.bin    16 tokens + PyTorch logits
│   ├── export_results.json export notebook metrics
│   └── export_bundle.zip   everything the export notebook produced
├── docs/                   architecture, file formats, training workflow
├── mini-tritty-training.ipynb / mini-tritty-export_models.ipynb
├── make_tokenizer.py       tokenizer.json -> tokenizer.bin
├── partitions.csv          custom partition table (4 MB app + 2 MB model)
├── sdkconfig.defaults      ESP-IDF settings (PSRAM, caches, 240 MHz, stack size)
├── espflash.toml           flash settings (16 MB, QIO, 80 MHz)
└── .github/workflows/      CI: build, rustfmt, clippy
```

## Known limitations

- The tokenizer's GPT-2 split rule is hand-coded (no regex crate), so rare Unicode edge cases may tokenize differently from the Hugging Face tokenizer.
- The context is 256 tokens and the model is tiny: it writes simple TinyStories-style text, not general-purpose answers.
- Only one story is generated at a time (two 3 MB KV caches would not fit in PSRAM), so other visitors queue.
- The KV cache is `f32`. Quantizing it would cut memory and could speed up attention.

## Acknowledgements

- [TinyStories](https://arxiv.org/abs/2305.07759) dataset (Eldan and Li)
- [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764) (BitNet b1.58)
- The [esp-rs](https://github.com/esp-rs) project

## License

MIT, see [LICENSE](LICENSE). Copyright (c) 2026 Haidar Ali Fawwaz Nasirodin (Dadarzz).
