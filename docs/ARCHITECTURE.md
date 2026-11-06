# Firmware architecture

The firmware is a single Rust binary (`esp-rs-sf`) on `esp-idf-svc` (ESP-IDF v5.5.3 with `std`, target `xtensa-esp32s3-espidf`). It runs the model with hand-written kernels and offers two front ends: a serial console and a Wi-Fi page.

```
src/
├── main.rs        boot banner; starts job task, Wi-Fi AP, then the console
├── console.rs     serial line editor and read loop
├── commands.rs    serial command dispatch
├── job.rs         the one generation task + the shared story ("view")
├── web.rs         SoftAP, captive-portal DNS, HTTP routes
├── page.html      phone page, compiled into the firmware
├── flash.rs       model partition mapping; embedded tokenizer and test vectors
├── engine.rs      BTLM parser: zero-copy views over the model bytes
├── runner.rs      forward pass, KV cache, dual-core kernels
├── generate.rs    sampler and generation loop
├── tokenizer.rs   BTOK parser, GPT-2-style pre-tokenizer and BPE
└── selfcheck.rs   `test` command: engine logits vs PyTorch
```

## Boot sequence

1. `main` prints a banner and calls `job::start()`, which spawns the generation task. That task maps the model, copies it into PSRAM, parses it, parses the tokenizer and allocates the `Runner` (with its KV cache) **before** it accepts jobs, so the first prompt is not the slow one.
2. `web::start()` brings up the access point, HTTP server and DNS responder. If it fails, the error is printed and the serial console still works.
3. `console::run()` takes over the main task for the serial shell.

## One generation task, two front ends

Only one `Runner` can exist: the model copy (2 MB) plus one KV cache (3 MB) already uses 5 of the 8 MB of PSRAM, and a second runner would fill it. So all work goes through a single thread in `job.rs`:

```
 serial `gen` ──┐                                    ┌─▶ mpsc reply channel (serial streams text from it)
                ├─▶ bounded queue (depth 2) ─▶ task ─┤
 HTTP POST /gen ┘     (also `test`)             │    └─▶ shared VIEW (id, prompt, text, generating, queued)
                                                │                         ▲
                                                ▼                         │ snapshot()
                                           Runner::step            HTTP GET /state (polled by page)
```

- `submit_generate` counts the job in `queued`, then `try_send`s it. A full queue returns a "busy" error (HTTP 503 on the web, a message on serial).
- The task appends every produced chunk of text to the **shared view**, so every phone sees the same story. The `test` command is queued through the same channel and borrows the same `Runner`.
- Each job carries its own sampling seed: fresh from the hardware RNG by default, or fixed with the serial `seed <n>` command so a run can be repeated token for token.

## Memory

| What | Where | Size |
|---|---|---|
| Model weights | Flash partition `model`, copied once to PSRAM | 1.77 MB file in a 2 MiB buffer |
| KV cache (K and V, `f32`) | PSRAM | 2 x 1.5 MiB = 3 MiB (`6 layers x 256 positions x 256 dim x 4 B` each) |
| RoPE tables | heap | 256 x 16 x 2 floats |
| Activation scratch buffers | internal RAM (small allocations) | a few KB |
| Tokenizer and test vectors | `include_bytes!` in the app image | 21.8 KB and 131 KB |

`sdkconfig.defaults` enables octal PSRAM at 80 MHz with `CONFIG_SPIRAM_USE_MALLOC` and `CONFIG_SPIRAM_MALLOC_ALWAYSINTERNAL=16384`, so allocations above 16 KB (model copy, KV cache) land in PSRAM while small buffers stay in fast internal RAM. The serial `mem` command prints free internal RAM and PSRAM.

Other relevant settings: 240 MHz CPU, 32 KB instruction cache, 64 KB data cache with 64-byte lines, QIO flash at 80 MHz, and a 16 KB main task stack. The generation task has a 32 KB stack and the worker thread 8 KB.

### Partition table (`partitions.csv`)

| Name | Type | Offset | Size |
|---|---|---|---|
| nvs | data | 0x9000 | 0x6000 |
| phy_init | data | 0xf000 | 0x1000 |
| factory | app | 0x10000 | 4 MB |
| model | data | 0x410000 | 2 MB |

`.cargo/config.toml` sets the cargo runner to `espflash flash --monitor --partition-table partitions.csv`, so `cargo run` flashes this table. The `model` partition is not written by `cargo run`; write the weights with `espflash write-bin 0x410000 models/bitlm_ternary.bin`.

## flash.rs

`model()` finds the `model` data partition, memory-maps it, copies it into a `u32`-aligned buffer (so the in-place `f32` views stay aligned), leaks that buffer to `'static` and caches it in a `OnceLock`. `TOKENIZER_BIN` and `TEST_VECTORS` are `include_bytes!` of the files in `models/`, so they must exist at build time.

## engine.rs: zero-copy model view

`Model::parse(&[u8])` validates the magic, version and dimensions, then walks the file with a bounds-checked `Reader`. Norm vectors become `&[f32]`, embedding values `&[i8]`, and each ternary matrix a `Tern { gamma, packed, rows, cols }`. Nothing is copied or unpacked, so the weights stay in their 2-bit form until used. `f32` data is reinterpreted in place and a misaligned section is rejected.

## runner.rs: forward pass

`Runner::step(token, pos)` processes **one token at a time** and returns the logits for the next token. It mirrors `RefModel` from the export notebook:

1. Embedding lookup: `int8 row * per-row scale`.
2. For each layer: RMSNorm, quantize, `wq`/`wk`/`wv`; RoPE (split-half pairing) on q and k; store k and v in the cache; causal attention over positions `0..=pos` with a stable softmax; RMSNorm (`wo_norm`), quantize, `wo`, residual add; RMSNorm (`mlp_norm`), quantize, `w_gate` and `w_up`, `silu(gate) * up`, RMSNorm (`down_norm`), quantize, `w_down`, residual add.
3. Final RMSNorm, then the tied output layer.

### The ternary matrix-vector kernel

`matvec_rows` multiplies packed 2-bit weights by int8 activations. Each byte holds four weights decoded as `((b >> shift) & 3) - 1`; products accumulate in `i32`, so the inner loop has no float math, and one multiply by `gamma / s` rescales each output.

`quantize` uses the per-token absmax scale. Rounding goes through `round_even` (add and subtract `1.5 * 2^23`), which matches NumPy and PyTorch's round-half-to-even without libm. The output layer multiplies the int8 embedding rows by the unquantized final hidden state, as in the reference engine.

### Using both cores

Every heavy loop is split in two halves that run at once: matrix multiplies (by output rows), attention (first and second half of the heads) and the output layer (first and second half of the vocabulary).

- One long-lived worker thread is pinned to **core 1**; the generation task runs on the other core. Spawning a thread per multiply ran the chip out of memory, so jobs are sent to the single worker over a channel.
- `par2(a, b)` runs `a` on the calling core and `b` on the worker, then waits for both.
- Each output value uses exactly the same arithmetic as a single-core version, so results do not depend on the split.
- `par2` uses one `unsafe` lifetime extension so the worker can borrow stack data. It is sound because `par2` does not return before the worker reports completion, and the firmware aborts on panic rather than unwinding.

## tokenizer.rs

A byte-level BPE tokenizer reading `tokenizer.bin`. There is no regex dependency: the GPT-2 split rule (contractions, optional leading space plus letters, digits or symbols, whitespace runs) is hand-coded in `next_piece`. Each piece is mapped to byte tokens and merged by repeatedly applying the lowest-rank pair from a `HashMap`. Decoding concatenates token byte slices.

## generate.rs

- Feeds the prompt through the model and keeps the logits after the last prompt token. An empty prompt starts from the end-of-text token. A prompt must be shorter than the 256-token context.
- Sampler, identical to `sample_next` in the export notebook: repetition penalty over the distinct tokens of the last 64 positions (positive logits divided, negative multiplied by 1.15), temperature 0.7, top-k 40 with a softmax over the survivors, then a draw from a small `xorshift64*` generator.
- There is no cap on new tokens: a story stops at the end-of-text token or when the context is full.
- Output goes through a sink callback (the serial reply channel and the shared view), and incomplete UTF-8 sequences are held back until complete.
- The loop sleeps 10 ms after each step so FreeRTOS can run the HTTP server and Wi-Fi tasks. This costs a little throughput compared with the raw `test` timing.
- The returned `Stats` (prompt and generation time, tokens, context use, seed) feed the serial summary line.

## web.rs and page.html: booth mode

- **Access point:** open network `Mini-Trits` on channel 1, up to 4 clients, no password.
- **Captive portal:** a small UDP DNS responder (port 53) answers every A query with the AP's own address, and the HTTP server redirects every unknown `GET` to the page. This makes phones offer "sign in to network" automatically.
- **Routes:**
  - `GET /` serves `page.html`.
  - `POST /gen` reads the prompt (up to 200 bytes), queues it and returns `202 queued` at once, or `503` if the queue is full. The story is *not* sent on this connection.
  - `GET /state` returns JSON `{id, generating, queued, full, prompt, text}`.
- The page polls `/state` every 300 ms and redraws when the story id or text changes, so all visitors share one screen and no socket is tied up while the model is writing.
- The HTTP server is leaked with `mem::forget`, because it stops when dropped.

## console.rs and commands.rs

ESP-IDF's console stdin is non-blocking and does not echo, so `console::run` reads one byte at a time, echoes printable ASCII, and handles Enter and Backspace itself (128-character lines). If stdin reports a real error (no USB host when running headless off a charger), it reports it once and backs off to avoid spamming. `commands::run` dispatches `help`, `ping`, `info`, `model`, `load`, `mem`, `test`, `gen`, `seed`, `wifi` and `exit`.

## selfcheck.rs

The `test` command feeds the 16 compiled-in reference tokens through the engine at positions 0 to 15 and compares every logit against the PyTorch values from `test_vectors.bin`, reporting time per token, argmax agreement, and max and mean absolute difference.

## Possible next steps

- Quantize the KV cache (int8) to cut its 3 MiB footprint and speed up attention.
- Use the Xtensa SIMD instructions of the ESP32-S3 for the int8 dot products.
- Decode four weights at a time with a lookup table, or skip zero weights.
- Add a password or rate limit to the access point for unattended deployments.
