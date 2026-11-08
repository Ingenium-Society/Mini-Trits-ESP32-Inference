# File formats

Three small binary formats connect the training side and the firmware. All integers and floats are **little-endian**, which is also the native byte order of the ESP32-S3, so the firmware reads most data in place without copying or converting.

| File | Written by | Read by | Purpose |
|---|---|---|---|
| `bitlm_ternary.bin` (BTLM) | export notebook, section 5 | `src/engine.rs` | Model weights |
| `tokenizer.bin` (BTOK) | `make_tokenizer.py` | `src/tokenizer.rs` | Byte-level BPE tokenizer |
| `test_vectors.bin` | export notebook, sections 6-7 | `src/selfcheck.rs` | PyTorch reference logits |

## 1. Model file: BTLM (version 1)

Every section is 4-byte aligned. This matters because the Rust reader reinterprets `f32` data in place and rejects a misaligned section.

### Header (64 bytes)

| Offset | Type | Field | Value for this model |
|---|---|---|---|
| 0 | `[u8; 4]` | magic `BTLM` | |
| 4 | u32 | version | 1 |
| 8 | u32 | vocab | 2048 |
| 12 | u32 | context length | 256 |
| 16 | u32 | layers | 6 |
| 20 | u32 | heads | 8 |
| 24 | u32 | dim | 256 |
| 28 | u32 | hidden_dim (MLP) | 704 |
| 32 | f32 | rope_theta | 10000.0 |
| 36 | f32 | norm_eps | 1e-5 |
| 40 | | zero padding up to byte 64 | |

### Body (in order, starting at byte 64)

| Section | Contents | Bytes |
|---|---|---|
| Embedding scales | `f32[vocab]`, one scale per token row | 8,192 |
| Embedding values | `i8[vocab * dim]`, row-major | 524,288 |
| Final norm | `f32[dim]` | 1,024 |
| Layer 0 ... layer 5 | see below | 206,620 each |

The embedding table is **tied**: it is used both to look up input tokens and as the output (logit) matrix.

### One layer

| Order | Contents | Size |
|---|---|---|
| 1 | `attn_norm` f32[dim] | 1,024 |
| 2 | `wo_norm` f32[dim] | 1,024 |
| 3 | `mlp_norm` f32[dim] | 1,024 |
| 4 | `down_norm` f32[hidden_dim] | 2,816 |
| 5 | `wq` ternary matrix (dim x dim) | 16,388 |
| 6 | `wk` ternary matrix (dim x dim) | 16,388 |
| 7 | `wv` ternary matrix (dim x dim) | 16,388 |
| 8 | `wo` ternary matrix (dim x dim) | 16,388 |
| 9 | `w_gate` ternary matrix (hidden x dim) | 45,060 |
| 10 | `w_up` ternary matrix (hidden x dim) | 45,060 |
| 11 | `w_down` ternary matrix (dim x hidden) | 45,060 |

`wo_norm` and `down_norm` are the extra RMSNorms placed in front of the `wo` and `w_down` projections (a BitNet-style stabilizer). The 2 inner norms per layer give 12 in total.

### A ternary matrix (rows x cols)

```
f32 gamma                      4 bytes   per-matrix scale (mean absolute value of the latent weights)
u8  packed[rows * cols / 4]    2 bits per weight, row by row
```

**Packing.** Each weight is stored as the 2-bit code `w + 1`:

| Weight | Code |
|---|---|
| -1 | `0b00` |
| 0 | `0b01` |
| +1 | `0b10` |

Four weights share a byte, and **the first weight sits in the lowest two bits**:

```
byte = c0 | (c1 << 2) | (c2 << 4) | (c3 << 6)
```

Rows are stored consecutively (output index major), so row `r` starts at byte `r * cols / 4`. `cols` must be a multiple of 4.

**Decoding one byte** (this is what the Rust kernel does):

```rust
let w0 = (b & 3) as i32 - 1;
let w1 = ((b >> 2) & 3) as i32 - 1;
let w2 = ((b >> 4) & 3) as i32 - 1;
let w3 = (b >> 6) as i32 - 1;
```

### Computing one output value

```
s     = 127 / max(|x|, 1e-5)            per-token activation scale
xq    = clamp(round_half_even(x * s), -128, 127)   int8 activations
acc   = sum_j  w[r][j] * xq[j]           i32 accumulation
out_r = acc * (gamma / s)                one float rescale
```

### Size check

```
64 (header) + 532,480 (embedding) + 1,024 (final norm) + 6 * 206,620 (layers) = 1,773,288 bytes
```

The firmware's `load` command prints the parsed size and it should equal this number. The `model` partition is 2,097,152 bytes, leaving about 316 KB of headroom.

## 2. Tokenizer file: BTOK

Produced by `make_tokenizer.py` from a Hugging Face byte-level BPE `tokenizer.json`.

| Section | Contents |
|---|---|
| Header (16 bytes) | `BTOK`, u32 vocab, u32 n_merges, u32 eos_id |
| Byte table | `u16[256]`: byte value -> token id of that single byte |
| Tokens | for each of `vocab` tokens: `u8 length`, then the raw bytes (max 255) |
| Merges | for each merge: `u16 left`, `u16 right`, `u16 merged`. The merge **rank is the position** in this list |

Token strings are stored as **raw bytes**, already converted from GPT-2's printable-character alphabet, so the firmware can print generated tokens directly. The special end-of-text token is stored as its literal text. The script asserts that the vocabulary has contiguous ids, that every merge result exists in the vocabulary, and that every single byte has a token.

Encoding in `src/tokenizer.rs`: split the text with a hand-coded version of the GPT-2 pre-tokenization rule, map each piece to byte tokens, then repeatedly apply the lowest-rank available merge.

## 3. Test vectors: `test_vectors.bin`

| Offset | Type | Contents |
|---|---|---|
| 0 | u32 | n (16) |
| 4 | u32 | vocab (2048) |
| 8 | `i32[n]` | token ids |
| 8 + 4n | `f32[n][vocab]` | PyTorch logits after each token |

Total size is `8 + 4n + 4n*vocab` bytes (131,144 for n = 16). Feed the ids to the model one at a time, in order, at positions 0 to n-1, and compare the logits after each token. The tokens are the first 16 of the validation data. The reference logits come from the fp32 PyTorch model **with the int8-quantized embedding**, so they match what the packed file can represent. Both files are compiled into the firmware with `include_bytes!`.
