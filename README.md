# llama.cpp + MTP + TurboQuant — Windows Prebuilt for RTX 50-series (Blackwell sm_120)

**A Windows x64 build of llama.cpp with Multi-Token Prediction (MTP) and TurboQuant KV cache compression, compiled natively for sm_120 (consumer Blackwell) with FP4 tensor cores.**

---

## Status, September 2026 — please read first

This repository is **not actively developed**. It was published in June 2026 to fill a packaging gap, and that gap has since been filled by the people doing the actual engineering.

**If your driver supports CUDA 13, use the official builds instead:**

➡️ **[Anbeeld/beellama.cpp releases](https://github.com/Anbeeld/beellama.cpp/releases)** — `beellama-*-bin-win-cuda-13.3-x64.zip`

BeeLlama is a maintained llama.cpp fork that already ships everything this repository was built for — MTP, TurboQuant KV (`turbo2/3/4`), KVarN cache quantization, native `120a-real` Blackwell kernels — on a current llama.cpp base, with regular releases. It is what we use ourselves for daily work.

**This repository still has one reason to exist:** the builds here use the **CUDA 12.8** runtime, so they run on drivers too old for the CUDA 13 packages. If that is you, v2.0 below is current, tested, and fixes every known issue of v1.0.

---

## v2.0 (September 2026)

Built from **BeeLlama v0.4.6** (`78af83265`), CUDA 12.8, MSVC 19.44.

**Fixes over v1.0:**

| Issue | Fix |
|---|---|
| Crash under tensor parallelism: `GGML_ASSERT(obj_new)` at `ggml.c:1808` ([#3](../../issues/3)) | Upstream [ggml-org/llama.cpp#22616](https://github.com/ggml-org/llama.cpp/pull/22616) plus follow-ups #23525 and #23480. v1.0 was a May 14 snapshot — 11 days older than the fix. |
| Missing `libssl-4-x64.dll` / `libcrypto-4-x64.dll` ([#1](../../issues/1)) | TLS is now **statically linked BoringSSL** (`-DLLAMA_BUILD_BORINGSSL=ON`). No external OpenSSL DLLs, nothing to install. **Apologies** — v1.0 linked the system OpenSSL and shipped without it, which broke first launch for everyone who did not already have OpenSSL installed. |
| Crash on CPUs without AVX-512, e.g. Intel Core i ([#2](../../issues/2)) | Built with `-DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=ON -DGGML_BACKEND_DL=ON`. Ten CPU backends ship in the zip (SSE4.2 through Alder Lake) and the right one is picked at runtime. **Apologies** — v1.0 was compiled with `GGML_NATIVE=ON`, which baked in the build machine's AVX-512 instructions. |
| `--mmproj` (vision) unusable together with `--spec-type draft-mtp` | Fixed upstream before the MTP merge ([issue #22867](https://github.com/ggml-org/llama.cpp/issues/22867), closed as completed). |
| `llama-memory-recurrent.cpp:173` assert on long context without MTP | Gone — a fork-specific bug of the old snapshot. |

**Also new via v0.4.6:** KVarN cache quantization, sparse FlashAttention, concurrent CUDA streams for multi-GPU splits, and four months of upstream llama.cpp work.

## Hardware support

Native `sm_120a` build — the zip contains **only** Blackwell kernels, which is why `ggml-cuda.dll` is half the size of v1.0.

| GPU | VRAM | Tested |
|---|---|---|
| RTX 5060 Ti 16GB | 16 GB | ✅ primary test platform |
| RTX 5060 Ti 8GB | 8 GB | — should work, low VRAM |
| RTX 5070 | 12 GB | — should work |
| RTX 5070 Ti | 16 GB | — should work |
| RTX 5080 | 16 GB | — should work |
| RTX 5090 | 32 GB | — should work, plenty of headroom |

Older NVIDIA cards (Ampere, Hopper, Ada) are **not** in this build. Use upstream or BeeLlama prebuilts.

## What's included

- All llama.cpp tools (`llama-server`, `llama-cli`, `llama-bench`, `llama-quantize`, `llama-mtmd-cli`, …)
- `ggml-cuda.dll` with TurboQuant (`turbo2/turbo3/turbo4`) and KVarN kernels
- Ten CPU backend variants, selected at runtime
- CUDA 12.8 runtime DLLs (cudart, cublas, cublasLt) — self-contained, no toolkit install
- Statically linked BoringSSL — no OpenSSL DLLs needed

## Software prerequisites

- Windows 10 / 11 x64
- An NVIDIA driver new enough for the **CUDA 12.8** runtime. That is the point of these builds: they run on drivers the official CUDA 13 packages reject.
- Nothing else. Extract and run.

## Quick start

1. Download **all five parts** of the zip from [releases](../../releases) — GitHub's upload endpoint refused the 668 MB archive as a single file, so it is published split
2. Put them in one folder and join them:

   ```cmd
   copy /b llama-...zip.part00 + llama-...zip.part01 + llama-...zip.part02 + llama-...zip.part03 + llama-...zip.part04 llama-...zip
   ```

   (Git Bash / WSL: `cat llama-...zip.part0* > llama-...zip`. The release page lists the full command and the SHA-256 of every part.)
3. Extract anywhere
4. Download a GGUF model
5. Run from the extracted folder:

```cmd
.\llama-server.exe ^
  -m path\to\model.gguf ^
  --ctx-size 262144 ^
  --n-gpu-layers 999 ^
  --flash-attn on ^
  --cache-type-k turbo3 --cache-type-v turbo3 ^
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.75 ^
  --host 127.0.0.1 --port 8080
```

MTP needs a model with an MTP head — for example Unsloth's `Qwen3.8-27B-GGUF` or `Qwen3.6-27B-MTP-GGUF`. Without one, drop the `--spec-type` line.

## Measured on RTX 5060 Ti 16GB

v2.0, `Qwen3.8-27B-UD-Q3_K_XL.gguf` (12.23 GiB), all layers on GPU, no speculative decoding:

```
pp256: 928.6 t/s      tg64: 33.3 t/s
build: 78af83265 (11794)
```

v1.0 numbers on `Qwen3.6-27B-UD-IQ3_XXS.gguf`, kept for reference:

| Configuration | Prompt t/s | Decode t/s | Max context |
|---|---|---|---|
| Upstream b9495 prebuilt (no TurboQuant, no MTP) | 1032 | 34 | 128K (q8_0 KV) |
| Upstream b9495 + MTP (no TurboQuant) | 109 | 54 | 128K |
| v1.0 (turbo3 + MTP n_max=2) | 97 | 47 | 256K |
| v1.0 (turbo3, no MTP, llama-bench) | 918 | 31 | 256K |

## Cache type options

Recommended combinations for a head_dim=128 27B model on 16 GB VRAM:

| `-ctk` / `-ctv` | Compression vs f16 | Quality loss | Typical use |
|---|---|---|---|
| `q8_0` / `q8_0` | 2× | minimal | Default for ≤128K context |
| `turbo4` / `turbo4` | 3.8× | very small | Best decode speed of the TurboQuant variants |
| `turbo3` / `turbo3` | 4.9× | small | Maximum context (256K on 16GB) |
| `turbo2` / `turbo2` | 6.4× | noticeable | Extreme compression, quality-sensitive |
| `turbo4` / `turbo3` | hybrid | small | K precision, V compression |

v0.4.6 also offers KVarN cache modes; see the BeeLlama documentation for those.

## MTP tuning

`--spec-draft-n-max` controls how many tokens are drafted per cycle:

- `n_max=2` — best when paired with `turbo3` KV (the dequant cost on the acceptance check is heavy)
- `n_max=5` — best with `q8_0` KV (cheap dequant, more acceptance opportunity)
- `n_max=8+` — only with a strong draft head; can be net-negative at a low acceptance rate

`--spec-draft-p-min 0.75` is a sensible default.

## Remaining caveats

### Quality degradation on 3-bit quants at long context

Independent research ([arXiv 2505.02214](https://arxiv.org/abs/2505.02214), "An Empirical Study of Qwen3 Quantization") reports that at 3 bits most of the original model's advantages are lost, and that even 4-bit methods lose substantially on long-context inputs. Running a 3-bit quant at 200K+ for retrieval-heavy work will miss things. Prefer a 4-bit quant at 128K when quality matters more than context length.

### VRAM is tight at 256K

256K context + turbo3 KV + MTP draft cache + compute buffers is roughly 14.7 GB on a 27B 3-bit quant, leaving about 400 MB of headroom on a 16 GB card. If prefill runs out of memory, drop `--ubatch-size` to 256 or reduce context to 192K.

### Tensor parallelism is untested here

The TP crash v1.0 had is fixed upstream and that fix is in v2.0, but we have a single GPU and cannot exercise `--split-mode tensor` ourselves. Reports welcome.

## Why this build existed

In June 2026 the pieces were scattered:

- **upstream llama.cpp** — MTP (PR #22673, merged May 2026), but no TurboQuant; and its widely used Windows CUDA 12.4 package cannot contain Blackwell kernels at all, since sm_120 requires CUDA 12.8 or newer (the CUDA 13 package does contain them)
- **TheTom/llama-cpp-turboquant** — TurboQuant, but `tqp-v0.1.1` (April 2026) predates the MTP merge, ships a CUDA 12.4 build, and has `FORCE_CUBLAS=ON` stuck in its CMake cache (19 t/s decode on our card)
- **AmesianX/TurboQuant** — Windows sm_120 binaries, but no MTP
- **NJannasch/llama.cpp** `mtp-turboquant` — both MTP and TurboQuant, but source only, and that branch has not moved since May 14, 2026

v1.0 was that source compiled with the right flags and the CUDA runtime bundled. v2.0 is the same service on a maintained base — and, as said at the top, for most people the official BeeLlama build is now the better answer.

## License and credits

MIT-licensed work from several upstream projects; see `LICENSE`.

- **ggml-org/llama.cpp** — upstream inference engine
- **Anbeeld/beellama.cpp** — the maintained fork v2.0 is built from (MTP, TurboQuant, KVarN, DFlash)
- **am17an/llama.cpp** `mtp-clean` — MTP integration (PR #22673)
- **TheTom/llama-cpp-turboquant** — original TurboQuant kernel port
- **NJannasch/llama.cpp** `mtp-turboquant` — the MTP + TurboQuant fork v1.0 was built from
- **Google DeepMind** — TurboQuant algorithm (Zandieh et al., ICLR 2026)

Built by an independent contributor as a community service. **AS-IS, no warranty, no support guarantees.**

## Building from source

See `BUILD-NOTES.md`.
