# Changelog

## v2.0 — 2026-09-17

Rebuilt on a maintained base: **BeeLlama v0.4.6** (`78af83265`, 2026-09-08) instead of the
NJannasch May snapshot. Four months of upstream llama.cpp work come with it.

### Fixed

- **Tensor-parallel crash** `GGML_ASSERT(obj_new) failed` at `ggml.c:1808` (issue #3).
  Root cause: in TP mode the meta backend created its ggml context with a hardcoded
  `1024*1024*1024`; externally created views accumulate in that buffer and deplete it, so the
  process died after minutes of generation, always right after a context checkpoint.
  Fixed upstream by ggml-org/llama.cpp#22616 (merged 2026-05-25), with follow-ups #23525
  and #23480 — all of them newer than the v1.0 snapshot.
- **Missing OpenSSL DLLs** (issue #1). v1.0 linked the system OpenSSL and did not ship
  `libssl-4-x64.dll` / `libcrypto-4-x64.dll`, so `llama-server.exe` refused to start on a clean
  machine. v2.0 links BoringSSL statically (`-DLLAMA_BUILD_BORINGSSL=ON`); no external TLS
  libraries are involved any more.
- **Crash on CPUs without AVX-512** (issue #2). v1.0 was built with `GGML_NATIVE=ON`, which
  compiled for the build machine's CPU. v2.0 uses `-DGGML_NATIVE=OFF`,
  `-DGGML_CPU_ALL_VARIANTS=ON` and `-DGGML_BACKEND_DL=ON`: ten CPU backends are shipped
  (sse42, x64, sandybridge, haswell, skylakex, icelake, cascadelake, cannonlake, alderlake)
  and the best supported one is loaded at runtime.
- `--mmproj` with `--spec-type draft-mtp` — fixed upstream (llama.cpp issue #22867, closed
  as completed before the MTP merge).
- `llama-memory-recurrent.cpp:173` assert without MTP — was specific to the old fork, gone.

### Added

- KVarN cache quantization and its CUDA kernels (`-DGGML_CUDA_KVARN=ON`)
- Sparse FlashAttention, concurrent CUDA streams for multi-GPU splits, faster KV-cell history
  lookup, and broader model support inherited from llama.cpp `465e49b9c` (b10830)

### Build configuration

- Source: `Anbeeld/beellama.cpp` tag `v0.4.6`, commit `78af83265`
- CUDA Toolkit 12.8.61, MSVC 19.44.35227, CMake 4.2.1, Ninja
- `-DCMAKE_CUDA_ARCHITECTURES=120a-real` — verified: `cuobjdump` reports 372 ELF sections,
  all `sm_120a`, no other architectures
- `-DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_CUDA_KVARN=ON`
- `-DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=ON -DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON`
- `-DLLAMA_BUILD_BORINGSSL=ON`
- Bundles CUDA 12.8 runtime DLLs (cudart, cublas, cublasLt)

### Verified on

RTX 5060 Ti 16GB, `Qwen3.8-27B-UD-Q3_K_XL.gguf` (12.23 GiB), `-ngl 99`, no speculative decoding:

```
pp256: 928.64 t/s      tg64: 33.29 t/s      build: 78af83265 (11794)
```

The extracted archive runs standalone: `llama-cli --list-devices` detects the GPU, the CUDA
backend loads from the bundled `ggml-cuda.dll`, and the CPU backend is auto-selected
(`ggml-cpu-cascadelake.dll` on the test machine).

Archive size: 668 MB.

### Not tested

Tensor parallelism (`--split-mode tensor`). The upstream fix is present, but this is a
single-GPU machine.

### Note on maintenance

This repository is not actively developed. Users whose driver supports CUDA 13 should prefer
the official builds at https://github.com/Anbeeld/beellama.cpp/releases — these CUDA 12.8
packages exist for older drivers only.


## v1.0 — 2026-06-04 (initial release)

First public build combining MTP + TurboQuant + native sm_120 for Windows.

### Included

- llama.cpp source: NJannasch/llama.cpp `mtp-turboquant` branch, commit `d1cfe5766` (2026-05-14)
- Built with CUDA Toolkit 12.8.61
- Built with MSVC 14.44.35207 (Visual Studio Build Tools 2022 17.14.33)
- Built with `CMAKE_CUDA_ARCHITECTURES=120a-real` (native Blackwell consumer GPU + FP4 tensor cores)
- Built with `GGML_CUDA_FA_ALL_QUANTS=ON` (all KV cache type cross-combinations)
- Built with `GGML_CUDA_FORCE_MMQ=ON` (safe MMQ kernel selection)
- Bundles CUDA 12.8 runtime DLLs (cudart, cublas, cublasLt) for zero-install deployment

### Cache types available

`f16`, `bf16`, `q8_0`, `q4_0`, `q5_0`, `q5_1`, `q4_1`, `iq4_nl`, plus TurboQuant `turbo2`, `turbo3`, `turbo4` and all cross-combinations.

### Speculative decoding

- `--spec-type draft-mtp` (Multi-Token Prediction, for models with MTP heads)
- `--spec-type draft-simple` (standard draft model)
- `--spec-type draft-eagle3` (EAGLE-3)
- `--spec-type ngram-*` (n-gram variants)

### Verified on

- RTX 5060 Ti 16GB (sm_120) with Qwen3.6-27B-UD-IQ3_XXS:
  - Without MTP, turbo3 KV: 31 t/s decode at short ctx
  - With MTP n_max=2, turbo3 KV: 47 t/s decode, loads 256K context in ~15 GB VRAM

### Known issues

- `--mmproj` + `--spec-type draft-mtp` triggers llama.cpp issue #22867 (not specific to this build)
- `--spec-type none` + large `-c` triggers `llama-memory-recurrent.cpp:173` assert (specific to NJannasch fork)
- CUDA 13.x runtime not tested with TurboQuant kernels

### Files included

- `llama-server.exe` — HTTP server with OpenAI-compatible API
- `llama-cli.exe` — interactive CLI
- `llama-bench.exe` — benchmarking tool
- `llama-quantize.exe` — GGUF quantization tool
- `llama-imatrix.exe` — importance matrix calculation
- `llama-perplexity.exe` — perplexity evaluation
- `llama-tokenize.exe` — tokenizer test
- `llama-completion.exe` — one-shot completion
- Other tools from llama.cpp standard set
- `ggml-cuda.dll` — CUDA backend with TurboQuant kernels (~44 MB)
- `ggml-cpu.dll`, `ggml-base.dll`, `ggml.dll`, `llama.dll` — core libraries
- `cudart64_12.dll`, `cublas64_12.dll`, `cublasLt64_12.dll` — CUDA 12.8 runtime
- `mtmd.dll`, `llama-common.dll`, etc.

Total archive size: ~870 MB.

### Future versions

Will rebuild when:
- llama.cpp issue #22867 is fixed (mmproj + MTP compatibility)
- NJannasch rebases on newer upstream master
- CUDA 12.9+ becomes meaningfully different for our target
- Significant TurboQuant kernel improvements appear in any upstream
