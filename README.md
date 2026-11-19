# SenseNova Loop: fused mHC Sinkhorn (fork of OpenSenseNova/llama.cpp)

This fork is [OpenSenseNova/llama.cpp](https://github.com/OpenSenseNova/llama.cpp) at `24ea7bb` with one commit that
makes [SenseNova-Loop-9B](https://huggingface.co/sensenova/SenseNova-Loop-9B-GGUF) decode about 3.7x faster on CUDA,
with the same output. Everything below the line is the upstream README, unchanged.

## What was slow

SenseNova Loop runs its 32 layers looped to 56. At each layer, mHC (manifold-constrained hyper-connections) mixes two
residual streams through a 2x2 matrix that 20 iterations of log-domain Sinkhorn make doubly stochastic. Upstream
builds every iteration out of ordinary ggml ops, so one decode step was about 53,000 graph nodes, nearly all of them
tiny kernel launches on a handful of floats. The router matmul also asks for FP32 precision, which the CUDA backend
refused on Ampere and newer (its cuBLAS path uses TF32), so the router ran on the CPU every layer.

## What this fork changes

1. **A fused op, `GGML_OP_MHC_SINKHORN2`** (`ggml_mhc_sinkhorn2(ctx, a, offset, n_iter)`), with a CPU implementation
   and a CUDA kernel (`ggml/src/ggml-cuda/mhc-sinkhorn.cu`) that keeps all 20 iterations in registers, one thread per
   column. It computes exactly what the original graph computes: per column, the pairs (0,1), (2,3), (0,2), (1,3)
   normalised in the log domain with `d = x - y; x = -softplus(-d); y = -softplus(d)`, repeated, then `exp`.
2. **The FP32 router stays on the GPU.** `supports_op` in `ggml-cuda.cu` now accepts an FP32-precision `MUL_MAT`
   when the true-FP32 vector kernel (mmvf) handles it, instead of sending it to the CPU.
3. **`src/models/sensenova-loop.cpp`** uses the fused op in `build_route`.
4. **`tests/test-mhc-sinkhorn2.cpp`** checks the fused op against the original unfused graph on CPU and CUDA.

## Results

RTX 3080 10 GB (PCIe 3.0 x4), SenseNova-Loop-9B Q5_K_M, CUDA 13.0, greedy:

| | Decode | Graph nodes per decode step |
| --- | ---: | ---: |
| Upstream `24ea7bb` | 12.1 tokens/s | ~53,000 |
| This fork | **44.5 tokens/s** | ~7,000 |

Accuracy (`llama-perplexity --kl-divergence`): the fused op matches the unfused graph exactly (KLD 0); the router on
the GPU against the router on the CPU differs by summation order only (mean KLD 0.0014, max 0.053). A closed-form 2x2
solution was tried and dropped (KLD 0.056). At 128K context with a 4-bit KV cache the model still finds a needle at
106K tokens.

## Runtime switches (environment variables, not CMake options)

| Variable | Effect |
| --- | --- |
| `SENSENOVA_MHC_UNFUSED=1` | build the original unfused Sinkhorn graph, for A/B checks |
| `SENSENOVA_MHC_GPU_ROUTER=1` | do not request FP32 precision for the router matmul |
| `GGML_CUDA_F32_PREC_MMVF=0` | restore the old CUDA rule (FP32-precision matmuls go to the CPU) |

None of them is needed: by default you get the fused op and the GPU router.

## Build

As upstream: `cmake -B build -DGGML_CUDA=ON && cmake --build build -j`. Build against one CUDA toolkit only: nvcc 13
headers linked with an older libcudart (for example Ubuntu's CUDA 12 package) break MoE models with "invalid
argument" in `ggml_cuda_mul_mat_q`; pass `-DCUDAToolkit_ROOT=` for the toolkit your nvcc belongs to.

Full list of changed files and the test's build command: [SENSENOVA-MHC-FUSED.md](SENSENOVA-MHC-FUSED.md).

---

# llama.cpp

![llama.cpp](https://raw.githubusercontent.com/ggml-org/llama-brand/refs/heads/master/logo/svg/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v1)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[v1.0.0](https://huggingface.co/blog/ggml-org/llama-cpp-v1) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/pulls?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Asanmai%20OR%20author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Athelittlefireman%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3AmasterFoad%20OR%20author%3Aallozaur%20OR%20author%3AAndreas-Krebbel%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Achaxu01%20OR%20author%3Alhez%20OR%20author%3AAlekseiNikiforovIBM%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming%20OR%20author%3Aynankani)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Description

The main goal of `llama.cpp` is to enable LLM (and MLLM) inference with minimal setup and state-of-the-art performance on a wide range of hardware - locally and in the cloud.

The `llama.cpp` project is built on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Quick start

A few options to get `llama.cpp` installed on your machine:

```bash
# curl
curl -LsSf https://llama.app/install.sh | sh

# powershell
irm https://llama.app/install.ps1 | iex
```

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Llama Chat, the built-in chat app, against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Llama Chat, the built-in chat app, against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [Text-to-speech (TTS) models](docs/multimodal/tts.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
