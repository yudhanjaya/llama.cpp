# Fused mHC Sinkhorn for SenseNova Loop

This branch is [OpenSenseNova/llama.cpp](https://github.com/OpenSenseNova/llama.cpp) at
`24ea7bb` ("llama : add SenseNova Loop MTP support") plus one commit that makes SenseNova-Loop-9B decode
about 3.7x faster on CUDA without changing its output.

## The problem

SenseNova Loop runs its 32 layers looped to 56. At every layer, mHC (manifold-constrained hyper-connections)
mixes two residual streams through a 2x2 matrix that is made doubly stochastic by 20 iterations of Sinkhorn
normalisation in the log domain. The upstream graph builds each iteration from ordinary ggml ops (views,
softplus, adds, exp), so one decode step created about 53,000 graph nodes, most of them tiny kernel launches
on tensors of a few floats. The router matmul is also forced to FP32 precision, which the CUDA backend refused
on Ampere and newer, so it ran on the CPU and copied data back and forth every layer.

On an RTX 3080, SenseNova-Loop-9B Q5_K_M decoded at **12.1 tokens/s**.

## The changes

| File | Change |
| --- | --- |
| `ggml/include/ggml.h` | New op `GGML_OP_MHC_SINKHORN2` and `ggml_mhc_sinkhorn2(ctx, a, offset, n_iter)` |
| `ggml/src/ggml.c` | The op's constructor (output `[4, ne1, ne2, ne3]`, F32), name and symbol tables, op count |
| `ggml/src/ggml-cpu/ops.cpp`, `ops.h`, `ggml-cpu.c` | CPU implementation and dispatch, threaded over columns |
| `ggml/src/ggml-cuda/mhc-sinkhorn.cu`, `.cuh` (new) | CUDA kernel: one thread per column, all iterations in registers |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | Dispatches the op. `supports_op` now lets an FP32-precision `MUL_MAT` stay on the GPU when the true-FP32 vector kernel (mmvf) handles it, instead of falling back to the CPU because cuBLAS would use TF32 |
| `src/models/sensenova-loop.cpp` | `build_route` uses the fused op for the Sinkhorn step |
| `tests/test-mhc-sinkhorn2.cpp` (new) | Parity test: the fused op against the original unfused graph, on CPU and CUDA |

The fused op computes exactly what the original graph does. For each column it normalises the pairs
(0,1), (2,3), (0,2), (1,3) in the log domain with `d = x - y; x = -softplus(-d); y = -softplus(d)`, where
`softplus(x) = x > 20 ? x : log1p(exp(x))`, repeats this `n_iter` times, then applies `exp`.

### Switches (environment variables)

| Variable | Effect |
| --- | --- |
| `SENSENOVA_MHC_UNFUSED=1` | Build the original unfused graph (for comparison) |
| `SENSENOVA_MHC_GPU_ROUTER=1` | Do not request FP32 precision for the router matmul |
| `GGML_CUDA_F32_PREC_MMVF=0` | Restore the old CUDA rule (FP32-precision matmuls go to the CPU) |

## Results

RTX 3080 10 GB (PCIe 3.0 x4), SenseNova-Loop-9B Q5_K_M, CUDA 13.0, greedy decoding:

| | Decode |
| --- | --- |
| Upstream `24ea7bb` | 12.1 tokens/s |
| This branch | **44.5 tokens/s** |

Graph size per decode step went from about 53,000 nodes to about 7,000.

Accuracy, measured with `llama-perplexity --kl-divergence`:

- Fused op against the unfused graph: identical (KLD 0).
- Router on the GPU through mmvf against the router on the CPU: mean KLD 0.0014, max 0.053. This is FP32
  summation order, not a precision loss.
- A closed-form 2x2 solution was tried and rejected (KLD 0.056).

## Building and testing

The build is unchanged: `cmake -B build -DGGML_CUDA=ON && cmake --build build -j`.

The parity test is not registered with CTest. Build it against the libraries of a build:

```sh
g++ -O2 -std=c++17 tests/test-mhc-sinkhorn2.cpp -Iggml/include -Lbuild/bin \
    -lggml -lggml-base -lggml-cpu -lggml-cuda -Wl,-rpath,$PWD/build/bin -o build/bin/test-mhc-sinkhorn2
build/bin/test-mhc-sinkhorn2
```

Note: build with a single CUDA toolkit. Compiling with nvcc 13 while CMake links an older libcudart (for
example Ubuntu's CUDA 12 package) breaks MoE models with "invalid argument" in `ggml_cuda_mul_mat_q`. Pass
`-DCUDAToolkit_ROOT=/usr/local/cuda-13.0` (or whichever toolkit nvcc belongs to).
