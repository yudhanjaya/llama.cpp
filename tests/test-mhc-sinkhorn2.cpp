// Parity check for ggml_mhc_sinkhorn2 (local SenseNova patch): fused op vs the sub/neg/softplus/exp graph it
// replaces, on the CPU backend and on CUDA device 0. Prints the max abs and max relative difference per pair.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static const int N_TOK  = 4096;
static const int N_ITER = 20;

// returns the 4 x N_TOK mix values (row-major by element index) computed on `backend`
static std::vector<float> run(ggml_backend_t backend, const std::vector<float> & logits, bool fused) {
    ggml_init_params ip = { 64 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * L = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, N_TOK);
    ggml_set_input(L);

    ggml_tensor * out[4];
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
    if (fused) {
        ggml_tensor * mix = ggml_mhc_sinkhorn2(ctx, L, 4, N_ITER);
        for (int i = 0; i < 4; ++i) {
            out[i] = ggml_cont(ctx, ggml_view_2d(ctx, mix, 1, N_TOK, mix->nb[1], i * mix->nb[0]));
        }
    } else {
        ggml_tensor * m[4];
        for (int i = 0; i < 4; ++i) {
            m[i] = ggml_view_2d(ctx, L, 1, N_TOK, L->nb[1], (4 + i) * L->nb[0]);
        }
        auto normalize = [&](ggml_tensor * & x, ggml_tensor * & y) {
            ggml_tensor * delta = ggml_sub(ctx, x, y);
            x = ggml_neg(ctx, ggml_softplus(ctx, ggml_neg(ctx, delta)));
            y = ggml_neg(ctx, ggml_softplus(ctx, delta));
        };
        for (int it = 0; it < N_ITER; ++it) {
            normalize(m[0], m[1]);
            normalize(m[2], m[3]);
            normalize(m[0], m[2]);
            normalize(m[1], m[3]);
        }
        for (int i = 0; i < 4; ++i) {
            out[i] = ggml_cont(ctx, ggml_exp(ctx, m[i]));
        }
    }
    for (int i = 0; i < 4; ++i) {
        ggml_set_output(out[i]);
        ggml_build_forward_expand(gf, out[i]);
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(L, logits.data(), 0, logits.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed\n");
    }
    std::vector<float> res(4 * N_TOK);
    for (int i = 0; i < 4; ++i) {
        ggml_backend_tensor_get(out[i], res.data() + i * N_TOK, 0, N_TOK * sizeof(float));
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

static void cmp(const char * name, const std::vector<float> & a, const std::vector<float> & b) {
    double max_abs = 0, max_rel = 0;
    size_t n_diff = 0, n_bad = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) { n_bad++; continue; }
        const double d = std::fabs((double) a[i] - b[i]);
        if (d != 0) n_diff++;
        max_abs = std::fmax(max_abs, d);
        max_rel = std::fmax(max_rel, d / std::fmax(1e-30, std::fabs((double) b[i])));
    }
    printf("%-34s max abs %.3e  max rel %.3e  differing %zu/%zu  non-finite %zu\n",
           name, max_abs, max_rel, n_diff, a.size(), n_bad);
}

int main() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> wide(-30.0f, 30.0f);
    std::normal_distribution<float>       narrow(0.0f, 2.0f);
    std::vector<float> logits(8 * N_TOK);
    for (int t = 0; t < N_TOK; ++t) {
        for (int k = 0; k < 8; ++k) {
            logits[t * 8 + k] = (t % 4 == 0) ? wide(rng) : narrow(rng);   // a quarter extreme, rest typical
        }
    }

    ggml_backend_t cpu = ggml_backend_cpu_init();
    const auto cpu_unf = run(cpu, logits, false);
    const auto cpu_fus = run(cpu, logits, true);
    cmp("CPU: fused vs unfused", cpu_fus, cpu_unf);
#ifdef GGML_USE_CUDA
    ggml_backend_t cuda = ggml_backend_cuda_init(0);
    const auto gpu_unf = run(cuda, logits, false);
    const auto gpu_fus = run(cuda, logits, true);
    cmp("CUDA: fused vs unfused", gpu_fus, gpu_unf);
    cmp("CUDA fused vs CPU unfused", gpu_fus, cpu_unf);
    cmp("(baseline) CUDA vs CPU, both unfused", gpu_unf, cpu_unf);
    ggml_backend_free(cuda);
#endif
    ggml_backend_free(cpu);
    return 0;
}
