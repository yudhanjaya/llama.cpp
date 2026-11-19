#include "mhc-sinkhorn.cuh"

#define CUDA_MHC_SINKHORN2_BLOCK_SIZE 128

// same as the unary SOFTPLUS op (unary.cu)
static __device__ __forceinline__ float mhc_softplus(float x) {
    return (x > 20.0f) ? x : log1pf(expf(x));
}

// d = x - y; x = -softplus(-d); y = -softplus(d): the sub/neg/softplus graph this op replaces
static __device__ __forceinline__ void mhc_normalize(float & x, float & y) {
    const float d = x - y;
    x = -mhc_softplus(-d);
    y = -mhc_softplus(d);
}

// one thread per column: all n_iter rounds in registers, one launch per route
static __global__ void mhc_sinkhorn2_kernel(
        const char * src, float * dst, const int offset, const int n_iter,
        const int64_t ne1, const int64_t ne2, const int64_t n_cols,
        const size_t s_nb0, const size_t s_nb1, const size_t s_nb2, const size_t s_nb3,
        const size_t d_nb0, const size_t d_nb1, const size_t d_nb2, const size_t d_nb3) {
    const int64_t ir = (int64_t) blockDim.x * blockIdx.x + threadIdx.x;
    if (ir >= n_cols) {
        return;
    }
    const int64_t i3 = ir / (ne2 * ne1);
    const int64_t i2 = (ir - i3 * ne2 * ne1) / ne1;
    const int64_t i1 = ir - i3 * ne2 * ne1 - i2 * ne1;

    const char * s = src + i1 * s_nb1 + i2 * s_nb2 + i3 * s_nb3;
    char       * d = (char *) dst + i1 * d_nb1 + i2 * d_nb2 + i3 * d_nb3;

    float m0 = *(const float *) (s + (offset + 0) * s_nb0);
    float m1 = *(const float *) (s + (offset + 1) * s_nb0);
    float m2 = *(const float *) (s + (offset + 2) * s_nb0);
    float m3 = *(const float *) (s + (offset + 3) * s_nb0);

    for (int it = 0; it < n_iter; ++it) {
        mhc_normalize(m0, m1);
        mhc_normalize(m2, m3);
        mhc_normalize(m0, m2);
        mhc_normalize(m1, m3);
    }

    *(float *) (d + 0 * d_nb0) = expf(m0);
    *(float *) (d + 1 * d_nb0) = expf(m1);
    *(float *) (d + 2 * d_nb0) = expf(m2);
    *(float *) (d + 3 * d_nb0) = expf(m3);
}

void ggml_cuda_op_mhc_sinkhorn2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    const int offset = ggml_get_op_params_i32(dst, 0);
    const int n_iter = ggml_get_op_params_i32(dst, 1);

    const int64_t n_cols = ggml_nrows(dst);
    if (n_cols == 0) {
        return;
    }
    const int64_t n_blocks = (n_cols + CUDA_MHC_SINKHORN2_BLOCK_SIZE - 1) / CUDA_MHC_SINKHORN2_BLOCK_SIZE;

    mhc_sinkhorn2_kernel<<<n_blocks, CUDA_MHC_SINKHORN2_BLOCK_SIZE, 0, ctx.stream()>>>(
        (const char *) src0->data, (float *) dst->data, offset, n_iter,
        src0->ne[1], src0->ne[2], n_cols,
        src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
        dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
}
