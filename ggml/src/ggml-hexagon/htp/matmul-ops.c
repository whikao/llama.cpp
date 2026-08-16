#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>
#include <HAP_compute_res.h>

#include <math.h>
#include <string.h>
#include <stdatomic.h>

#include "hex-dma.h"
#include "hvx-utils.h"
#include "hvx-dump.h"
#include "hvx-arith.h"
#include "hvx-reduce.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "matmul-ops.h"
#include "htp-vtcm.h"

static void hvx_tensor_add_f32_grid(
    const struct htp_tensor * restrict dst,
    const struct htp_tensor * restrict src2,
    uint32_t start_row,
    uint32_t end_row,
    uint32_t start_col,
    uint32_t end_col,
    const struct fastdiv_values * div_ne11_12,
    const struct fastdiv_values * div_ne11
);

typedef struct {
    float        *dst;
    const float  *src2;
    const float  *activation;
    const __fp16 *weight;
    int           m;
    int           k;
    int           n;
    int           act_stride;
    int           weight_stride;
    int           dst_stride;
    uint32_t      src2_stride;
    int           ne02;
    int           ne03;
    int           ne12;
    int           ne13;
    size_t        src0_nb2;
    size_t        src0_nb3;
    size_t        src1_nb2;
    size_t        src1_nb3;
    size_t        dst_nb2;
    size_t        dst_nb3;
    size_t        src2_nb2;
    size_t        src2_nb3;
} hmx_mm_f16_f32_batched_params_t;

struct htp_mm_context {
    const char * type;
    struct htp_ops_context * octx;

    // v10.3 raw-Q4_0 correctness return. These live only for one op invocation,
    // so worker threads can record the first selected expert without relying on FARF/logcat.
    volatile uint32_t dbg_v103_claimed;
    uint32_t dbg_v103_expert;
    uint32_t dbg_v103_raw_fnv;
    uint32_t dbg_v103_tile_fnv;
    uint32_t dbg_v103_src_off;
    uint32_t dbg_v106_ct;
    uint32_t dbg_v106_ith;
    uint32_t dbg_v106_valid_rows;

    // v10.7: exact parameters passed to the first q4_0 dot for the captured tile.
    uint32_t dbg_v107_ne00;
    uint32_t dbg_v107_ne10;
    uint32_t dbg_v107_nb01;
    uint32_t dbg_v107_nb02;
    uint32_t dbg_v107_src1_stride;
    uint32_t dbg_v107_rm1;
    uint32_t dbg_v107_rm2;
    uint32_t dbg_v107_ir1;
    uint32_t dbg_v107_src1_off;

    // v10.8.1: bit patterns of the first four outputs produced by the
    // deterministic raw-Q4_0 dot call (expert captured by v10.6, ct=0, cid=0).
    uint32_t dbg_v108_out0;
    uint32_t dbg_v108_out1;
    uint32_t dbg_v108_out2;
    uint32_t dbg_v108_out3;

    // v10.9: exact src1/Q8 activation diagnostics for the same captured MMID row.
    uint32_t dbg_v109_quant_fnv;     // first 1152-byte Q8 tile after quant barrier
    uint32_t dbg_v109_dot_fnv;       // same 1152 bytes immediately before dot
    uint32_t dbg_v109_q8_head;       // first 4 bytes of tiled Q8 data
    uint32_t dbg_v109_scale_head;    // first 4 bytes at Q8 tile + 1024 (scale vector)
    uint32_t dbg_v109_src_f0_bits;   // source F32 first element
    uint32_t dbg_v109_src_max_bits;  // max(abs(src[0..127])) as F32 bits
    uint32_t dbg_v109_row_fnv;       // whole quantized src1 row (vtcm_src1_stride bytes)

    // v10.11: first K=32 block, row-0 HTP-vs-scalar reference.
    uint32_t dbg_v111_hvx_k32_bits;
    uint32_t dbg_v111_ref_k32_bits;
    int32_t  dbg_v111_int_dot;
    uint32_t dbg_v111_scales_fp16;   // low16=Q4 scale, high16=Q8 scale

    // v10.12 runtime debug controls.
    uint32_t dbg_v112_enabled;
    int32_t  dbg_v112_want_expert;
    uint32_t dbg_v112_want_ct;
    uint32_t dbg_v112_want_cid;
    uint32_t dbg_v112_want_k;

    // v10.21: actual activation-quantization implementation selected for
    // this MMID invocation (block or row), including an optional forced A/B.
    uint32_t dbg_v121_quant_mode;

    // v10.14.1: exact row-0 integer accumulator from accum_4bit_32x1().
    int32_t dbg_v114_hvx_int_dot;

    // v10.18: full-K row-0 reference using the exact v10.16 Q8 quantizer replay.
    uint32_t dbg_v118_full_ref_bits;

    // v10.15: selected-tile element diagnostics.
    int8_t  dbg_v115_q8_actual[32];
    int8_t  dbg_v115_q8_scalar[32];
    int8_t  dbg_v115_q4_weight[32];
    int16_t dbg_v115_prod_delta[32];

    void (*vec_dot_1x1)(const uint32_t n, float * restrict s0,
         const void * restrict vx0,
         const void * restrict vy0);

    void (*vec_dot_2x1)(const uint32_t n, float * restrict s0,
         const void * restrict vx0, const void * restrict vx1,
         const void * restrict vy0);

    void (*vec_dot_2x2)(const uint32_t n, float * restrict s0, float * restrict s1,
         const void * restrict vx0, const void * restrict vx1,
         const void * restrict vy0, const void * restrict vy1);

    void (*vec_dot_32x1)(const uint32_t n, float * restrict s,
         const void * restrict vx,
         const void * restrict vy, uint32_t valid_rows,
         const float * restrict sz);

    // Precomputed values
    uint32_t src0_nrows_per_thread;
    uint32_t src0_row_size_padded;
    uint32_t src1_nrows;

    struct fastdiv_values mm_div_ne12_ne1;
    struct fastdiv_values mm_div_ne1;
    struct fastdiv_values mm_div_r2;
    struct fastdiv_values mm_div_r3;
    struct fastdiv_values mm_div_ne11;

    // Per thread quant tasks
    // Precomputed block-parallel quantization values
    worker_callback_t quant_task_func;
    uint32_t          quant_ib_first[WORK_QUEUE_MAX_N_THREADS];
    uint32_t          quant_ib_last[WORK_QUEUE_MAX_N_THREADS];
    uint32_t          quant_r[WORK_QUEUE_MAX_N_THREADS];
    uint32_t          quant_c[WORK_QUEUE_MAX_N_THREADS];
    uint32_t          n_quant_tasks;
    uint32_t          n_quant_rows_per_thread;
    atomic_uint       quant_barrier;

    // Fields for scattered mapping & HMX support in MUL_MAT_ID
    const uint32_t * matrix_row_counts;
    const struct mmid_row_mapping * matrix_rows;
    uint32_t mapping_stride;

    // Dynamic VTCM pointers allocated sequentially
    uint8_t * vtcm_src0;
    uint8_t * vtcm_src1;
    uint8_t * vtcm_src2;
    uint8_t * vtcm_src3;
    uint8_t * vtcm_dst;

    // Cached strides
    uint32_t vtcm_src0_stride;
    uint32_t vtcm_src1_stride;
    uint32_t vtcm_src2_stride;
    uint32_t vtcm_src3_stride;

    // Cached thread offsets/sizes
    uint32_t vtcm_src0_size_per_thread;
    uint32_t vtcm_src1_size_per_thread;
    uint32_t vtcm_src2_size_per_thread;
    uint32_t vtcm_src3_size_per_thread;
    uint32_t vtcm_dst_size_per_thread;
};

// vdelta control to expand first 32 e8m0 values into 32 uint32 elements
static const uint8_t __attribute__((aligned(128))) expand_x32_e8m0[128] = {
    0x00, 0x00, 0x00, 0x00, 0x01, 0x04, 0x00, 0x00, 0x02, 0x00, 0x08, 0x08, 0x01, 0x02, 0x00, 0x04, 0x04, 0x00, 0x00,
    0x00, 0x11, 0x10, 0x10, 0x10, 0x02, 0x00, 0x04, 0x00, 0x01, 0x02, 0x08, 0x08, 0x08, 0x08, 0x00, 0x00, 0x01, 0x04,
    0x00, 0x00, 0x22, 0x20, 0x20, 0x20, 0x21, 0x22, 0x20, 0x24, 0x04, 0x00, 0x00, 0x00, 0x09, 0x08, 0x00, 0x00, 0x02,
    0x00, 0x04, 0x00, 0x11, 0x12, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x01, 0x04, 0x00, 0x00, 0x02, 0x00, 0x08, 0x08,
    0x01, 0x02, 0x00, 0x04, 0x44, 0x40, 0x40, 0x40, 0x41, 0x40, 0x40, 0x40, 0x42, 0x40, 0x44, 0x40, 0x41, 0x42, 0x48,
    0x48, 0x08, 0x08, 0x00, 0x00, 0x01, 0x04, 0x00, 0x00, 0x12, 0x10, 0x10, 0x10, 0x01, 0x02, 0x00, 0x04, 0x04, 0x00,
    0x00, 0x00, 0x09, 0x08, 0x00, 0x00, 0x22, 0x20, 0x24, 0x20, 0x21, 0x22, 0x20, 0x20,
};

// IQ4_NL dequantization LUT: maps 4-bit index (0-15) to int8 kvalue
// kvalues: -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
static const uint8_t __attribute__((aligned(VLEN))) kvalues_iq4nl_lut[] = {
    0x81, 0, 0x98, 0, 0xAD, 0, 0xBF, 0, 0xCF, 0, 0xDD, 0, 0xEA, 0, 0xF6, 0, 0x01, 0, 0x0D, 0, 0x19, 0, 0x26, 0,
    0x35, 0, 0x45, 0, 0x59, 0, 0x71, 0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
};

static const uint8_t __attribute__((aligned(VLEN))) kvalues_mxfp4_lut[] = {
    0,    0, 1,    0, 2,    0, 3, 0, 4, 0, 6, 0, 8, 0, 12, 0, 0, 0, 0xff, 0, 0xfe, 0, 0xfd, 0, 0xfc, 0,
    0xfa, 0, 0xf8, 0, 0xf4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0,
};

#define htp_matmul_tensors_preamble                                 \
    const struct htp_tensor * restrict src0 = octx->src[0];         \
    const struct htp_tensor * restrict src1 = octx->src[1];         \
    const struct htp_tensor * restrict src2 = octx->src[2];         \
    const struct htp_tensor * restrict  dst = octx->dst;            \
                                                                    \
    const uint32_t ne00 = src0->ne[0];                              \
    const uint32_t ne01 = src0->ne[1];                              \
    const uint32_t ne02 = src0->ne[2];                              \
    const uint32_t ne03 = src0->ne[3];                              \
                                                                    \
    const uint32_t ne10 = src1->ne[0];                              \
    const uint32_t ne11 = src1->ne[1];                              \
    const uint32_t ne12 = src1->ne[2];                              \
    const uint32_t ne13 = src1->ne[3];                              \
                                                                    \
    const uint32_t ne20 = src2 ? src2->ne[0] : 0;                   \
    const uint32_t ne21 = src2 ? src2->ne[1] : 0;                   \
    const uint32_t ne22 = src2 ? src2->ne[2] : 0;                   \
    const uint32_t ne23 = src2 ? src2->ne[3] : 0;                   \
                                                                    \
    const uint32_t ne0 = dst->ne[0];                                \
    const uint32_t ne1 = dst->ne[1];                                \
    const uint32_t ne2 = dst->ne[2];                                \
    const uint32_t ne3 = dst->ne[3];                                \
                                                                    \
    const uint32_t nb00 = src0->nb[0];                              \
    const uint32_t nb01 = src0->nb[1];                              \
    const uint32_t nb02 = src0->nb[2];                              \
    const uint32_t nb03 = src0->nb[3];                              \
                                                                    \
    const uint32_t nb10 = src1->nb[0];                              \
    const uint32_t nb11 = src1->nb[1];                              \
    const uint32_t nb12 = src1->nb[2];                              \
    const uint32_t nb13 = src1->nb[3];                              \
                                                                    \
    const uint32_t nb0 = dst->nb[0];                                \
    const uint32_t nb1 = dst->nb[1];                                \
    const uint32_t nb2 = dst->nb[2];                                \
    const uint32_t nb3 = dst->nb[3];

#define htp_matmul_preamble                                         \
    struct htp_mm_context * mmctx  = data;                          \
    struct htp_ops_context * octx  = mmctx->octx;                   \
    dma_queue *dma_queue           = octx->ctx->dma[ith];           \
    uint32_t src0_nrows_per_thread = mmctx->src0_nrows_per_thread;  \
    htp_matmul_tensors_preamble;

static inline void hvx_mm_run_quant_task(struct htp_mm_context * mmctx, unsigned int ith) {
    if (mmctx->quant_task_func) {
        if (ith < mmctx->n_quant_tasks) {
            mmctx->quant_task_func(mmctx->n_quant_tasks, ith, mmctx);
            atomic_fetch_sub(&mmctx->quant_barrier, 1);
        }
        while (atomic_load(&mmctx->quant_barrier) > 0) {
            // spin
        }
    }
}

// *** matmul with support for 4d tensors and full broadcasting

static void hvx_mm_4d(unsigned int nth, unsigned int ith, void * data) {
    htp_matmul_preamble;

    assert(ne12 % ne02 == 0);
    assert(ne13 % ne03 == 0);

    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
    const uint32_t nr0 = ne0;

    // This is the size of the rest of the dimensions of the result
    const uint32_t nr1 = ne1 * ne2 * ne3;

    // distribute the thread work across the inner or outer loop based on which one is larger
    uint32_t nchunk0 = nr0 > nr1 ? nth : 1;  // parallelize by src0 rows
    uint32_t nchunk1 = nr0 > nr1 ? 1 : nth;  // parallelize by src1 rows

    // The number of elements in each chunk
    const uint32_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    const uint32_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    uint32_t current_chunk = ith;

    const uint32_t ith0 = current_chunk % nchunk0;
    const uint32_t ith1 = current_chunk / nchunk0;

    const uint32_t ir0_start = dr0 * ith0;
    const uint32_t ir0_end   = MIN(ir0_start + dr0, nr0);

    const uint32_t ir1_start = dr1 * ith1;
    const uint32_t ir1_end   = MIN(ir1_start + dr1, nr1);

    // no work for this thread
    if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        return;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0_start);

    const uint32_t blck_0 = 64;
    const uint32_t blck_1 = 64;

    for (uint32_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (uint32_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (uint32_t ir1 = iir1; ir1 < MIN(iir1 + blck_1, ir1_end); ir1++) {
                const uint32_t i13 = fastdiv(ir1, &mmctx->mm_div_ne12_ne1);
                const uint32_t i12 = fastdiv(ir1 - i13 * ne12 * ne1, &mmctx->mm_div_ne1);
                const uint32_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                // broadcast src0 into src1
                const uint32_t i03 = fastdiv(i13, &mmctx->mm_div_r3);
                const uint32_t i02 = fastdiv(i12, &mmctx->mm_div_r2);

                const uint32_t i1 = i11;
                const uint32_t i2 = i12;
                const uint32_t i3 = i13;

                const uint8_t * restrict src0_base = (const uint8_t *) src0->data + (0 + i02 * nb02 + i03 * nb03);
                const uint8_t * restrict src1_col  = (const uint8_t *) src1->data + (i11 * nb11 + i12 * nb12 + i13 * nb13);
                float * dst_col = (float *) ((uint8_t * restrict) dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                const uint32_t ir0_block_end = MIN(iir0 + blck_0, ir0_end);
                for (uint32_t ir0 = iir0; ir0 < ir0_block_end; ir0++) {
                    const uint8_t * restrict src0_row = src0_base + ir0 * nb01;
                    mmctx->vec_dot_1x1(ne00, &dst_col[ir0], src0_row, src1_col);
                }
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0_start);
    if (src2) {
        hvx_tensor_add_f32_grid(dst, src2, ir1_start, ir1_end, ir0_start, ir0_end, &mmctx->mm_div_ne12_ne1, &mmctx->mm_div_ne1);
    }
}

#include "hmx-mm-kernels-tiled.h"
#include "hvx-mm-kernels-tiled.h"
#include "hvx-mm-kernels-flat.h"

// Specialized repacked matmul macros
#define MATMUL_2D_REPACKED_IMPL(SUFFIX, TILE_SIZE, DOT_2X2, DOT_2X1)                                                              \
static void hvx_mm_2d_repacked_##SUFFIX(unsigned int nth, unsigned int ith, void * data) {                                        \
    htp_matmul_preamble;                                                                                                          \
                                                                                                                                  \
    const uint32_t src0_nrows = ne01 * ne02 * ne03;                                                                               \
    const uint32_t src1_nrows = ne11 * ne12 * ne13;                                                                               \
                                                                                                                                  \
    const uint32_t src0_start_row  = src0_nrows_per_thread * ith;                                                                 \
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);                                     \
                                                                                                                                  \
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];                                                                        \
                                                                                                                                  \
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;                      \
    const uint32_t n_prefetch = kparams->n_prefetch;                                                                              \
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);                         \
                                                                                                                                  \
    const size_t dst_row_size  = nb1;                                                                                             \
    const size_t src1_row_size = nb11;                                                                                            \
    const size_t src1_stride = mmctx->vtcm_src1_stride;                                                                           \
    const size_t src2_stride = src2 ? ((src2->ne[1] == 1) ? 0 : src2->nb[1]) : 0;                                                 \
                                                                                                                                  \
    uint8_t * restrict vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_size_per_thread  * ith;                                 \
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;                                 \
    uint8_t * restrict src1_data = mmctx->vtcm_src1;                                                                              \
                                                                                                                                  \
    const uint8_t * restrict src0_row = (const uint8_t *) src0->data;                                                             \
                                                                                                                                  \
    const uint32_t tile_size = TILE_SIZE;                                                                                         \
    const uint32_t aligned_tile_size = hex_align_up(tile_size, 128);                                                              \
                                                                                                                                  \
    uint32_t n_k_tiles_w = ne00 / 32;                                                                                             \
    uint32_t n_k_tiles_a = ne10 / 32;                                                                                             \
    uint32_t tile_row_stride = n_k_tiles_w * tile_size;                                                                           \
    uint32_t tile_row_transfer_size_aligned = n_k_tiles_a * aligned_tile_size;                                                    \
                                                                                                                                  \
    uint32_t ct_start = src0_start_row / 32;                                                                                      \
    uint32_t ct_end   = (src0_end_row + 31) / 32;                                                                                 \
                                                                                                                                  \
    uint32_t push_ct = ct_start;                                                                                                  \
    if (src0_start_row < src0_end_row) {                                                                                          \
        for (uint32_t d = 0; d < n_prefetch && push_ct < ct_end; d++, push_ct++) {                                                \
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + d * tile_row_transfer_size_aligned,                            \
                           src0_row + push_ct * tile_row_stride), aligned_tile_size, tile_size, tile_size, n_k_tiles_a);          \
        }                                                                                                                         \
    }                                                                                                                             \
                                                                                                                                  \
    hvx_mm_run_quant_task(mmctx, ith);                                                                                            \
                                                                                                                                  \
    if (src0_start_row >= src0_end_row) {                                                                                         \
        return;                                                                                                                   \
    }                                                                                                                             \
                                                                                                                                  \
    for (uint32_t ct = ct_start; ct < ct_end; ct++) {                                                                             \
        const uint8_t * w_tile = dma_queue_pop(dma_queue).dst;                                                                    \
                                                                                                                                  \
        int valid_rows = (int)ne0 - (int)(ct * 32);                                                                               \
        valid_rows = MIN(32, MAX(0, valid_rows));                                                                                 \
                                                                                                                                  \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);                                                                    \
        uint32_t ir1 = 0;                                                                                                         \
        for (; ir1 + 1 < src1_nrows; ir1 += 2) {                                                                                  \
            const uint8_t * restrict src1_col0 = (const uint8_t *) (src1_data + (ir1+0) * src1_stride);                           \
            const uint8_t * restrict src1_col1 = (const uint8_t *) (src1_data + (ir1+1) * src1_stride);                           \
            float * restrict dst_row0 = (float *) (dst->data + ((ir1+0) * dst_row_size));                                         \
            float * restrict dst_row1 = (float *) (dst->data + ((ir1+1) * dst_row_size));                                         \
                                                                                                                                  \
            float * dst_ptr0 = &dst_row0[ct * 32];                                                                                \
            float * dst_ptr1 = &dst_row1[ct * 32];                                                                                \
                                                                                                                                  \
            const float * src2_ptr0 = NULL;                                                                                       \
            const float * src2_ptr1 = NULL;                                                                                       \
            if (src2) {                                                                                                           \
                const float * restrict src2_row0 = (const float *) ((const uint8_t *) src2->data + ((ir1+0) * src2_stride));      \
                const float * restrict src2_row1 = (const float *) ((const uint8_t *) src2->data + ((ir1+1) * src2_stride));      \
                src2_ptr0 = &src2_row0[ct * 32];                                                                                  \
                src2_ptr1 = &src2_row1[ct * 32];                                                                                  \
            }                                                                                                                     \
            DOT_2X2(ne10, dst_ptr0, dst_ptr1, w_tile, src1_col0, src1_col1, valid_rows, src2_ptr0, src2_ptr1);                    \
        }                                                                                                                         \
                                                                                                                                  \
        for (; ir1 < src1_nrows; ++ir1) {                                                                                         \
            const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);                                \
            float * restrict dst_row          = (float *) (dst->data + (ir1 * dst_row_size));                                     \
            float * dst_ptr = &dst_row[ct * 32];                                                                                  \
                                                                                                                                  \
            const float * src2_ptr = NULL;                                                                                        \
            if (src2) {                                                                                                           \
                const float * restrict src2_row = (const float *) ((const uint8_t *) src2->data + (ir1 * src2_stride));           \
                src2_ptr = &src2_row[ct * 32];                                                                                    \
            }                                                                                                                     \
            DOT_2X1(ne10, dst_ptr, w_tile, src1_col, valid_rows, src2_ptr);                                                       \
        }                                                                                                                         \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);                                                                     \
                                                                                                                                  \
        if (push_ct < ct_end) {                                                                                                   \
            dma_queue_push(dma_queue, dma_make_ptr((uint8_t *)w_tile, src0_row + push_ct * tile_row_stride),                      \
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_a);                                                 \
            push_ct++;                                                                                                            \
        }                                                                                                                         \
    }                                                                                                                             \
}

#define MATVEC_2D_REPACKED_IMPL(SUFFIX, TILE_SIZE, DOT_2X1)                                                                       \
static void hvx_mv_2d_repacked_##SUFFIX(unsigned int nth, unsigned int ith, void * data) {                                        \
    htp_matmul_preamble;                                                                                                          \
                                                                                                                                  \
    const uint32_t src0_nrows = ne01;                                                                                             \
                                                                                                                                  \
    const uint32_t src0_start_row  = src0_nrows_per_thread * ith;                                                                 \
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);                                     \
                                                                                                                                  \
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];                                                                        \
                                                                                                                                  \
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;                      \
    const uint32_t n_prefetch = kparams->n_prefetch;                                                                              \
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);                         \
                                                                                                                                  \
    const size_t dst_row_size  = nb1;                                                                                             \
    const size_t src1_row_size = nb11;                                                                                            \
    const size_t src1_stride = mmctx->vtcm_src1_stride;                                                                           \
                                                                                                                                  \
    uint8_t * vtcm_dst_ptr  = mmctx->vtcm_dst + mmctx->vtcm_dst_size_per_thread * ith;                                            \
    uint8_t * vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;                                          \
    uint8_t * src1_data = mmctx->vtcm_src1;                                                                                       \
                                                                                                                                  \
    float * tmp = (float *) vtcm_dst_ptr;                                                                                         \
                                                                                                                                  \
    const uint8_t * restrict src0_row = (const uint8_t *) src0->data;                                                             \
                                                                                                                                  \
    const uint8_t * restrict src1_col = (const uint8_t *) src1_data;                                                              \
    float * restrict dst_col          = (float *) dst->data;                                                                      \
                                                                                                                                  \
    const uint32_t tile_size = TILE_SIZE;                                                                                         \
    const uint32_t aligned_tile_size = hex_align_up(tile_size, 128);                                                              \
                                                                                                                                  \
    uint32_t n_k_tiles_w = ne00 / 32;                                                                                             \
    uint32_t n_k_tiles_a = ne10 / 32;                                                                                             \
    uint32_t tile_row_stride = n_k_tiles_w * tile_size;                                                                           \
    uint32_t tile_row_transfer_size_aligned = n_k_tiles_a * aligned_tile_size;                                                    \
                                                                                                                                  \
    uint32_t ct_start = src0_start_row / 32;                                                                                      \
    uint32_t ct_end   = (src0_end_row + 31) / 32;                                                                                 \
                                                                                                                                  \
    uint32_t push_ct = ct_start;                                                                                                  \
    if (src0_start_row < src0_end_row) {                                                                                          \
        if (src2) {                                                                                                               \
            float * vtcm_src2_ptr = (float *) mmctx->vtcm_src2 + src0_start_row;                                                  \
            const float * src2_ptr = (const float *) src2->data + src0_start_row;                                                 \
            int slice_size = (int)MIN(src0_end_row, ne0) - (int)src0_start_row;                                                   \
            if (slice_size > 0) {                                                                                                 \
                dma_queue_push(dma_queue, dma_make_ptr(vtcm_src2_ptr, src2_ptr),                                                  \
                               slice_size * sizeof(float), slice_size * sizeof(float), slice_size * sizeof(float), 1);            \
                dma_queue_pop_nowait(dma_queue);                                                                                  \
            }                                                                                                                     \
        }                                                                                                                         \
        for (uint32_t d = 0; d < n_prefetch && push_ct < ct_end; d++, push_ct++) {                                                \
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + d * tile_row_transfer_size_aligned,                            \
                           src0_row + push_ct * tile_row_stride), aligned_tile_size, tile_size, tile_size, n_k_tiles_a);          \
        }                                                                                                                         \
    }                                                                                                                             \
                                                                                                                                  \
    hvx_mm_run_quant_task(mmctx, ith);                                                                                            \
                                                                                                                                  \
    if (src0_start_row >= src0_end_row) {                                                                                         \
        return;                                                                                                                   \
    }                                                                                                                             \
                                                                                                                                  \
    for (uint32_t ct = ct_start; ct < ct_end; ct++) {                                                                             \
        const uint8_t * w_tile = dma_queue_pop(dma_queue).dst;                                                                    \
                                                                                                                                  \
        float * dst_ptr = &tmp[ct * 32 - src0_start_row];                                                                         \
        int valid_rows = (int)ne0 - (int)(ct * 32);                                                                               \
        valid_rows = MIN(32, MAX(0, valid_rows));                                                                                 \
                                                                                                                                  \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);                                                                    \
        DOT_2X1(ne10, dst_ptr, w_tile, src1_col, valid_rows, NULL);                                                               \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);                                                                    \
                                                                                                                                  \
        if (push_ct < ct_end) {                                                                                                   \
            dma_queue_push(dma_queue, dma_make_ptr((uint8_t *)w_tile, src0_row + push_ct * tile_row_stride),                      \
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_a);                                                 \
            push_ct++;                                                                                                            \
        }                                                                                                                         \
    }                                                                                                                             \
                                                                                                                                  \
    int copy_cnt = (int)MIN(src0_end_row, ne0) - (int)src0_start_row;                                                             \
    if (copy_cnt > 0) {                                                                                                           \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct_end);                                                                \
        if (src2) {                                                                                                               \
            hvx_add_f32_uaa((uint8_t *) &dst_col[src0_start_row],                                                                 \
                            (const uint8_t *) tmp,                                                                                \
                            (const uint8_t *) ((const float *) mmctx->vtcm_src2 + src0_start_row),                                \
                            copy_cnt);                                                                                            \
        } else {                                                                                                                  \
            hvx_copy_f32_ua((uint8_t *) &dst_col[src0_start_row], (uint8_t *) tmp, copy_cnt);                                     \
        }                                                                                                                         \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct_end);                                                                 \
    }                                                                                                                             \
}

#define MATMUL_QKV_2D_REPACKED_IMPL(SUFFIX, TILE_SIZE, DOT_2X2, DOT_2X1)                                                          \
static void hvx_mm_qkv_2d_repacked_##SUFFIX(unsigned int nth, unsigned int ith, void * data) {                                    \
    struct htp_mm_context * mmctx = data;                                                                                         \
    struct htp_ops_context * octx = mmctx->octx;                                                                                  \
                                                                                                                                  \
    const struct htp_tensor * restrict src0 = octx->src[0]; /* Wk */                                                              \
    const struct htp_tensor * restrict src1 = octx->src[1]; /* x */                                                               \
    const struct htp_tensor * restrict src2 = octx->src[2]; /* Wv */                                                              \
    const struct htp_tensor * restrict src3 = octx->src[3]; /* Wq */                                                              \
    const struct htp_tensor * restrict dst_k = octx->dsts[0];                                                                     \
    const struct htp_tensor * restrict dst_v = octx->dsts[1];                                                                     \
    const struct htp_tensor * restrict dst_q = octx->dsts[2];                                                                     \
                                                                                                                                  \
    const uint32_t ne00 = src0->ne[0];                                                                                            \
    const uint32_t ne10 = src1->ne[0];                                                                                            \
    const uint32_t src1_nrows = src1->ne[1] * src1->ne[2] * src1->ne[3];                                                          \
                                                                                                                                  \
    const size_t dst_k_row_size = dst_k->nb[1]; /* K and V share output width */                                                  \
    const size_t dst_q_row_size = dst_q->nb[1]; /* Q may be wider (GQA) */                                                        \
    const size_t src1_stride = mmctx->vtcm_src1_stride;                                                                           \
                                                                                                                                  \
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;                                 \
    uint8_t * restrict vtcm_src2_ptr = mmctx->vtcm_src2 + mmctx->vtcm_src2_size_per_thread * ith;                                 \
    uint8_t * restrict vtcm_src3_ptr = mmctx->vtcm_src3 + mmctx->vtcm_src3_size_per_thread * ith;                                 \
    uint8_t * restrict src1_data = mmctx->vtcm_src1;                                                                              \
                                                                                                                                  \
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];                                                                        \
                                                                                                                                  \
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;                      \
    const uint32_t n_prefetch = kparams->n_prefetch;                                                                              \
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);                         \
                                                                                                                                  \
    const uint8_t * restrict src0_row = (const uint8_t *) src0->data;                                                             \
    const uint8_t * restrict src2_row = (const uint8_t *) src2->data;                                                             \
    const uint8_t * restrict src3_row = (const uint8_t *) src3->data;                                                             \
                                                                                                                                  \
    const uint32_t tile_size = TILE_SIZE;                                                                                         \
    const uint32_t aligned_tile_size = hex_align_up(tile_size, 128);                                                              \
                                                                                                                                  \
    uint32_t n_k_tiles_w = ne00 / 32;                                                                                             \
    uint32_t n_k_tiles_a = ne10 / 32;                                                                                             \
    uint32_t tile_row_stride = n_k_tiles_w * tile_size;                                                                           \
    uint32_t tile_row_transfer_size_aligned = n_k_tiles_a * aligned_tile_size;                                                    \
                                                                                                                                  \
    dma_queue * dma_queue = octx->ctx->dma[ith];                                                                                  \
                                                                                                                                  \
    /* 1. Process K and V together */                                                                                             \
    const uint32_t src0_nrows_kv = src0->ne[1] * src0->ne[2] * src0->ne[3]; /* src0 is Wk */                                      \
    uint32_t src0_nrows_per_thread_kv = (src0_nrows_kv + nth - 1) / nth;                                                          \
    src0_nrows_per_thread_kv = hex_round_up(src0_nrows_per_thread_kv, 32);                                                        \
                                                                                                                                  \
    const uint32_t start_row_kv = src0_nrows_per_thread_kv * ith;                                                                 \
    const uint32_t end_row_kv   = MIN(start_row_kv + src0_nrows_per_thread_kv, src0_nrows_kv);                                    \
                                                                                                                                  \
    uint32_t ct_start_kv = start_row_kv / 32;                                                                                     \
    uint32_t ct_end_kv   = (end_row_kv + 31) / 32;                                                                                \
                                                                                                                                  \
    uint32_t push_ct = ct_start_kv;                                                                                               \
    if (start_row_kv < end_row_kv) {                                                                                              \
        for (uint32_t d = 0; d < n_prefetch && push_ct < ct_end_kv; d++, push_ct++) {                                             \
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + d * tile_row_transfer_size_aligned,                            \
                           src0_row + push_ct * tile_row_stride), aligned_tile_size, tile_size, tile_size, n_k_tiles_a);          \
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src2_ptr + d * tile_row_transfer_size_aligned,                            \
                           src2_row + push_ct * tile_row_stride), aligned_tile_size, tile_size, tile_size, n_k_tiles_a);          \
        }                                                                                                                         \
    }                                                                                                                             \
                                                                                                                                  \
    hvx_mm_run_quant_task(mmctx, ith);                                                                                            \
                                                                                                                                  \
    if (start_row_kv < end_row_kv) {                                                                                              \
                                                                                                                                  \
        for (uint32_t ct = ct_start_kv; ct < ct_end_kv; ct++) {                                                                   \
            const uint8_t * w_tile_k = dma_queue_pop(dma_queue).dst;                                                              \
            const uint8_t * w_tile_v = dma_queue_pop(dma_queue).dst;                                                              \
                                                                                                                                  \
            int valid_rows = (int)src0->ne[1] - (int)(ct * 32);                                                                   \
            valid_rows = MIN(32, MAX(0, valid_rows));                                                                             \
                                                                                                                                  \
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ith);                                                               \
            uint32_t ir1 = 0;                                                                                                     \
            for (; ir1 + 1 < src1_nrows; ir1 += 2) {                                                                              \
                const uint8_t * restrict src1_col0 = (const uint8_t *) (src1_data + (ir1+0) * src1_stride);                       \
                const uint8_t * restrict src1_col1 = (const uint8_t *) (src1_data + (ir1+1) * src1_stride);                       \
                                                                                                                                  \
                float * restrict dst_row0_k = (float *) (dst_k->data + ((ir1+0) * dst_k_row_size));                               \
                float * restrict dst_row1_k = (float *) (dst_k->data + ((ir1+1) * dst_k_row_size));                               \
                float * dst_ptr0_k = &dst_row0_k[ct * 32];                                                                        \
                float * dst_ptr1_k = &dst_row1_k[ct * 32];                                                                        \
                                                                                                                                  \
                float * restrict dst_row0_v = (float *) (dst_v->data + ((ir1+0) * dst_k_row_size));                               \
                float * restrict dst_row1_v = (float *) (dst_v->data + ((ir1+1) * dst_k_row_size));                               \
                float * dst_ptr0_v = &dst_row0_v[ct * 32];                                                                        \
                float * dst_ptr1_v = &dst_row1_v[ct * 32];                                                                        \
                                                                                                                                  \
                DOT_2X2(ne10, dst_ptr0_k, dst_ptr1_k, w_tile_k, src1_col0, src1_col1, valid_rows, NULL, NULL);                    \
                DOT_2X2(ne10, dst_ptr0_v, dst_ptr1_v, w_tile_v, src1_col0, src1_col1, valid_rows, NULL, NULL);                    \
            }                                                                                                                     \
                                                                                                                                  \
            for (; ir1 < src1_nrows; ++ir1) {                                                                                     \
                const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);                            \
                                                                                                                                  \
                float * restrict dst_row_k = (float *) (dst_k->data + (ir1 * dst_k_row_size));                                    \
                float * dst_ptr_k = &dst_row_k[ct * 32];                                                                          \
                                                                                                                                  \
                float * restrict dst_row_v = (float *) (dst_v->data + (ir1 * dst_k_row_size));                                    \
                float * dst_ptr_v = &dst_row_v[ct * 32];                                                                          \
                                                                                                                                  \
                DOT_2X1(ne10, dst_ptr_k, w_tile_k, src1_col, valid_rows, NULL);                                                   \
                DOT_2X1(ne10, dst_ptr_v, w_tile_v, src1_col, valid_rows, NULL);                                                   \
            }                                                                                                                     \
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ith);                                                                \
                                                                                                                                  \
            if (push_ct < ct_end_kv) {                                                                                            \
                dma_queue_push(dma_queue, dma_make_ptr((uint8_t *)w_tile_k, src0_row + push_ct * tile_row_stride),                \
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_a);                                             \
                dma_queue_push(dma_queue, dma_make_ptr((uint8_t *)w_tile_v, src2_row + push_ct * tile_row_stride),                \
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_a);                                             \
                push_ct++;                                                                                                        \
            }                                                                                                                     \
        }                                                                                                                         \
    }                                                                                                                             \
                                                                                                                                  \
    /* 2. Process Q separately */                                                                                                 \
    const uint32_t src0_nrows_q = src3->ne[1] * src3->ne[2] * src3->ne[3]; /* src3 is Wq */                                       \
    uint32_t src0_nrows_per_thread_q = (src0_nrows_q + nth - 1) / nth;                                                            \
    src0_nrows_per_thread_q = hex_round_up(src0_nrows_per_thread_q, 32);                                                          \
                                                                                                                                  \
    const uint32_t start_row_q = src0_nrows_per_thread_q * ith;                                                                   \
    const uint32_t end_row_q   = MIN(start_row_q + src0_nrows_per_thread_q, src0_nrows_q);                                        \
                                                                                                                                  \
    if (start_row_q < end_row_q) {                                                                                                \
        uint32_t ct_start_q = start_row_q / 32;                                                                                   \
        uint32_t ct_end_q   = (end_row_q + 31) / 32;                                                                              \
                                                                                                                                  \
        uint32_t push_ct = ct_start_q;                                                                                            \
        for (uint32_t d = 0; d < n_prefetch && push_ct < ct_end_q; d++, push_ct++) {                                              \
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src3_ptr + d * tile_row_transfer_size_aligned,                            \
                           src3_row + push_ct * tile_row_stride), aligned_tile_size, tile_size, tile_size, n_k_tiles_a);          \
        }                                                                                                                         \
                                                                                                                                  \
        for (uint32_t ct = ct_start_q; ct < ct_end_q; ct++) {                                                                     \
            const uint8_t * w_tile_q = dma_queue_pop(dma_queue).dst;                                                              \
                                                                                                                                  \
            int valid_rows = (int)src3->ne[1] - (int)(ct * 32);                                                                   \
            valid_rows = MIN(32, MAX(0, valid_rows));                                                                             \
                                                                                                                                  \
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);                                                                \
            uint32_t ir1 = 0;                                                                                                     \
            for (; ir1 + 1 < src1_nrows; ir1 += 2) {                                                                              \
                const uint8_t * restrict src1_col0 = (const uint8_t *) (src1_data + (ir1+0) * src1_stride);                       \
                const uint8_t * restrict src1_col1 = (const uint8_t *) (src1_data + (ir1+1) * src1_stride);                       \
                                                                                                                                  \
                float * restrict dst_row0_q = (float *) (dst_q->data + ((ir1+0) * dst_q_row_size));                               \
                float * restrict dst_row1_q = (float *) (dst_q->data + ((ir1+1) * dst_q_row_size));                               \
                float * dst_ptr0_q = &dst_row0_q[ct * 32];                                                                        \
                float * dst_ptr1_q = &dst_row1_q[ct * 32];                                                                        \
                                                                                                                                  \
                DOT_2X2(ne10, dst_ptr0_q, dst_ptr1_q, w_tile_q, src1_col0, src1_col1, valid_rows, NULL, NULL);                    \
            }                                                                                                                     \
                                                                                                                                  \
            for (; ir1 < src1_nrows; ++ir1) {                                                                                     \
                const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);                            \
                                                                                                                                  \
                float * restrict dst_row_q = (float *) (dst_q->data + (ir1 * dst_q_row_size));                                    \
                float * dst_ptr_q = &dst_row_q[ct * 32];                                                                          \
                                                                                                                                  \
                DOT_2X1(ne10, dst_ptr_q, w_tile_q, src1_col, valid_rows, NULL);                                                   \
            }                                                                                                                     \
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);                                                                 \
                                                                                                                                  \
            if (push_ct < ct_end_q) {                                                                                             \
                dma_queue_push(dma_queue, dma_make_ptr((uint8_t *)w_tile_q, src3_row + push_ct * tile_row_stride),                \
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_a);                                             \
                push_ct++;                                                                                                        \
            }                                                                                                                     \
        }                                                                                                                         \
    }                                                                                                                             \
}

#define MATMUL_FFN_2D_REPACKED_IMPL(SUFFIX, TILE_SIZE, DOT_2X2, DOT_2X1)                                                          \
static void hvx_mm_ffn_2d_repacked_##SUFFIX(unsigned int nth, unsigned int ith, void * data) {                                    \
    struct htp_mm_context * mmctx = data;                                                                                         \
    struct htp_ops_context * octx = mmctx->octx;                                                                                  \
                                                                                                                                  \
    const struct htp_tensor * restrict src0 = octx->src[0]; /* Wgate */                                                           \
    const struct htp_tensor * restrict src1 = octx->src[1]; /* y */                                                               \
    const struct htp_tensor * restrict src2 = octx->src[2]; /* Wup */                                                             \
    const struct htp_tensor * restrict dst_gate = octx->dsts[0];                                                                  \
    const struct htp_tensor * restrict dst_up = octx->dsts[1];                                                                    \
                                                                                                                                  \
    const uint32_t ne00 = src0->ne[0];                                                                                            \
    const uint32_t ne01 = src0->ne[1];                                                                                            \
    const uint32_t ne10 = src1->ne[0];                                                                                            \
    const uint32_t src1_nrows = src1->ne[1] * src1->ne[2] * src1->ne[3];                                                          \
                                                                                                                                  \
    const size_t dst_row_size  = dst_gate->nb[1];                                                                                 \
    const size_t src1_stride = mmctx->vtcm_src1_stride;                                                                           \
                                                                                                                                  \
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;                                 \
    uint8_t * restrict vtcm_src2_ptr = mmctx->vtcm_src2 + mmctx->vtcm_src2_size_per_thread * ith;                                 \
    uint8_t * restrict src1_data = mmctx->vtcm_src1;                                                                              \
                                                                                                                                  \
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];                                                                        \
                                                                                                                                  \
    const uint8_t * restrict src0_row = (const uint8_t *) src0->data;                                                             \
    const uint8_t * restrict src2_row = (const uint8_t *) src2->data;                                                             \
                                                                                                                                  \
    const uint32_t tile_size = TILE_SIZE;                                                                                         \
    const uint32_t aligned_tile_size = hex_align_up(tile_size, 128);                                                              \
                                                                                                                                  \
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;                      \
    const uint32_t n_prefetch = kparams->n_prefetch;                                                                              \
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);                         \
                                                                                                                                  \
    uint32_t n_k_tiles_w = ne00 / 32;                                                                                             \
    uint32_t n_k_tiles_a = ne10 / 32;                                                                                             \
    uint32_t tile_row_stride = n_k_tiles_w * tile_size;                                                                           \
    uint32_t tile_row_transfer_size_aligned = n_k_tiles_a * aligned_tile_size;                                                    \
    dma_queue * dma_queue = octx->ctx->dma[ith];                                                                                  \
                                                                                                                                  \
    const uint32_t src0_nrows = ne01 * src0->ne[2] * src0->ne[3];                                                                 \
    const uint32_t src0_start_row = mmctx->src0_nrows_per_thread * ith;                                                           \
    const uint32_t src0_end_row   = MIN(src0_start_row + mmctx->src0_nrows_per_thread, src0_nrows);                               \
                                                                                                                                  \
    uint32_t ct_start = src0_start_row / 32;                                                                                      \
    uint32_t ct_end   = (src0_end_row + 31) / 32;                                                                                 \
                                                                                                                                  \
    uint32_t push_ct = ct_start;                                                                                                  \
    if (src0_start_row < src0_end_row) {                                                                                          \
        for (uint32_t d = 0; d < n_prefetch && push_ct < ct_end; d++, push_ct++) {                                                \
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + d * tile_row_transfer_size_aligned,                            \
                           src0_row + push_ct * tile_row_stride), aligned_tile_size, tile_size, tile_size, n_k_tiles_a);          \
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src2_ptr + d * tile_row_transfer_size_aligned,                            \
                           src2_row + push_ct * tile_row_stride), aligned_tile_size, tile_size, tile_size, n_k_tiles_a);          \
        }                                                                                                                         \
    }                                                                                                                             \
                                                                                                                                  \
    hvx_mm_run_quant_task(mmctx, ith);                                                                                            \
                                                                                                                                  \
    if (src0_start_row >= src0_end_row) {                                                                                         \
        return;                                                                                                                   \
    }                                                                                                                             \
                                                                                                                                  \
    for (uint32_t ct = ct_start; ct < ct_end; ct++) {                                                                             \
        const uint8_t * w_tile_gate = dma_queue_pop(dma_queue).dst;                                                               \
        const uint8_t * w_tile_up   = dma_queue_pop(dma_queue).dst;                                                               \
                                                                                                                                  \
        int valid_rows = (int)ne01 - (int)(ct * 32);                                                                              \
        valid_rows = MIN(32, MAX(0, valid_rows));                                                                                 \
                                                                                                                                  \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);                                                                    \
        uint32_t ir1 = 0;                                                                                                         \
        for (; ir1 + 1 < src1_nrows; ir1 += 2) {                                                                                  \
            const uint8_t * restrict src1_col0 = (const uint8_t *) (src1_data + (ir1+0) * src1_stride);                           \
            const uint8_t * restrict src1_col1 = (const uint8_t *) (src1_data + (ir1+1) * src1_stride);                           \
                                                                                                                                  \
            float * restrict dst_row0_gate = (float *) (dst_gate->data + ((ir1+0) * dst_row_size));                               \
            float * restrict dst_row1_gate = (float *) (dst_gate->data + ((ir1+1) * dst_row_size));                               \
            float * dst_ptr0_gate = &dst_row0_gate[ct * 32];                                                                      \
            float * dst_ptr1_gate = &dst_row1_gate[ct * 32];                                                                      \
                                                                                                                                  \
            float * restrict dst_row0_up = (float *) (dst_up->data + ((ir1+0) * dst_row_size));                                   \
            float * restrict dst_row1_up = (float *) (dst_up->data + ((ir1+1) * dst_row_size));                                   \
            float * dst_ptr0_up = &dst_row0_up[ct * 32];                                                                          \
            float * dst_ptr1_up = &dst_row1_up[ct * 32];                                                                          \
                                                                                                                                  \
            DOT_2X2(ne10, dst_ptr0_gate, dst_ptr1_gate, w_tile_gate, src1_col0, src1_col1, valid_rows, NULL, NULL);               \
            DOT_2X2(ne10, dst_ptr0_up, dst_ptr1_up, w_tile_up, src1_col0, src1_col1, valid_rows, NULL, NULL);                     \
        }                                                                                                                         \
                                                                                                                                  \
        for (; ir1 < src1_nrows; ++ir1) {                                                                                         \
            const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);                                \
                                                                                                                                  \
            float * restrict dst_row_gate = (float *) (dst_gate->data + (ir1 * dst_row_size));                                    \
            float * dst_ptr_gate = &dst_row_gate[ct * 32];                                                                        \
                                                                                                                                  \
            float * restrict dst_row_up = (float *) (dst_up->data + (ir1 * dst_row_size));                                        \
            float * dst_ptr_up = &dst_row_up[ct * 32];                                                                            \
                                                                                                                                  \
            DOT_2X1(ne10, dst_ptr_gate, w_tile_gate, src1_col, valid_rows, NULL);                                                 \
            DOT_2X1(ne10, dst_ptr_up, w_tile_up, src1_col, valid_rows, NULL);                                                     \
        }                                                                                                                         \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);                                                                     \
                                                                                                                                  \
        if (push_ct < ct_end) {                                                                                                   \
            dma_queue_push(dma_queue, dma_make_ptr((uint8_t *)w_tile_gate, src0_row + push_ct * tile_row_stride),                 \
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_a);                                                 \
            dma_queue_push(dma_queue, dma_make_ptr((uint8_t *)w_tile_up, src2_row + push_ct * tile_row_stride),                   \
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_a);                                                 \
            push_ct++;                                                                                                            \
        }                                                                                                                         \
    }                                                                                                                             \
}

MATMUL_2D_REPACKED_IMPL(q4_0,       576,  tiled_vec_dot_q4_0_32x2,  tiled_vec_dot_q4_0_32x1)
MATMUL_2D_REPACKED_IMPL(q4_1,       640,  tiled_vec_dot_q4_1_32x2,  tiled_vec_dot_q4_1_32x1)
MATMUL_2D_REPACKED_IMPL(q8_0,       1088, tiled_vec_dot_q8_0_32x2,  tiled_vec_dot_q8_0_32x1)
MATMUL_2D_REPACKED_IMPL(iq4nl,      576,  tiled_vec_dot_iq4nl_32x2, tiled_vec_dot_iq4nl_32x1)
MATMUL_2D_REPACKED_IMPL(mxfp4,      544,  tiled_vec_dot_mxfp4_32x2, tiled_vec_dot_mxfp4_32x1)

MATMUL_2D_REPACKED_IMPL(q4_0_flat,  576,  flat_vec_dot_q4_0_32x2,   flat_vec_dot_q4_0_32x1)
MATMUL_2D_REPACKED_IMPL(q4_1_flat,  640,  flat_vec_dot_q4_1_32x2,   flat_vec_dot_q4_1_32x1)
MATMUL_2D_REPACKED_IMPL(q8_0_flat,  1088, flat_vec_dot_q8_0_32x2,   flat_vec_dot_q8_0_32x1)
MATMUL_2D_REPACKED_IMPL(iq4nl_flat, 576,  flat_vec_dot_iq4nl_32x2,  flat_vec_dot_iq4nl_32x1)
MATMUL_2D_REPACKED_IMPL(mxfp4_flat, 544,  flat_vec_dot_mxfp4_32x2,  flat_vec_dot_mxfp4_32x1)

#define QUANTIZE_IMPL(name, log_name, kernel_fn, dst_row_size_expr)                                        \
static void name(unsigned int nth, unsigned int ith, void * data) {                                        \
    struct htp_mm_context * mmctx = data;                                                                  \
    struct htp_ops_context * octx = mmctx->octx;                                                           \
    const struct htp_tensor * src = octx->src[1];                                                          \
    const uint32_t ne0 = src->ne[0];                                                                       \
    const uint32_t ne1 = src->ne[1];                                                                       \
    const uint32_t ne2 = src->ne[2];                                                                       \
    const uint32_t ne3 = src->ne[3];                                                                       \
    const uint32_t nrows = ne1 * ne2 * ne3;                                                                \
    const uint32_t nrows_per_thread = mmctx->n_quant_rows_per_thread;                                      \
                                                                                                           \
    const uint32_t ir_first = nrows_per_thread * ith;                                                      \
    if (ir_first >= nrows) {                                                                               \
        return;                                                                                            \
    }                                                                                                      \
                                                                                                           \
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];                                                 \
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, ir_first);                                        \
                                                                                                           \
    uint8_t * restrict dst = mmctx->vtcm_src1;                                                             \
    const uint32_t ir_last = MIN(ir_first + nrows_per_thread, nrows);                                      \
    const size_t src_row_size = src->nb[1];                                                                \
    const size_t dst_row_size = (dst_row_size_expr);                                                       \
    const uint8_t * restrict src_data = (const uint8_t *) src->data + (src_row_size * ir_first);           \
    uint8_t * restrict dst_data = (uint8_t *) dst + (dst_row_size * ir_first);                             \
    uint8_t * restrict tmp_data = (uint8_t *) mmctx->vtcm_dst + (mmctx->vtcm_dst_size_per_thread * ith);   \
    kernel_fn(src_data, dst_data, tmp_data, ne0, ir_last - ir_first, src_row_size, dst_row_size);          \
                                                                                                           \
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, ir_first);                                         \
}

QUANTIZE_IMPL(quantize_f32_q8_0_tiled, "quantize-f32-q8_0_tiled", quantize_f32_q8_0_tiled_kernel, htp_mm_q8_0_tiled_row_size(ne0))
QUANTIZE_IMPL(quantize_f32_q8_1_tiled, "quantize-f32-q8_1_tiled", quantize_f32_q8_1_tiled_kernel, htp_mm_q8_1_tiled_row_size(ne0))
QUANTIZE_IMPL(quantize_f32_q8_0_flat,  "quantize-f32-q8_0_flat",  quantize_f32_q8_0_flat_kernel,  htp_mm_q8_0_flat_row_size(ne0))
QUANTIZE_IMPL(quantize_f32_q8_1_flat,  "quantize-f32-q8_1_flat",  quantize_f32_q8_1_flat_kernel,  htp_mm_q8_1_flat_row_size(ne0))
QUANTIZE_IMPL(quantize_f32_f32_flat,   "quantize-f32-f32",        quantize_f32_f32_flat_kernel,   mmctx->vtcm_src1_stride)
QUANTIZE_IMPL(quantize_f32_f16_flat,   "quantize-f32-f16",        quantize_f32_f16_flat_kernel,   mmctx->vtcm_src1_stride)
QUANTIZE_IMPL(quantize_f16_f16_flat,   "quantize-f16-f16",        quantize_f16_f16_flat_kernel,   mmctx->vtcm_src1_stride)

static void quantize_f32_q8_0_tiled_block(unsigned int nth, unsigned int ith, void * data) {
    struct htp_mm_context * mmctx = data;
    struct htp_ops_context * octx = mmctx->octx;
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    const struct htp_tensor * src = octx->src[1];

    quantize_f32_q8_0_tiled_block_kernel(
        (const float *) src->data,
        mmctx->vtcm_src1,
        (uint8_t *) mmctx->vtcm_dst + (mmctx->vtcm_dst_size_per_thread * ith),
        src->ne[0],
        mmctx->quant_ib_first[ith],
        mmctx->quant_ib_last[ith],
        src->nb[1],
        htp_mm_q8_0_tiled_row_size(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);
}

static void quantize_f32_q8_1_tiled_block(unsigned int nth, unsigned int ith, void * data) {
    struct htp_mm_context * mmctx = data;
    struct htp_ops_context * octx = mmctx->octx;
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    const struct htp_tensor * src = octx->src[1];

    quantize_f32_q8_1_tiled_block_kernel(
        (const float *) src->data,
        mmctx->vtcm_src1,
        (uint8_t *) mmctx->vtcm_dst + (mmctx->vtcm_dst_size_per_thread * ith),
        src->ne[0],
        mmctx->quant_ib_first[ith],
        mmctx->quant_ib_last[ith],
        src->nb[1],
        htp_mm_q8_1_tiled_row_size(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);
}

MATVEC_2D_REPACKED_IMPL(q4_0,       576,  tiled_vec_dot_q4_0_32x1)
MATVEC_2D_REPACKED_IMPL(q4_1,       640,  tiled_vec_dot_q4_1_32x1)
MATVEC_2D_REPACKED_IMPL(q8_0,       1088, tiled_vec_dot_q8_0_32x1)
MATVEC_2D_REPACKED_IMPL(iq4nl,      576,  tiled_vec_dot_iq4nl_32x1)
MATVEC_2D_REPACKED_IMPL(mxfp4,      544,  tiled_vec_dot_mxfp4_32x1)

MATVEC_2D_REPACKED_IMPL(q4_0_flat,  576,  flat_vec_dot_q4_0_32x1)
MATVEC_2D_REPACKED_IMPL(q4_1_flat,  640,  flat_vec_dot_q4_1_32x1)
MATVEC_2D_REPACKED_IMPL(q8_0_flat,  1088, flat_vec_dot_q8_0_32x1)
MATVEC_2D_REPACKED_IMPL(iq4nl_flat, 576,  flat_vec_dot_iq4nl_32x1)
MATVEC_2D_REPACKED_IMPL(mxfp4_flat, 544,  flat_vec_dot_mxfp4_32x1)


MATMUL_QKV_2D_REPACKED_IMPL(q4_0,       576,  tiled_vec_dot_q4_0_32x2,  tiled_vec_dot_q4_0_32x1)
MATMUL_QKV_2D_REPACKED_IMPL(q4_1,       640,  tiled_vec_dot_q4_1_32x2,  tiled_vec_dot_q4_1_32x1)
MATMUL_QKV_2D_REPACKED_IMPL(q8_0,       1088, tiled_vec_dot_q8_0_32x2,  tiled_vec_dot_q8_0_32x1)
MATMUL_QKV_2D_REPACKED_IMPL(iq4nl,      576,  tiled_vec_dot_iq4nl_32x2, tiled_vec_dot_iq4nl_32x1)
MATMUL_QKV_2D_REPACKED_IMPL(mxfp4,      544,  tiled_vec_dot_mxfp4_32x2, tiled_vec_dot_mxfp4_32x1)

MATMUL_QKV_2D_REPACKED_IMPL(q4_0_flat,  576,  flat_vec_dot_q4_0_32x2,   flat_vec_dot_q4_0_32x1)
MATMUL_QKV_2D_REPACKED_IMPL(q4_1_flat,  640,  flat_vec_dot_q4_1_32x2,   flat_vec_dot_q4_1_32x1)
MATMUL_QKV_2D_REPACKED_IMPL(q8_0_flat,  1088, flat_vec_dot_q8_0_32x2,   flat_vec_dot_q8_0_32x1)
MATMUL_QKV_2D_REPACKED_IMPL(iq4nl_flat, 576,  flat_vec_dot_iq4nl_32x2,  flat_vec_dot_iq4nl_32x1)
MATMUL_QKV_2D_REPACKED_IMPL(mxfp4_flat, 544,  flat_vec_dot_mxfp4_32x2,  flat_vec_dot_mxfp4_32x1)


MATMUL_FFN_2D_REPACKED_IMPL(q4_0,       576,  tiled_vec_dot_q4_0_32x2,  tiled_vec_dot_q4_0_32x1)
MATMUL_FFN_2D_REPACKED_IMPL(q4_1,       640,  tiled_vec_dot_q4_1_32x2,  tiled_vec_dot_q4_1_32x1)
MATMUL_FFN_2D_REPACKED_IMPL(q8_0,       1088, tiled_vec_dot_q8_0_32x2,  tiled_vec_dot_q8_0_32x1)
MATMUL_FFN_2D_REPACKED_IMPL(iq4nl,      576,  tiled_vec_dot_iq4nl_32x2, tiled_vec_dot_iq4nl_32x1)
MATMUL_FFN_2D_REPACKED_IMPL(mxfp4,      544,  tiled_vec_dot_mxfp4_32x2, tiled_vec_dot_mxfp4_32x1)

MATMUL_FFN_2D_REPACKED_IMPL(q4_0_flat,  576,  flat_vec_dot_q4_0_32x2,   flat_vec_dot_q4_0_32x1)
MATMUL_FFN_2D_REPACKED_IMPL(q4_1_flat,  640,  flat_vec_dot_q4_1_32x2,   flat_vec_dot_q4_1_32x1)
MATMUL_FFN_2D_REPACKED_IMPL(q8_0_flat,  1088, flat_vec_dot_q8_0_32x2,   flat_vec_dot_q8_0_32x1)
MATMUL_FFN_2D_REPACKED_IMPL(iq4nl_flat, 576,  flat_vec_dot_iq4nl_32x2,  flat_vec_dot_iq4nl_32x1)
MATMUL_FFN_2D_REPACKED_IMPL(mxfp4_flat, 544,  flat_vec_dot_mxfp4_32x2,  flat_vec_dot_mxfp4_32x1)

static void hvx_mm_2d(unsigned int nth, unsigned int ith, void * data) {
    htp_matmul_preamble;

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_prefetch = kparams->n_prefetch;
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    const uint32_t prefetch_mask = n_prefetch - 1;

    const uint32_t src0_nrows = ne01 * ne02 * ne03;  // src0 rows
    const uint32_t src1_nrows = ne11 * ne12 * ne13;  // src1 rows

    const uint32_t src0_start_row  = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);
    const uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    const size_t dst_row_size  = nb1;
    const size_t src0_row_size = nb01;
    const size_t src1_row_size = nb11;

    const size_t src0_stride = mmctx->vtcm_src0_stride;
    const size_t src1_stride = mmctx->vtcm_src1_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * restrict vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_size_per_thread  * ith;
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict src1_data     = mmctx->vtcm_src1;

    const uint8_t * restrict src0_row = (const uint8_t *) src0->data;

    // Prefill vtcm with src0 rows
    if (src0_start_row < src0_end_row) {
        for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
            const int is0 = (ir0 - src0_start_row);
            if (is0 >= (int)n_prefetch) {
                break;
            }
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }
    }

    hvx_mm_run_quant_task(mmctx, ith);

    if (src0_start_row >= src0_end_row) {
        return;
    }

    // Process src0 rows
    for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        const uint8_t * ss0 = dma_queue_pop(dma_queue).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        // Process src1 columns in pairs (2×2 tiling)
        uint32_t ir1 = 0;
        for (; ir1 + 1 < src1_nrows; ir1 += 2) {
            const uint8_t * restrict src1_col0 = (const uint8_t *) (src1_data + (ir1+0) * src1_stride);
            const uint8_t * restrict src1_col1 = (const uint8_t *) (src1_data + (ir1+1) * src1_stride);
            float * restrict dst_row0 = (float *) (dst->data + ((ir1+0) * dst_row_size));
            float * restrict dst_row1 = (float *) (dst->data + ((ir1+1) * dst_row_size));
            mmctx->vec_dot_2x2(ne00, &dst_row0[ir0], &dst_row1[ir0], ss0, ss0 + src0_stride, src1_col0, src1_col1);
        }

        // Handle remaining src1 rows (fallback to 2×1)
        for (; ir1 < src1_nrows; ++ir1) {
            const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);
            float * restrict dst_row          = (float *) (dst->data + (ir1 * dst_row_size));
            mmctx->vec_dot_2x1(ne00, &dst_row[ir0], ss0, ss0 + src0_stride, src1_col);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

        // Prefetch next (n + vtcm_nrows) row
        const int pr0 = (ir0 + n_prefetch);
        const int is0 = (pr0 - src0_start_row) & prefetch_mask;
        if (pr0 < src0_end_row_x2) {
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + is0 * src0_stride, src0_row + pr0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }
    }

    // Process the last row (if any)
    if (src0_end_row != src0_end_row_x2) {
        uint32_t  ir0 = src0_end_row_x2;
        const int is0 = (ir0 - src0_start_row) & prefetch_mask;
        dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                       src0_stride, src0_row_size, src0_row_size, 1);
        const uint8_t * ss0 = dma_queue_pop(dma_queue).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        #pragma unroll(2)
        for (uint32_t ir1 = 0; ir1 < src1_nrows; ++ir1) {
            const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);
            float * restrict dst_row          = (float *) (dst->data + (ir1 * dst_row_size));
            mmctx->vec_dot_1x1(ne00, &dst_row[ir0], ss0, src1_col);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
    }
    if (src2) {
        hvx_tensor_add_f32_grid(dst, src2, 0, src1_nrows, src0_start_row, src0_end_row, &kparams->div_ne12_ne1, &kparams->div_ne1);
    }
}

static void hvx_mv_2d(unsigned int nth, unsigned int ith, void * data) {
    htp_matmul_preamble;

    const uint32_t src0_nrows = ne01;

    const uint32_t src0_start_row  = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    const size_t dst_row_size  = nb1;
    const size_t src0_row_size = nb01;
    const size_t src1_row_size = nb11;

    const size_t src0_stride = mmctx->vtcm_src0_stride;
    const size_t src1_stride = mmctx->vtcm_src1_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_size_per_thread  * ith;
    uint8_t * vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * src1_data     = mmctx->vtcm_src1;

    float * tmp = (float *) vtcm_dst_ptr;

    const uint8_t * restrict src0_row = (const uint8_t *) src0->data;
    const uint8_t * restrict src1_col = (const uint8_t *) src1_data;
    float * restrict dst_col          = (float *) dst->data;

    const uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_prefetch = kparams->n_prefetch;
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    const uint32_t prefetch_mask = n_prefetch - 1;

    // Prefill vtcm with 2x src0 rows
    if (src0_start_row < src0_end_row) {
        if (src2) {
            float * vtcm_src2_ptr = (float *) mmctx->vtcm_src2 + src0_start_row;
            const float * src2_ptr = (const float *) src2->data + src0_start_row;
            int slice_size = (int)src0_end_row - (int)src0_start_row;
            if (slice_size > 0) {
                dma_queue_push(dma_queue, dma_make_ptr(vtcm_src2_ptr, src2_ptr),
                               slice_size * sizeof(float), slice_size * sizeof(float), slice_size * sizeof(float), 1);
                dma_queue_pop_nowait(dma_queue);
            }
        }
        for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
            const uint32_t is0 = (ir0 - src0_start_row);
            if (is0 >= n_prefetch) {
                break;
            }
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }
    }

    hvx_mm_run_quant_task(mmctx, ith);

    if (src0_start_row >= src0_end_row) {
        return;
    }

    // Process src0 rows
    for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        const uint8_t * ss0 = dma_queue_pop(dma_queue).dst;
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        mmctx->vec_dot_2x1(ne00, &tmp[ir0 - src0_start_row], ss0, ss0 + src0_stride, src1_col);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

        // Prefetch next (n + vtcm_nrows) row
        const uint32_t pr0 = (ir0 + n_prefetch);
        const uint32_t is0 = (pr0 - src0_start_row) & prefetch_mask;
        if (pr0 < src0_end_row_x2) {
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + is0 * src0_stride, src0_row + pr0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }
    }

    // Process the last row (if any)
    if (src0_end_row != src0_end_row_x2) {
        const uint32_t ir0 = src0_end_row_x2;
        const uint32_t is0 = (ir0 - src0_start_row) & prefetch_mask;
        dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                       src0_stride, src0_row_size, src0_row_size, 1);
        const uint8_t * ss0 = dma_queue_pop(dma_queue).dst;
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        mmctx->vec_dot_1x1(ne00, &tmp[ir0 - src0_start_row], ss0, src1_col);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
    }

    int copy_cnt = src0_end_row - src0_start_row;
    if (copy_cnt > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, src0_end_row);
        if (src2) {
            hvx_add_f32_uaa((uint8_t *) &dst_col[src0_start_row],
                            (const uint8_t *) tmp,
                            (const uint8_t *) ((const float *) mmctx->vtcm_src2 + src0_start_row),
                            copy_cnt);
        } else {
            hvx_copy_f32_ua((uint8_t *) &dst_col[src0_start_row], (uint8_t *) tmp, copy_cnt);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, src0_end_row);
    }
}

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id) * mmctx->mapping_stride + (i1)]

// EXPERIMENTAL: convert 32 raw GGUF Q4_0 rows into the existing HTP tiled Q4_0
// layout entirely in VTCM. This is a pure byte-layout transform; scales are copied bitwise.
#define HTP_RAW_Q4_0_BLOCK_BYTES 18u

static inline uint8_t htp_raw_q4_0_get_nibble(const uint8_t * qs, uint32_t q) {
    if (q < 16) {
        return qs[q] & 0x0f;
    }
    return qs[q - 16] >> 4;
}

static inline uint32_t htp_dbg_v121_fnv1a(
        const uint8_t * data, size_t n) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; ++i) {
        h ^= data[i];
        h *= 16777619u;
    }
    return h;
}

static inline uint32_t htp_dbg_v121_load_u32(
        const uint8_t * data, size_t available) {
    uint32_t value = 0;
    memcpy(&value, data, MIN(available, sizeof(value)));
    return value;
}

_Static_assert(HTP_MM_WEIGHT_TILE_SIZE_Q4_0 == 576, "unexpected Q4_0 logical tile size");
_Static_assert(HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q4_0 == 640, "unexpected Q4_0 aligned tile stride");

static void htp_raw_q4_0_32rows_to_tiled(
        const uint8_t * restrict raw,
        size_t raw_row_size,
        uint8_t * restrict tiled,
        uint32_t k,
        uint32_t valid_rows) {
    const uint32_t n_k_tiles = k / 32;

    for (uint32_t kt = 0; kt < n_k_tiles; ++kt) {
        // v10.10 correctness fix:
        // The Q4_0 dot kernel advances weight tiles at the aligned 640-byte
        // stride, while this raw converter previously packed them every 576
        // bytes. kt=0 therefore looked correct in our hashes, but kt>=1 was
        // read from the wrong address.
        uint8_t * tile = tiled + kt * HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q4_0;

        // Logical Q4_0 data occupies 576 bytes; clear the 64-byte alignment tail.
        memset(tile, 0, HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q4_0);

        for (uint32_t cp = 0; cp < 16; ++cp) {
            for (uint32_t row = 0; row < 32; ++row) {
                uint8_t packed = 0x88; // zero in q4_0's biased nibble representation
                if (row < valid_rows) {
                    const uint8_t * block = raw + row * raw_row_size + kt * HTP_RAW_Q4_0_BLOCK_BYTES;
                    const uint8_t * qs = block + 2; // fp16 scale occupies first 2 bytes
                    const uint8_t q0 = htp_raw_q4_0_get_nibble(qs, 2 * cp + 0);
                    const uint8_t q1 = htp_raw_q4_0_get_nibble(qs, 2 * cp + 1);
                    packed = (uint8_t) (q0 | (q1 << 4));
                }
                tile[cp * 32 + row] = packed;
            }
        }

        uint8_t * scale_dst = tile + 512;
        for (uint32_t row = 0; row < 32; ++row) {
            if (row < valid_rows) {
                const uint8_t * block = raw + row * raw_row_size + kt * HTP_RAW_Q4_0_BLOCK_BYTES;
                scale_dst[2 * row + 0] = block[0];
                scale_dst[2 * row + 1] = block[1];
            } else {
                scale_dst[2 * row + 0] = 0;
                scale_dst[2 * row + 1] = 0;
            }
        }
    }
}

// v10.24: parallelize the low-memory raw-Q4_0 layout transform by assigning
// independent 32-row output tiles to the HTP worker pool.  Each worker writes
// a disjoint [row-tile, all-K-tiles] range, so the byte layout is identical to
// htp_raw_q4_0_32rows_to_tiled() and no synchronization is needed inside the
// transform.
typedef struct {
    const uint8_t * raw;
    size_t          raw_row_size;
    uint8_t       * tiled;
    size_t          n_rows;
    uint32_t        k;
    size_t          tiled_32_rows;
} htp_raw_q4_0_tiled_task_state_t;

static void htp_raw_q4_0_to_tiled_worker(
        unsigned int n,
        unsigned int i,
        void * data) {
    htp_raw_q4_0_tiled_task_state_t * state = data;
    const size_t n_row_tiles = hmx_ceil_div(state->n_rows, 32);

    for (size_t rt = i; rt < n_row_tiles; rt += n) {
        const size_t row = rt * 32;
        const uint32_t valid_rows = (uint32_t) hex_smin(state->n_rows - row, 32);
        htp_raw_q4_0_32rows_to_tiled(
            state->raw + row * state->raw_row_size,
            state->raw_row_size,
            state->tiled + rt * state->tiled_32_rows,
            state->k,
            valid_rows);
    }
}

static void htp_raw_q4_0_to_tiled_threaded(
        struct htp_context * ctx,
        const uint8_t * raw,
        size_t raw_row_size,
        uint8_t * tiled,
        size_t n_rows,
        uint32_t k,
        uint32_t n_threads) {
    const size_t n_row_tiles = hmx_ceil_div(n_rows, 32);
    if (n_row_tiles == 0) {
        return;
    }

    htp_raw_q4_0_tiled_task_state_t state = {
        .raw             = raw,
        .raw_row_size    = raw_row_size,
        .tiled           = tiled,
        .n_rows          = n_rows,
        .k               = k,
        .tiled_32_rows   = (size_t) (k / 32) * HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q4_0,
    };

    const uint32_t n_tasks = (uint32_t) hex_smin(n_row_tiles, n_threads);
    if (n_tasks <= 1) {
        htp_raw_q4_0_to_tiled_worker(1, 0, &state);
    } else {
        worker_pool_run_func(ctx->worker_pool, htp_raw_q4_0_to_tiled_worker, &state, n_tasks);
    }
}

static void hvx_mm_id_raw_q4_0(unsigned int nth, unsigned int ith, void * data) {
    htp_matmul_preamble;

    const struct htp_tensor * restrict ids = octx->src[2];
    const uint32_t src0_nrows      = ne01;
    const uint32_t src1_nrows      = ne11;
    const uint32_t src0_start_row  = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);

    hvx_mm_run_quant_task(mmctx, ith);
    if (src0_start_row >= src0_end_row) {
        return;
    }

    // v10.9: hvx_mm_run_quant_task() does not return until quant_barrier == 0,
    // so vtcm_src1 is fully produced here. Capture only ith==0 to avoid races.
    if (ith == 0 && mmctx->vtcm_src1 && mmctx->vtcm_src1_stride >= 1152) {
        const uint8_t * q = mmctx->vtcm_src1;

        uint32_t fnv_tile = 2166136261u;
        for (size_t di = 0; di < 1152; ++di) {
            fnv_tile ^= q[di];
            fnv_tile *= 16777619u;
        }
        mmctx->dbg_v109_quant_fnv = fnv_tile;

        uint32_t fnv_row = 2166136261u;
        for (size_t di = 0; di < mmctx->vtcm_src1_stride; ++di) {
            fnv_row ^= q[di];
            fnv_row *= 16777619u;
        }
        mmctx->dbg_v109_row_fnv = fnv_row;

        mmctx->dbg_v109_q8_head =
            ((uint32_t) q[0]) |
            ((uint32_t) q[1] << 8) |
            ((uint32_t) q[2] << 16) |
            ((uint32_t) q[3] << 24);

        const uint8_t * scale = q + 1024;
        mmctx->dbg_v109_scale_head =
            ((uint32_t) scale[0]) |
            ((uint32_t) scale[1] << 8) |
            ((uint32_t) scale[2] << 16) |
            ((uint32_t) scale[3] << 24);

        // The captured v10.7 mapping for the first selected expert resolves to
        // the same src1 row used below. At this point that mapping is not yet
        // claimed, so record row-0 source diagnostics; v10.7 will report ir1/rm2
        // and lets us verify that this is the row actually consumed.
        const float * src_f = (const float *) octx->src[1]->data;
        union { float f; uint32_t u; } f0, fmax;
        f0.f = src_f[0];
        float maxabs = 0.0f;
        const uint32_t ncheck = ne10 < 128 ? ne10 : 128;
        for (uint32_t di = 0; di < ncheck; ++di) {
            const float av = fabsf(src_f[di]);
            if (av > maxabs) {
                maxabs = av;
            }
        }
        fmax.f = maxabs;
        mmctx->dbg_v109_src_f0_bits  = f0.u;
        mmctx->dbg_v109_src_max_bits = fmax.u;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    const uint32_t n_as = ne02;
    const uint32_t * matrix_row_counts = mmctx->matrix_row_counts;
    const struct mmid_row_mapping * matrix_rows = mmctx->matrix_rows;

    const size_t src1_stride = mmctx->vtcm_src1_stride;
    uint8_t * restrict thread_vtcm = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    const size_t raw_row_size = nb01;
    const size_t raw_32_size = htp_mm_round_up(32 * raw_row_size, 256);
    uint8_t * restrict raw_buf = thread_vtcm;
    uint8_t * restrict tiled_buf = thread_vtcm + raw_32_size;
    uint8_t * restrict src1_data = mmctx->vtcm_src1;

    for (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
        const int32_t cne1 = matrix_row_counts[cur_a];
        if (cne1 == 0) {
            continue;
        }

        const uint8_t * src0_expert = (const uint8_t *) src0->data + cur_a * nb02;
        const uint32_t ct_start = src0_start_row / 32;
        const uint32_t ct_end   = (src0_end_row + 31) / 32;

        for (uint32_t ct = ct_start; ct < ct_end; ++ct) {
            uint32_t valid_rows = ne01 - ct * 32;
            valid_rows = MIN(valid_rows, 32u);

            // DMA raw GGUF rows into VTCM.
            dma_queue_push(
                dma_queue,
                dma_make_ptr(raw_buf, src0_expert + (size_t) ct * 32 * raw_row_size),
                raw_row_size, raw_row_size, raw_row_size, valid_rows);
            (void) dma_queue_pop(dma_queue);

            // Convert only this working set; no model-sized REPACK allocation.
            htp_raw_q4_0_32rows_to_tiled(raw_buf, raw_row_size, tiled_buf, ne00, valid_rows);

            // v10.3: capture the first selected expert into this op's mmctx.
            // The result is copied into the shared kernel_params blob only after all
            // worker threads finish, so it cannot disturb execution parameters.
            // v10.6: Host reference hashes ct=0 (rows 0..31), so only the
            // DSP worker processing ct=0 may claim the debug slot.
            const bool dbg_ct_match = mmctx->dbg_v112_enabled
                ? (ct == mmctx->dbg_v112_want_ct)
                : (ct == 0);
            const bool dbg_expert_match = mmctx->dbg_v112_enabled
                ? (mmctx->dbg_v112_want_expert < 0 ||
                   cur_a == (uint32_t) mmctx->dbg_v112_want_expert)
                : true;
            const bool dbg_cid_exists = mmctx->dbg_v112_enabled
                ? (mmctx->dbg_v112_want_cid < (uint32_t) cne1)
                : true;

            if (dbg_ct_match && dbg_expert_match && dbg_cid_exists &&
                __sync_bool_compare_and_swap(&mmctx->dbg_v103_claimed, 0, 1)) {
                const size_t raw_bytes = (size_t) valid_rows * raw_row_size;

                uint32_t raw_fnv = 2166136261u;
                for (size_t di = 0; di < raw_bytes; ++di) {
                    raw_fnv ^= raw_buf[di];
                    raw_fnv *= 16777619u;
                }

                uint32_t tile_fnv = 2166136261u;
                for (size_t di = 0; di < HTP_MM_WEIGHT_TILE_SIZE_Q4_0; ++di) {
                    tile_fnv ^= tiled_buf[di];
                    tile_fnv *= 16777619u;
                }

                mmctx->dbg_v103_expert     = cur_a;
                mmctx->dbg_v103_raw_fnv    = raw_fnv;
                mmctx->dbg_v103_tile_fnv   = tile_fnv;
                mmctx->dbg_v103_src_off    = cur_a * nb02;
                mmctx->dbg_v106_ct         = ct;
                mmctx->dbg_v106_ith        = ith;
                mmctx->dbg_v106_valid_rows = valid_rows;

                // Capture the exact first row mapping and activation address that will
                // be paired with this weight tile. No computation is changed.
                const struct mmid_row_mapping dbg_map = MMID_MATRIX_ROW(cur_a, 0);
                const uint32_t dbg_rm1 = dbg_map.i1;
                const uint32_t dbg_rm2 = dbg_map.i2;
                const uint32_t dbg_ir1 = fastmodulo(dbg_rm1, ne11, &mmctx->mm_div_ne11);

                mmctx->dbg_v107_ne00        = ne00;
                mmctx->dbg_v107_ne10        = ne10;
                mmctx->dbg_v107_nb01        = nb01;
                mmctx->dbg_v107_nb02        = nb02;
                mmctx->dbg_v107_src1_stride = (uint32_t) src1_stride;
                mmctx->dbg_v107_rm1         = dbg_rm1;
                mmctx->dbg_v107_rm2         = dbg_rm2;
                mmctx->dbg_v107_ir1         = dbg_ir1;
                mmctx->dbg_v107_src1_off    =
                    (uint32_t) ((dbg_ir1 + dbg_rm2 * ne11) * src1_stride);
            }

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            for (uint32_t cid = 0; cid < (uint32_t)cne1; ++cid) {
                struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, cid);
                const int rm1 = row_mapping.i1;
                const int rm2 = row_mapping.i2;
                const uint32_t ir1 = fastmodulo(rm1, ne11, &mmctx->mm_div_ne11);
                const uint8_t * restrict src1_col =
                    src1_data + (ir1 + rm2 * ne11) * src1_stride;
                float * restrict dst_row =
                    (float *) (dst->data + (rm1 * nb1 + rm2 * nb2));

                // v10.21: claim exactly one real dot invocation per rm2 slice.
                // This record joins the outer V120 pre/post snapshot, allowing
                // the host to determine whether corruption first appears in
                // source addressing, Q8 quantization, expert/Q4 staging, or the
                // dot output.  The target ct comes from the existing runtime
                // debug control and defaults to row tile zero.
                struct htp_mmid_slice_trace_record * dbg_v121_rec = NULL;
                const uint32_t dbg_v121_target_ct =
                    mmctx->dbg_v112_enabled ? mmctx->dbg_v112_want_ct : 0u;
                if (octx->dbg_mmid_slice_trace &&
                    ct == dbg_v121_target_ct &&
                    rm2 >= 0 && (uint32_t) rm2 < HTP_MMID_SLICE_TRACE_MAX) {
                    const uint32_t bit = 1u << (uint32_t) rm2;
                    const uint32_t old = __sync_fetch_and_or(
                        &octx->dbg_mmid_slice_claimed, bit);
                    if ((old & bit) == 0) {
                        dbg_v121_rec =
                            &octx->dbg_mmid_slice_trace[(uint32_t) rm2];

                        const uint32_t q8_row = ir1 + (uint32_t) rm2 * ne11;
                        const uint32_t src1_z2 = ne12 > 0 ?
                            (uint32_t) rm2 % ne12 : 0u;
                        const uint32_t src1_z3 = ne12 > 0 ?
                            (uint32_t) rm2 / ne12 : 0u;
                        const uint8_t * src1_orig =
                            (const uint8_t *) src1->data +
                            (size_t) ir1 * nb11 +
                            (size_t) src1_z2 * nb12 +
                            (size_t) src1_z3 * nb13;
                        const uint8_t * src1_quant =
                            (const uint8_t *) src1->data +
                            (size_t) q8_row * nb11;
                        const size_t src1_span = ne10 == 0 ? 0 :
                            (size_t) (ne10 - 1u) * nb10 + sizeof(float);
                        const size_t q8_bytes = src1_stride;
                        const size_t raw_bytes =
                            (size_t) valid_rows * raw_row_size;
                        const size_t tiled_bytes =
                            (size_t) (ne00 / 32u) *
                            HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q4_0;
                        const uint8_t * vtcm_base =
                            (const uint8_t *) octx->ctx->vtcm_base;

                        dbg_v121_rec->quant_mode =
                            mmctx->dbg_v121_quant_mode;
                        dbg_v121_rec->n_threads = octx->n_threads;
                        dbg_v121_rec->n_quant_tasks =
                            mmctx->n_quant_tasks;
                        dbg_v121_rec->src1_nrows = mmctx->src1_nrows;
                        dbg_v121_rec->quant_rows_per_thread =
                            mmctx->n_quant_rows_per_thread;

                        dbg_v121_rec->expert = cur_a;
                        dbg_v121_rec->cid = cid;
                        dbg_v121_rec->ct = ct;
                        dbg_v121_rec->ith = ith;
                        dbg_v121_rec->rm1 = (uint32_t) rm1;
                        dbg_v121_rec->rm2 = (uint32_t) rm2;
                        dbg_v121_rec->ir1 = ir1;
                        dbg_v121_rec->q8_row = q8_row;
                        dbg_v121_rec->valid_rows = valid_rows;
                        dbg_v121_rec->mapping_stride =
                            mmctx->mapping_stride;
                        dbg_v121_rec->expert_mapping_count =
                            (uint32_t) cne1;

                        dbg_v121_rec->src_orig_off =
                            (uint32_t) (src1_orig -
                                (const uint8_t *) src1->data);
                        dbg_v121_rec->src_orig_hash_bytes =
                            (uint32_t) src1_span;
                        dbg_v121_rec->src_orig_hash =
                            htp_dbg_v121_fnv1a(src1_orig, src1_span);
                        dbg_v121_rec->src_quant_off =
                            (uint32_t) (src1_quant -
                                (const uint8_t *) src1->data);
                        dbg_v121_rec->src_quant_hash =
                            htp_dbg_v121_fnv1a(src1_quant, src1_span);
                        dbg_v121_rec->src_orig_word0 =
                            htp_dbg_v121_load_u32(src1_orig, src1_span);
                        dbg_v121_rec->src_orig_word1 = src1_span > 4 ?
                            htp_dbg_v121_load_u32(src1_orig + 4, src1_span - 4) : 0;
                        dbg_v121_rec->src_orig_word2 = src1_span > 8 ?
                            htp_dbg_v121_load_u32(src1_orig + 8, src1_span - 8) : 0;
                        dbg_v121_rec->src_orig_word3 = src1_span > 12 ?
                            htp_dbg_v121_load_u32(src1_orig + 12, src1_span - 12) : 0;

                        dbg_v121_rec->q8_vtcm_off =
                            (uint32_t) (src1_col - vtcm_base);
                        dbg_v121_rec->q8_stride = (uint32_t) src1_stride;
                        dbg_v121_rec->q8_hash_bytes = (uint32_t) q8_bytes;
                        dbg_v121_rec->q8_hash =
                            htp_dbg_v121_fnv1a(src1_col, q8_bytes);
                        dbg_v121_rec->q8_word0 =
                            htp_dbg_v121_load_u32(src1_col, q8_bytes);
                        dbg_v121_rec->q8_word1 = q8_bytes > 4 ?
                            htp_dbg_v121_load_u32(src1_col + 4, q8_bytes - 4) : 0;
                        dbg_v121_rec->q8_word2 = q8_bytes > 8 ?
                            htp_dbg_v121_load_u32(src1_col + 8, q8_bytes - 8) : 0;
                        dbg_v121_rec->q8_word3 = q8_bytes > 12 ?
                            htp_dbg_v121_load_u32(src1_col + 12, q8_bytes - 12) : 0;
                        dbg_v121_rec->q8_scale0 = q8_bytes > 1024 ?
                            htp_dbg_v121_load_u32(
                                src1_col + 1024, q8_bytes - 1024) : 0;
                        dbg_v121_rec->q8_scale1 = q8_bytes > 1028 ?
                            htp_dbg_v121_load_u32(
                                src1_col + 1028, q8_bytes - 1028) : 0;

                        dbg_v121_rec->q4_model_off =
                            (uint32_t) ((src0_expert +
                                (size_t) ct * 32u * raw_row_size) -
                                (const uint8_t *) src0->data);
                        dbg_v121_rec->q4_raw_vtcm_off =
                            (uint32_t) (raw_buf - vtcm_base);
                        dbg_v121_rec->q4_raw_hash_bytes =
                            (uint32_t) raw_bytes;
                        dbg_v121_rec->q4_raw_hash =
                            htp_dbg_v121_fnv1a(raw_buf, raw_bytes);
                        dbg_v121_rec->q4_raw_word0 =
                            htp_dbg_v121_load_u32(raw_buf, raw_bytes);
                        dbg_v121_rec->q4_raw_scale0 =
                            htp_dbg_v121_load_u32(raw_buf, MIN(raw_bytes, 2u));

                        dbg_v121_rec->q4_tiled_vtcm_off =
                            (uint32_t) (tiled_buf - vtcm_base);
                        dbg_v121_rec->q4_tiled_hash_bytes =
                            (uint32_t) tiled_bytes;
                        dbg_v121_rec->q4_tiled_hash =
                            htp_dbg_v121_fnv1a(tiled_buf, tiled_bytes);
                        dbg_v121_rec->q4_tiled_word0 =
                            htp_dbg_v121_load_u32(tiled_buf, tiled_bytes);
                        dbg_v121_rec->q4_tiled_scale0 = tiled_bytes > 512 ?
                            htp_dbg_v121_load_u32(
                                tiled_buf + 512, tiled_bytes - 512) : 0;

                        dbg_v121_rec->dst_off =
                            (uint32_t) ((uint8_t *) dst_row -
                                (uint8_t *) dst->data);
                    }
                }

                if (ct == (mmctx->dbg_v112_enabled ? mmctx->dbg_v112_want_ct : 0u) &&
                    cid == (mmctx->dbg_v112_enabled ? mmctx->dbg_v112_want_cid : 0u) &&
                    cur_a == mmctx->dbg_v103_expert &&
                    mmctx->dbg_v103_claimed) {
                    uint32_t fnv_dot = 2166136261u;
                    for (size_t di = 0; di < 1152; ++di) {
                        fnv_dot ^= src1_col[di];
                        fnv_dot *= 16777619u;
                    }
                    mmctx->dbg_v109_dot_fnv = fnv_dot;
                }

                // v10.11: exact first-K=32 probe for the deterministic debug path.
                // We call the SAME tiled kernel with n=32 into a temporary buffer,
                // then independently form a scalar row-0 reference from:
                //   - the raw GGUF Q4_0 block in raw_buf,
                //   - the original F32 activation row,
                //   - the actual Q8 scale emitted by the HTP quantizer.
                // The normal full-K result below is untouched.
                const uint32_t dbg_target_ct =
                    mmctx->dbg_v112_enabled ? mmctx->dbg_v112_want_ct : 0u;
                const uint32_t dbg_target_cid =
                    mmctx->dbg_v112_enabled ? mmctx->dbg_v112_want_cid : 0u;

                if (ct == dbg_target_ct && cid == dbg_target_cid &&
                    cur_a == mmctx->dbg_v103_expert &&
                    mmctx->dbg_v103_claimed) {
                    uint32_t dbg_k =
                        mmctx->dbg_v112_enabled ? mmctx->dbg_v112_want_k : 32u;
                    if (dbg_k > ne10) dbg_k = ne10;
                    dbg_k = (dbg_k / 32u) * 32u;
                    if (dbg_k < 32u) dbg_k = 32u;

                    // v10.14: GGML_HEXAGON_DEBUG_K selects the tile immediately
                    // before this boundary. K=192 => tile 5 => logical K 160..191.
                    const uint32_t dbg_tile_idx = dbg_k / 32u - 1u;
                    const uint8_t * dbg_wtile =
                        tiled_buf + (size_t) dbg_tile_idx * HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q4_0;
                    const uint8_t * dbg_atile =
                        src1_col + (size_t) dbg_tile_idx * HTP_MM_ACT_TILE_SIZE_Q8_0;

                    // Run the exact integer helper used by the official kernel.
                    // This deliberately avoids any scalar reconstruction of the
                    // tiled lane permutation.
                    const HVX_Vector * dbg_vptr = (const HVX_Vector *) dbg_wtile;
                    const HVX_Vector * dbg_vact = (const HVX_Vector *) dbg_atile;
                    HVX_Vector dbg_i8 = Q6_Vb_vsplat_R(8);
                    HVX_Vector dbg_vsum =
                        accum_4bit_32x1(dbg_vptr, dbg_vact, dbg_i8);
                    int32_t dbg_vsum_words[32] __attribute__((aligned(128)));
                    hvx_vec_store_u(dbg_vsum_words, 128, dbg_vsum);
                    const int32_t dbg_hvx_int_dot = dbg_vsum_words[0];

                    float hvx_k32[32] __attribute__((aligned(128)));
                    memset(hvx_k32, 0, sizeof(hvx_k32));
                    tiled_vec_dot_q4_0_32x1(
                        dbg_k, hvx_k32, tiled_buf, src1_col, valid_rows, NULL);

                    const uint8_t * q4_block = raw_buf; // row 0, K tile 0
                    uint16_t q4_scale_u16 = 0;
                    memcpy(&q4_scale_u16, q4_block, sizeof(q4_scale_u16));
                    __fp16 q4_scale_h;
                    memcpy(&q4_scale_h, &q4_scale_u16, sizeof(q4_scale_h));

                    uint16_t q8_scale_u16 = 0;
                    memcpy(&q8_scale_u16, src1_col + 1024, sizeof(q8_scale_u16));
                    __fp16 q8_scale_h;
                    memcpy(&q8_scale_h, &q8_scale_u16, sizeof(q8_scale_h));

                    const float q4_scale_f = (float) q4_scale_h;
                    const float q8_scale_f = (float) q8_scale_h;

                    // src1_col flattens (ir1 + rm2 * ne11).  The corresponding
                    // original F32 row is src1 + ir1*nb11 + rm2*nb12.
                    const uint8_t * src1_orig_bytes =
                        (const uint8_t *) src1->data +
                        (size_t) ir1 * nb11 +
                        (size_t) rm2 * nb12;

                    // Scalar comparison for exactly the selected logical
                    // Q4_0 block, using raw GGUF weights + original F32 activation.
                    // This is the already-useful v10.12 reference model, now
                    // applied to tile dbg_tile_idx rather than a K prefix.
                    const uint8_t * q4b =
                        raw_buf + (size_t) dbg_tile_idx * 18u;

                    uint16_t q4s_u16 = 0;
                    memcpy(&q4s_u16, q4b, sizeof(q4s_u16));
                    __fp16 q4s_h;
                    memcpy(&q4s_h, &q4s_u16, sizeof(q4s_h));

                    uint16_t q8s_u16 = 0;
                    memcpy(&q8s_u16, dbg_atile + 1024, sizeof(q8s_u16));
                    __fp16 q8s_h;
                    memcpy(&q8s_h, &q8s_u16, sizeof(q8s_h));
                    const float q8s_f = (float) q8s_h;

                    int32_t scalar_int_dot = 0;
                    const uint8_t * qs = q4b + 2;
                    for (uint32_t kli = 0; kli < 32; ++kli) {
                        const uint32_t ki = dbg_tile_idx * 32u + kli;
                        const int32_t qw =
                            (int32_t) htp_raw_q4_0_get_nibble(qs, kli) - 8;

                        float xf = 0.0f;
                        memcpy(&xf,
                            src1_orig_bytes + (size_t) ki * nb10,
                            sizeof(xf));

                        const __fp16 xh = (__fp16) xf;
                        int32_t qa = 0;
                        if (q8s_f != 0.0f && isfinite(q8s_f)) {
                            qa = (int32_t) roundf((float) xh / q8s_f);
                            if (qa > 127)  qa = 127;
                            if (qa < -128) qa = -128;
                        }
                        scalar_int_dot += qw * qa;
                    }

                    const float scalar_ref =
                        (float) scalar_int_dot * (float) q4s_h * q8s_f;
                    // v10.16: replay the current official
                    // quantize_block_f32_q8_0_tiled() arithmetic up to vx_i8,
                    // BEFORE vdelta lays it out into tiled activation memory.
                    //
                    // This removes scalar division/roundf from "q8_quant":
                    // F32 -> qf32 -> F16 -> max(abs) -> d=max/127 (F16)
                    // -> F16 reciprocal -> F16 multiply -> rnd/sat i16 -> i8.
                    const uint32_t q8_group_base =
                        (dbg_tile_idx / 4u) * 128u;
                    const uint32_t q8_subblock =
                        dbg_tile_idx % 4u;

                    float q8_replay_x[128]
                        __attribute__((aligned(128)));
                    for (uint32_t qi = 0; qi < 128u; ++qi) {
                        memcpy(&q8_replay_x[qi],
                            src1_orig_bytes +
                                (size_t) (q8_group_base + qi) * nb10,
                            sizeof(float));
                    }

                    HVX_Vector * qvx =
                        (HVX_Vector *) q8_replay_x;
                    HVX_Vector qzero = Q6_V_vzero();

                    HVX_Vector qvx0_qf =
                        Q6_Vqf32_vsub_VsfVsf(qvx[0], qzero);
                    HVX_Vector qvx1_qf =
                        Q6_Vqf32_vsub_VsfVsf(qvx[1], qzero);
                    HVX_Vector qvx2_qf =
                        Q6_Vqf32_vsub_VsfVsf(qvx[2], qzero);
                    HVX_Vector qvx3_qf =
                        Q6_Vqf32_vsub_VsfVsf(qvx[3], qzero);

                    HVX_Vector qvx01_hf =
                        Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(
                            Q6_W_vcombine_VV(qvx1_qf, qvx0_qf)));
                    HVX_Vector qvx23_hf =
                        Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(
                            Q6_W_vcombine_VV(qvx3_qf, qvx2_qf)));

                    HVX_Vector qvmax_hf =
                        hvx_vec_reduce_max_f16(
                            hvx_vec_abs_f16(qvx01_hf));
                    qvmax_hf =
                        hvx_vec_reduce_max2_f16(
                            hvx_vec_abs_f16(qvx23_hf),
                            qvmax_hf);

                    HVX_Vector qvd_qf16 =
                        Q6_Vqf16_vmpy_VhfVhf(
                            qvmax_hf,
                            Q6_Vh_vsplat_R(0x2008));
                    HVX_Vector qvd_hf =
                        Q6_Vhf_equals_Vqf16(qvd_qf16);
                    HVX_Vector qvd_inv_hf =
                        hvx_vec_inverse_f16(qvd_hf);

                    qvx01_hf =
                        Q6_Vhf_equals_Vqf16(
                            Q6_Vqf16_vmpy_VhfVhf(
                                qvx01_hf, qvd_inv_hf));
                    qvx23_hf =
                        Q6_Vhf_equals_Vqf16(
                            Q6_Vqf16_vmpy_VhfVhf(
                                qvx23_hf, qvd_inv_hf));

                    HVX_Vector qvx01_i16 =
                        hvx_vec_i16_from_hf_rnd_sat(
                            qvx01_hf);
                    HVX_Vector qvx23_i16 =
                        hvx_vec_i16_from_hf_rnd_sat(
                            qvx23_hf);
                    HVX_Vector qvx_i8 =
                        Q6_Vb_vpack_VhVh_sat(
                            qvx23_i16, qvx01_i16);

                    int8_t q8_replay_all[128]
                        __attribute__((aligned(128)));
                    hvx_vec_store_u(
                        q8_replay_all, 128, qvx_i8);

                    for (uint32_t kli = 0; kli < 32u; ++kli) {
                        const int32_t qw =
                            (int32_t)
                                htp_raw_q4_0_get_nibble(
                                    qs, kli) - 8;

                        const int32_t qa_quant =
                            (int32_t) q8_replay_all[
                                q8_subblock * 32u + kli];

                        const uint32_t ki =
                            dbg_tile_idx * 32u + kli;
                        float xf = 0.0f;
                        memcpy(&xf,
                            src1_orig_bytes +
                                (size_t) ki * nb10,
                            sizeof(xf));
                        const __fp16 xh = (__fp16) xf;

                        int32_t qa_scalar = 0;
                        if (q8s_f != 0.0f &&
                            isfinite(q8s_f)) {
                            qa_scalar =
                                (int32_t) roundf(
                                    (float) xh / q8s_f);
                            if (qa_scalar > 127)
                                qa_scalar = 127;
                            if (qa_scalar < -128)
                                qa_scalar = -128;
                        }

                        mmctx->dbg_v115_q4_weight[kli] =
                            (int8_t) qw;
                        mmctx->dbg_v115_q8_actual[kli] =
                            (int8_t) qa_quant;
                        mmctx->dbg_v115_q8_scalar[kli] =
                            (int8_t) qa_scalar;
                        mmctx->dbg_v115_prod_delta[kli] =
                            (int16_t) (
                                qw *
                                (qa_quant - qa_scalar));
                    }

                    // Exact single-tile floating result from the official kernel.
                    float hvx_tile[32] __attribute__((aligned(128)));
                    memset(hvx_tile, 0, sizeof(hvx_tile));
                    tiled_vec_dot_q4_0_32x1(
                        32, hvx_tile, dbg_wtile, dbg_atile, valid_rows, NULL);

                    union { float f; uint32_t u; } hvx_u, ref_u;
                    hvx_u.f = hvx_tile[0];
                    ref_u.f = scalar_ref;

                    mmctx->dbg_v111_hvx_k32_bits = hvx_u.u;
                    mmctx->dbg_v111_ref_k32_bits = ref_u.u;
                    mmctx->dbg_v111_int_dot      = scalar_int_dot;
                    mmctx->dbg_v111_scales_fp16  =
                        (uint32_t) q4s_u16 |
                        ((uint32_t) q8s_u16 << 16);

                    // v10.18: full-K exact reference for row 0.
                    // For every 128 activation values, replay the same
                    // HVX/F16 Q8 quantizer arithmetic validated by v10.16.
                    float dbg_full_ref = 0.0f;
                    for (uint32_t gbase = 0; gbase < ne10; gbase += 128u) {
                        const uint32_t gcount = MIN(128u, ne10 - gbase);

                        float gx[128] __attribute__((aligned(128)));
                        memset(gx, 0, sizeof(gx));
                        for (uint32_t qi = 0; qi < gcount; ++qi) {
                            memcpy(&gx[qi],
                                src1_orig_bytes +
                                    (size_t) (gbase + qi) * nb10,
                                sizeof(float));
                        }

                        HVX_Vector * gvx = (HVX_Vector *) gx;
                        HVX_Vector gzero = Q6_V_vzero();

                        HVX_Vector gvx0_qf =
                            Q6_Vqf32_vsub_VsfVsf(gvx[0], gzero);
                        HVX_Vector gvx1_qf =
                            Q6_Vqf32_vsub_VsfVsf(gvx[1], gzero);
                        HVX_Vector gvx2_qf =
                            Q6_Vqf32_vsub_VsfVsf(gvx[2], gzero);
                        HVX_Vector gvx3_qf =
                            Q6_Vqf32_vsub_VsfVsf(gvx[3], gzero);

                        HVX_Vector gvx01_hf =
                            Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(
                                Q6_W_vcombine_VV(gvx1_qf, gvx0_qf)));
                        HVX_Vector gvx23_hf =
                            Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(
                                Q6_W_vcombine_VV(gvx3_qf, gvx2_qf)));

                        HVX_Vector gvmax_hf =
                            hvx_vec_reduce_max_f16(
                                hvx_vec_abs_f16(gvx01_hf));
                        gvmax_hf =
                            hvx_vec_reduce_max2_f16(
                                hvx_vec_abs_f16(gvx23_hf), gvmax_hf);

                        HVX_Vector gvd_qf16 =
                            Q6_Vqf16_vmpy_VhfVhf(
                                gvmax_hf, Q6_Vh_vsplat_R(0x2008));
                        HVX_Vector gvd_hf =
                            Q6_Vhf_equals_Vqf16(gvd_qf16);
                        HVX_Vector gvd_inv_hf =
                            hvx_vec_inverse_f16(gvd_hf);

                        gvx01_hf = Q6_Vhf_equals_Vqf16(
                            Q6_Vqf16_vmpy_VhfVhf(
                                gvx01_hf, gvd_inv_hf));
                        gvx23_hf = Q6_Vhf_equals_Vqf16(
                            Q6_Vqf16_vmpy_VhfVhf(
                                gvx23_hf, gvd_inv_hf));

                        HVX_Vector gvx01_i16 =
                            hvx_vec_i16_from_hf_rnd_sat(gvx01_hf);
                        HVX_Vector gvx23_i16 =
                            hvx_vec_i16_from_hf_rnd_sat(gvx23_hf);
                        HVX_Vector gvx_i8 =
                            Q6_Vb_vpack_VhVh_sat(
                                gvx23_i16, gvx01_i16);

                        int8_t gq8[128] __attribute__((aligned(128)));
                        hvx_vec_store_u(gq8, 128, gvx_i8);

                        const uint32_t nsub = (gcount + 31u) / 32u;
                        for (uint32_t sb = 0; sb < nsub; ++sb) {
                            const uint32_t tile_idx = gbase / 32u + sb;
                            const uint8_t * rq4 =
                                raw_buf + (size_t) tile_idx * 18u;
                            const uint8_t * rqs = rq4 + 2;

                            uint16_t q4d_u16 = 0;
                            memcpy(&q4d_u16, rq4, sizeof(q4d_u16));
                            __fp16 q4d_h;
                            memcpy(&q4d_h, &q4d_u16, sizeof(q4d_h));

                            const uint8_t * qat =
                                src1_col +
                                (size_t) tile_idx * HTP_MM_ACT_TILE_SIZE_Q8_0;
                            uint16_t q8d_u16 = 0;
                            memcpy(&q8d_u16, qat + 1024, sizeof(q8d_u16));
                            __fp16 q8d_h;
                            memcpy(&q8d_h, &q8d_u16, sizeof(q8d_h));

                            int32_t idot = 0;
                            const uint32_t remain =
                                MIN(32u, ne10 - tile_idx * 32u);
                            for (uint32_t kli = 0; kli < remain; ++kli) {
                                const int32_t qw =
                                    (int32_t) htp_raw_q4_0_get_nibble(
                                        rqs, kli) - 8;
                                const int32_t qa =
                                    (int32_t) gq8[sb * 32u + kli];
                                idot += qw * qa;
                            }

                            dbg_full_ref +=
                                (float) idot *
                                (float) q4d_h *
                                (float) q8d_h;
                        }
                    }

                    union { float f; uint32_t u; } dbg_full_ref_u;
                    dbg_full_ref_u.f = dbg_full_ref;
                    mmctx->dbg_v118_full_ref_bits = dbg_full_ref_u.u;

                    // Word 31 was previously scales; keep it.  The new HVX
                    // integer dot is carried through an added response field
                    // populated from a spare debug word below.
                    mmctx->dbg_v114_hvx_int_dot = dbg_hvx_int_dot;
                }

                tiled_vec_dot_q4_0_32x1(
                    ne10, &dst_row[ct * 32], tiled_buf, src1_col, valid_rows, NULL);

                if (dbg_v121_rec) {
                    const uint32_t out_base = ct * 32u;
                    if (valid_rows > 0) {
                        memcpy(&dbg_v121_rec->dot_out0,
                            &dst_row[out_base + 0], sizeof(uint32_t));
                    }
                    if (valid_rows > 1) {
                        memcpy(&dbg_v121_rec->dot_out1,
                            &dst_row[out_base + 1], sizeof(uint32_t));
                    }
                    if (valid_rows > 2) {
                        memcpy(&dbg_v121_rec->dot_out2,
                            &dst_row[out_base + 2], sizeof(uint32_t));
                    }
                    if (valid_rows > 3) {
                        memcpy(&dbg_v121_rec->dot_out3,
                            &dst_row[out_base + 3], sizeof(uint32_t));
                    }
                    __sync_synchronize();
                    dbg_v121_rec->internal_valid = 1;
                }

                // v10.8.1: this is the real raw-Q4_0 MMID dot scope.
                // Capture only cid=0 / ct=0 so it corresponds to the same
                // deterministic debug expert/tile used by v10.6/v10.7.
                if (ct == (mmctx->dbg_v112_enabled ? mmctx->dbg_v112_want_ct : 0u) &&
                    cid == (mmctx->dbg_v112_enabled ? mmctx->dbg_v112_want_cid : 0u) &&
                    cur_a == mmctx->dbg_v103_expert &&
                    mmctx->dbg_v103_claimed) {
                    union { float f; uint32_t u; } o0, o1, o2, o3;
                    o0.f = dst_row[0];
                    o1.f = dst_row[1];
                    o2.f = dst_row[2];
                    o3.f = dst_row[3];
                    mmctx->dbg_v108_out0 = o0.u;
                    mmctx->dbg_v108_out1 = o1.u;
                    mmctx->dbg_v108_out2 = o2.u;
                    mmctx->dbg_v108_out3 = o3.u;
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);
        }
    }

    (void) nth;
}

static void hvx_mv_id_raw_q4_0(unsigned int nth, unsigned int ith, void * data) {
    htp_matmul_preamble;

    const struct htp_tensor * restrict ids = octx->src[2];
    const uint32_t src0_nrows      = ne01;
    const uint32_t src0_start_row  = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);

    hvx_mm_run_quant_task(mmctx, ith);
    if (src0_start_row >= src0_end_row) {
        return;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    const size_t src1_stride = mmctx->vtcm_src1_stride;
    uint8_t * restrict thread_vtcm = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    const size_t raw_row_size = nb01;
    const size_t raw_32_size = htp_mm_round_up(32 * raw_row_size, 256);
    uint8_t * restrict raw_buf = thread_vtcm;
    uint8_t * restrict tiled_buf = thread_vtcm + raw_32_size;
    const uint8_t * restrict src1_col = mmctx->vtcm_src1;

    const uint32_t n_aids = ids->ne[0];
    const uint32_t n_ids = ne02;

    for (uint32_t ie1 = 0; ie1 < n_aids; ++ie1) {
        const int32_t eid = *(const int32_t *) ((const uint8_t *) ids->data + ie1 * ids->nb[0]);
        if (eid < 0) continue;
        assert(eid < (int32_t)n_ids);

        const uint8_t * src0_expert = (const uint8_t *) src0->data + (size_t) eid * nb02;
        float * restrict dst_row = (float *) (dst->data + ie1 * nb1);

        const uint32_t ct_start = src0_start_row / 32;
        const uint32_t ct_end   = (src0_end_row + 31) / 32;
        for (uint32_t ct = ct_start; ct < ct_end; ++ct) {
            uint32_t valid_rows = ne01 - ct * 32;
            valid_rows = MIN(valid_rows, 32u);

            dma_queue_push(
                dma_queue,
                dma_make_ptr(raw_buf, src0_expert + (size_t) ct * 32 * raw_row_size),
                raw_row_size, raw_row_size, raw_row_size, valid_rows);
            (void) dma_queue_pop(dma_queue);

            htp_raw_q4_0_32rows_to_tiled(raw_buf, raw_row_size, tiled_buf, ne00, valid_rows);

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            tiled_vec_dot_q4_0_32x1(
                ne10, &dst_row[ct * 32], tiled_buf, src1_col, valid_rows, NULL);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);
        }
    }

    (void) nth;
}

static void hvx_mm_id(unsigned int nth, unsigned int ith, void * data) {
    htp_matmul_preamble;

    const struct htp_tensor * restrict ids = octx->src[2];

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    const uint32_t src0_nrows      = ne01;  // src0 rows per expert
    const uint32_t src1_nrows      = ne11;
    const uint32_t src0_start_row  = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);

    hvx_mm_run_quant_task(mmctx, ith);

    if (src0_start_row >= src0_end_row) {
        return;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_prefetch = kparams->n_prefetch;
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    const uint32_t n_ids = ids->ne[0];  // n_expert_used
    const uint32_t n_as  = ne02;        // n_expert

    const uint32_t *                matrix_row_counts = mmctx->matrix_row_counts;
    const struct mmid_row_mapping * matrix_rows       = mmctx->matrix_rows;

    const size_t dst_row_size  = nb1;
    const size_t src1_row_size = htp_mm_q8_0_tiled_row_size(ne10);

    const size_t src1_stride = mmctx->vtcm_src1_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict src1_data = mmctx->vtcm_src1;

    for (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
        const int32_t cne1 = matrix_row_counts[cur_a];
        if (cne1 == 0) {
            continue;
        }

        const uint8_t * src0_row = (const uint8_t *) src0->data + cur_a * nb02;

        const uint32_t tile_size = htp_mm_get_weight_tile_size(src0->type);
        const uint32_t aligned_tile_size = htp_mm_get_weight_aligned_tile_size(src0->type);
        const uint32_t n_k_tiles_w = ne00 / 32;
        const uint32_t n_k_tiles_a = ne10 / 32;
        const uint32_t tile_row_stride = n_k_tiles_w * tile_size;
        const uint32_t tile_row_transfer_size_aligned = n_k_tiles_a * aligned_tile_size;

        const uint32_t ct_start = src0_start_row / 32;
        const uint32_t ct_end   = (src0_end_row + 31) / 32;

        uint32_t push_ct = ct_start;
        for (uint32_t d = 0; d < n_prefetch && push_ct < ct_end; d++, push_ct++) {
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + d * tile_row_transfer_size_aligned, src0_row + push_ct * tile_row_stride),
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_a);
        }

        for (uint32_t ct = ct_start; ct < ct_end; ct++) {
            const uint8_t * w_tile = dma_queue_pop(dma_queue).dst;

            int valid_rows = (int)ne01 - (int)(ct * 32);
            valid_rows = MIN(32, MAX(0, valid_rows));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            for (uint32_t cid = 0; cid < cne1; ++cid) {
                struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, cid);
                const int               rm1         = row_mapping.i1;  // expert idx
                const int               rm2         = row_mapping.i2;  // token idx

                const uint32_t ir1 = fastmodulo(rm1, ne11, &mmctx->mm_div_ne11);        // src1 row idx
                const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + (ir1 + rm2 * ne11 + 0) * src1_stride);
                float * restrict dst_row = (float *) (dst->data + (rm1 * nb1 + rm2 * nb2 + 0));

                mmctx->vec_dot_32x1(ne10, &dst_row[ct * 32], w_tile, src1_col, valid_rows, NULL);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            if (push_ct < ct_end) {
                dma_queue_push(dma_queue, dma_make_ptr((uint8_t *)w_tile, src0_row + push_ct * tile_row_stride),
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_a);
                push_ct++;
            }
        }
    }
}

static void hvx_mv_id(unsigned int nth, unsigned int ith, void * data) {
    htp_matmul_preamble;

    const struct htp_tensor * restrict ids = octx->src[2];

    const uint32_t src0_nrows      = ne01;  // src0 rows per expert
    const uint32_t src0_start_row  = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);

    hvx_mm_run_quant_task(mmctx, ith);

    if (src0_start_row >= src0_end_row) {
        return;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_prefetch = kparams->n_prefetch;
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    assert(ne13 % ne03 == 0);

    const size_t dst_row_size  = nb1;
    const size_t src1_row_size = htp_mm_q8_0_tiled_row_size(ne10);

    const uint32_t n_aids = src2->ne[0];  // num activated experts
    const uint32_t n_ids  = ne02;         // num experts

    // Per-thread VTCMs for all tensors
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict src1_data = mmctx->vtcm_src1;

    for (uint32_t ie1 = 0; ie1 < n_aids; ++ie1) {  // for each expert
        const int32_t eid = *(const int32_t *) ((const uint8_t *) src2->data + ie1 * src2->nb[0]);
        if (eid < 0) {
            continue;
        }
        assert(eid < (int32_t) n_ids);

        const uint8_t * restrict src0_row = (const uint8_t *) src0->data + eid * nb02;
        const uint8_t * restrict src1_col = (const uint8_t *) src1_data;
        float * restrict dst_row          = (float *) (dst->data + ie1 * nb1);

        const uint32_t tile_size = htp_mm_get_weight_tile_size(src0->type);
        const uint32_t aligned_tile_size = htp_mm_get_weight_aligned_tile_size(src0->type);
        const uint32_t n_k_tiles_w = ne00 / 32;
        const uint32_t n_k_tiles_a = ne10 / 32;
        const uint32_t tile_row_stride = n_k_tiles_w * tile_size;
        const uint32_t tile_row_transfer_size_aligned = n_k_tiles_a * aligned_tile_size;

        const uint32_t ct_start = src0_start_row / 32;
        const uint32_t ct_end   = (src0_end_row + 31) / 32;

        uint32_t push_ct = ct_start;
        for (uint32_t d = 0; d < n_prefetch && push_ct < ct_end; d++, push_ct++) {
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + d * tile_row_transfer_size_aligned, src0_row + push_ct * tile_row_stride),
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_a);
        }

        for (uint32_t ct = ct_start; ct < ct_end; ct++) {
            const uint8_t * w_tile = dma_queue_pop(dma_queue).dst;

            int valid_rows = (int)ne01 - (int)(ct * 32);
            valid_rows = MIN(32, MAX(0, valid_rows));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            mmctx->vec_dot_32x1(ne10, &dst_row[ct * 32], w_tile, src1_col, valid_rows, NULL);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            if (push_ct < ct_end) {
                dma_queue_push(dma_queue, dma_make_ptr((uint8_t *)w_tile, src0_row + push_ct * tile_row_stride),
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_a);
                push_ct++;
            }
        }
    }
}

static int hvx_mm_init_vec_dot(struct htp_mm_context * mmctx, enum htp_data_type type) {
    switch (type) {
        case HTP_TYPE_Q4_0:
            mmctx->type         = "q4_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q4_0_32x1;
            return 0;
        case HTP_TYPE_Q4_1:
            mmctx->type         = "q4_1_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q4_1_32x1;
            return 0;
        case HTP_TYPE_Q8_0:
            mmctx->type         = "q8_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q8_0_32x1;
            return 0;
        case HTP_TYPE_IQ4_NL:
            mmctx->type         = "iq4nl_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_iq4nl_32x1;
            return 0;
        case HTP_TYPE_MXFP4:
            mmctx->type         = "mxfp4_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_mxfp4_32x1;
            return 0;
        default:
            return -1;
    }
}

static int hvx_mm_matmul(struct htp_ops_context * octx) {
    htp_matmul_tensors_preamble;

    struct htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    struct htp_mm_context mmctx_struct = {0};
    struct htp_mm_context * mmctx = &mmctx_struct;
    mmctx->octx = octx;

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;

    const uint32_t src0_nrows = ne01 * ne02 * ne03;
    const uint32_t src1_nrows = ne11 * ne12 * ne13;

    bool is_repacked = (src0->type == HTP_TYPE_Q4_0 || src0->type == HTP_TYPE_Q4_1 ||
                        src0->type == HTP_TYPE_Q8_0 || src0->type == HTP_TYPE_IQ4_NL ||
                        src0->type == HTP_TYPE_MXFP4);

    // Compute src0_nrows_per_thread
    mmctx->src0_nrows_per_thread  = (src0_nrows + octx->n_threads - 1) / octx->n_threads;
    if (is_repacked) {
        mmctx->src0_nrows_per_thread = hex_round_up(mmctx->src0_nrows_per_thread, 32);
    } else {
        mmctx->src0_nrows_per_thread += (mmctx->src0_nrows_per_thread & 1); // round up to even
    }

    const size_t src0_row_size = nb01;
    const size_t dst_row_size  = nb1;
    size_t       src1_row_size = nb11;

    const size_t src0_row_size_padded = hex_round_up(src0_row_size, 128);
    size_t       src1_row_size_padded;

    worker_callback_t quant_task_func;
    worker_callback_t matmul_job_func;
    uint32_t n_quant_tasks = 1;
    if (src1_nrows > 1) {
        if (is_repacked) {
            switch (src0->type) {
                case HTP_TYPE_Q4_0:   matmul_job_func = hvx_mm_2d_repacked_q4_0;   break;
                case HTP_TYPE_Q4_1:   matmul_job_func = hvx_mm_2d_repacked_q4_1;   break;
                case HTP_TYPE_Q8_0:   matmul_job_func = hvx_mm_2d_repacked_q8_0;   break;
                case HTP_TYPE_IQ4_NL: matmul_job_func = hvx_mm_2d_repacked_iq4nl;  break;
                case HTP_TYPE_MXFP4:  matmul_job_func = hvx_mm_2d_repacked_mxfp4;  break;
                default:              return HTP_STATUS_NO_SUPPORT;
            }
        } else {
            matmul_job_func = hvx_mm_2d;
        }
    } else {
        if (is_repacked) {
            switch (src0->type) {
                case HTP_TYPE_Q4_0:   matmul_job_func = hvx_mv_2d_repacked_q4_0;   break;
                case HTP_TYPE_Q4_1:   matmul_job_func = hvx_mv_2d_repacked_q4_1;   break;
                case HTP_TYPE_Q8_0:   matmul_job_func = hvx_mv_2d_repacked_q8_0;   break;
                case HTP_TYPE_IQ4_NL: matmul_job_func = hvx_mv_2d_repacked_iq4nl;  break;
                case HTP_TYPE_MXFP4:  matmul_job_func = hvx_mv_2d_repacked_mxfp4;  break;
                default:              return HTP_STATUS_NO_SUPPORT;
            }
        } else {
            matmul_job_func = hvx_mv_2d;
        }
    }

    bool need_quant = true;

    switch (kparams->kernel_type) {
        case HTP_MM_KERNEL_HVX_F16_F16_VTCM:
            quant_task_func        = (src1->type == HTP_TYPE_F32) ? quantize_f32_f16_flat : quantize_f16_f16_flat;
            mmctx->type            = "f16-f16";
            mmctx->vec_dot_1x1     = vec_dot_f16_f16_aa_1x1;
            mmctx->vec_dot_2x1     = vec_dot_f16_f16_aa_2x1;
            mmctx->vec_dot_2x2     = vec_dot_f16_f16_aa_2x2;
            src1_row_size          = hex_round_up(ne10 * 2, 128);
            break;

        case HTP_MM_KERNEL_HVX_F16_F32_DDR:
            mmctx->type            = "f16-f32";
            mmctx->vec_dot_1x1     = vec_dot_f16_f32_uu_1x1;
            matmul_job_func        = hvx_mm_4d;
            mmctx->mm_div_ne12_ne1 = kparams->div_ne12_ne1;
            mmctx->mm_div_ne1      = kparams->div_ne1;
            mmctx->mm_div_r2       = kparams->div_r2;
            mmctx->mm_div_r3       = kparams->div_r3;
            need_quant             = false;
            quant_task_func        = NULL;
            src1_row_size          = nb11;
            break;

        case HTP_MM_KERNEL_HVX_F16_F16_DDR:
            mmctx->type            = "f16-f16";
            mmctx->vec_dot_1x1     = vec_dot_f16_f16_uu_1x1;
            matmul_job_func        = hvx_mm_4d;
            mmctx->mm_div_ne12_ne1 = kparams->div_ne12_ne1;
            mmctx->mm_div_ne1      = kparams->div_ne1;
            mmctx->mm_div_r2       = kparams->div_r2;
            mmctx->mm_div_r3       = kparams->div_r3;
            src1_row_size          = nb11;
            need_quant             = false;
            quant_task_func        = NULL;
            break;

        case HTP_MM_KERNEL_HVX_F32_F32_VTCM:
            quant_task_func        = quantize_f32_f32_flat;
            mmctx->type            = "f32-f32";
            mmctx->vec_dot_1x1     = vec_dot_f32_f32_aa_1x1;
            mmctx->vec_dot_2x1     = vec_dot_f32_f32_aa_2x1;
            mmctx->vec_dot_2x2     = vec_dot_f32_f32_aa_2x2;
            src1_row_size          = hex_round_up(ne10 * 4, 128);
            break;

        case HTP_MM_KERNEL_HVX_F32_F32_DDR:
            quant_task_func        = NULL;
            mmctx->type            = "f32-f32";
            mmctx->vec_dot_1x1     = vec_dot_f32_f32_uu_1x1;
            mmctx->mm_div_ne12_ne1 = kparams->div_ne12_ne1;
            mmctx->mm_div_ne1      = kparams->div_ne1;
            mmctx->mm_div_r2       = kparams->div_r2;
            mmctx->mm_div_r3       = kparams->div_r3;
            src1_row_size          = nb11;
            need_quant             = false;
            matmul_job_func        = hvx_mm_4d;
            break;

        case HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT: {
            n_quant_tasks = MIN(src1_nrows, octx->n_threads);
            quant_task_func = (src0->type == HTP_TYPE_Q4_1) ? quantize_f32_q8_1_flat : quantize_f32_q8_0_flat;
            src1_row_size = (src0->type == HTP_TYPE_Q4_1) ? htp_mm_q8_1_flat_row_size(ne10) : htp_mm_q8_0_flat_row_size(ne10);

            if (src1_nrows > 1) {
                switch (src0->type) {
                    case HTP_TYPE_Q4_0:   matmul_job_func = hvx_mm_2d_repacked_q4_0_flat;   break;
                    case HTP_TYPE_Q4_1:   matmul_job_func = hvx_mm_2d_repacked_q4_1_flat;   break;
                    case HTP_TYPE_Q8_0:   matmul_job_func = hvx_mm_2d_repacked_q8_0_flat;   break;
                    case HTP_TYPE_IQ4_NL: matmul_job_func = hvx_mm_2d_repacked_iq4nl_flat;  break;
                    case HTP_TYPE_MXFP4:  matmul_job_func = hvx_mm_2d_repacked_mxfp4_flat;  break;
                    default:              return HTP_STATUS_NO_SUPPORT;
                }
            } else {
                switch (src0->type) {
                    case HTP_TYPE_Q4_0:   matmul_job_func = hvx_mv_2d_repacked_q4_0_flat;   break;
                    case HTP_TYPE_Q4_1:   matmul_job_func = hvx_mv_2d_repacked_q4_1_flat;   break;
                    case HTP_TYPE_Q8_0:   matmul_job_func = hvx_mv_2d_repacked_q8_0_flat;   break;
                    case HTP_TYPE_IQ4_NL: matmul_job_func = hvx_mv_2d_repacked_iq4nl_flat;  break;
                    case HTP_TYPE_MXFP4:  matmul_job_func = hvx_mv_2d_repacked_mxfp4_flat;  break;
                    default:              return HTP_STATUS_NO_SUPPORT;
                }
            }
            break;
        }

        case HTP_MM_KERNEL_HVX_QUANT_BLOCK:
        case HTP_MM_KERNEL_HVX_QUANT_ROW:
        default:
            if (hvx_mm_init_vec_dot(mmctx, src0->type) != 0) {
                return HTP_STATUS_NO_SUPPORT;
            }

            const uint32_t qk = QK_Q8_0_TILED;
            const uint32_t nb = (ne10 + qk - 1) / qk;
            const uint32_t total_nb = src1_nrows * nb;

            if (src1_nrows < octx->n_threads) {
                n_quant_tasks = MIN(total_nb, octx->n_threads);
                quant_task_func = (src0->type == HTP_TYPE_Q4_1) ? quantize_f32_q8_1_tiled_block : quantize_f32_q8_0_tiled_block;
                for (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
                    uint32_t ib_first = (total_nb * ith) / n_quant_tasks;
                    uint32_t ib_last  = (total_nb * (ith + 1)) / n_quant_tasks;
                    mmctx->quant_ib_first[ith] = ib_first;
                    mmctx->quant_ib_last[ith]  = ib_last;
                    mmctx->quant_r[ith]        = ib_first / nb;
                    mmctx->quant_c[ith]        = ib_first % nb;
                }
            } else {
                n_quant_tasks = MIN(src1_nrows, octx->n_threads);
                quant_task_func = (src0->type == HTP_TYPE_Q4_1) ? quantize_f32_q8_1_tiled : quantize_f32_q8_0_tiled;
            }
            src1_row_size = (src0->type == HTP_TYPE_Q4_1) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);
            break;
    }

    struct htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, ne10, src1_nrows, octx->n_threads,
                                 dst_row_size, src0_row_size, src1_row_size, src2 ? src2->nb[1] : 0, kparams->n_prefetch, false, false, false);

    if (kparams->kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK) {
        mmctx->vtcm_src1_size_per_thread = L.src1_bytes;
    } else {
        mmctx->vtcm_src1_size_per_thread = L.src1_bytes / octx->n_threads;
    }

    mmctx->vtcm_src0_size_per_thread = L.src0_bytes / octx->n_threads;
    mmctx->vtcm_dst_size_per_thread  = L.dst_bytes / octx->n_threads;

    size_t vtcm_size = kparams->vtcm_size > 0 ? (size_t)kparams->vtcm_size : L.total_bytes;

    FARF(HIGH, "matmul-%s : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu (%zu)\n", mmctx->type,
         L.src0_bytes, L.src1_bytes, L.dst_bytes, vtcm_size);

    FARF(HIGH, "matmul-%s : %ux%ux%ux%u * %ux%ux%ux%u-> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", mmctx->type, src0->ne[0],
         src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0],
         dst->ne[1], dst->ne[2], dst->ne[3], src0->data, src1->data, dst->data);

    if (octx->ctx->vtcm_size < vtcm_size) {
        FARF(ERROR, "matmul-%s : current VTCM reservation %zu is too small, needed %zu\n", mmctx->type,
             octx->ctx->vtcm_size, vtcm_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * const base = (uint8_t *) octx->ctx->vtcm_base;
    mmctx->vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src1);
    mmctx->vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
    mmctx->vtcm_src2 = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src2);
    mmctx->vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);

    octx->src1_spad.src  = NULL;
    octx->src0_spad.src  = NULL;
    octx->dst_spad.src   = NULL;

    mmctx->vtcm_src0_stride = src0_row_size_padded;
    mmctx->vtcm_src1_stride = src1_row_size;

    if (need_quant) {
        mmctx->n_quant_rows_per_thread = (src1_nrows + n_quant_tasks - 1) / n_quant_tasks;
        mmctx->quant_task_func = quant_task_func;
        mmctx->n_quant_tasks = n_quant_tasks;
        atomic_init(&mmctx->quant_barrier, n_quant_tasks);
    } else {
        mmctx->quant_task_func = NULL;
        mmctx->n_quant_tasks = 0;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    worker_pool_run_func(octx->ctx->worker_pool, matmul_job_func, mmctx, octx->n_threads);

    return HTP_STATUS_OK;
}

static void hvx_mm_qkv_2d(unsigned int nth, unsigned int ith, void * data) {
    struct htp_mm_context * mmctx = data;
    struct htp_ops_context * octx = mmctx->octx;

    const struct htp_tensor * restrict src0 = octx->src[0]; // Wk
    const struct htp_tensor * restrict src1 = octx->src[1]; // x
    const struct htp_tensor * restrict src2 = octx->src[2]; // Wv
    const struct htp_tensor * restrict src3 = octx->src[3]; // Wq
    const struct htp_tensor * restrict dst_k = octx->dsts[0];
    const struct htp_tensor * restrict dst_v = octx->dsts[1];
    const struct htp_tensor * restrict dst_q = octx->dsts[2];

    const uint32_t ne00 = src0->ne[0];
    const uint32_t ne01 = src0->ne[1];
    const uint32_t ne02 = src0->ne[2];
    const uint32_t ne03 = src0->ne[3];

    const uint32_t ne11 = src1->ne[1];
    const uint32_t ne12 = src1->ne[2];
    const uint32_t ne13 = src1->ne[3];

    const uint32_t src0_nrows = ne01 * ne02 * ne03;
    const uint32_t src1_nrows = ne11 * ne12 * ne13;

    const uint32_t src0_nrows_per_thread = mmctx->src0_nrows_per_thread;
    const uint32_t src0_start_row  = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);
    const uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);

    const size_t dst_k_row_size  = dst_k->nb[1]; // K and V share output width
    const size_t dst_q_row_size  = dst_q->nb[1]; // Q may be wider (GQA)
    const size_t src0_row_size = src0->nb[1];
    const size_t src2_row_size = src2->nb[1];
    const size_t src3_row_size = src3->nb[1];

    const size_t src0_stride = mmctx->vtcm_src0_stride;
    const size_t src2_stride = mmctx->vtcm_src2_stride;
    const size_t src3_stride = mmctx->vtcm_src3_stride;
    const size_t src1_stride = mmctx->vtcm_src1_stride;

    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict vtcm_src2_ptr = mmctx->vtcm_src2 + mmctx->vtcm_src2_size_per_thread * ith;
    uint8_t * restrict vtcm_src3_ptr = mmctx->vtcm_src3 + mmctx->vtcm_src3_size_per_thread * ith;
    uint8_t * restrict src1_data = mmctx->vtcm_src1;

    dma_queue * dma_queue = octx->ctx->dma[ith];

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_prefetch = kparams->n_prefetch;
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    const uint32_t prefetch_mask = n_prefetch - 1;

    const uint8_t * restrict src0_row = (const uint8_t *) src0->data;
    const uint8_t * restrict src2_row = (const uint8_t *) src2->data;
    const uint8_t * restrict src3_row = (const uint8_t *) src3->data;

    // Prefill spad with src0, src2, src3 rows
    if (src0_start_row < src0_end_row) {
        for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
            const int is0 = (ir0 - src0_start_row);
            if (is0 >= (int)n_prefetch) {
                break;
            }
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src2_ptr + is0 * src2_stride, src2_row + ir0 * src2_row_size),
                           src2_stride, src2_row_size, src2_row_size, 2);
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src3_ptr + is0 * src3_stride, src3_row + ir0 * src3_row_size),
                           src3_stride, src3_row_size, src3_row_size, 2);
        }
    }

    hvx_mm_run_quant_task(mmctx, ith);

    if (src0_start_row >= src0_end_row) {
        return;
    }

    // Process rows
    for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        const uint8_t * ss0 = dma_queue_pop(dma_queue).dst;
        const uint8_t * ss2 = dma_queue_pop(dma_queue).dst;
        const uint8_t * ss3 = dma_queue_pop(dma_queue).dst;

        // Process src1 columns in pairs (2×2 tiling)
        uint32_t ir1 = 0;
        for (; ir1 + 1 < src1_nrows; ir1 += 2) {
            const uint8_t * restrict src1_col0 = (const uint8_t *) (src1_data + (ir1+0) * src1_stride);
            const uint8_t * restrict src1_col1 = (const uint8_t *) (src1_data + (ir1+1) * src1_stride);

            float * restrict dst_row0_k = (float *) (dst_k->data + ((ir1+0) * dst_k_row_size));
            float * restrict dst_row1_k = (float *) (dst_k->data + ((ir1+1) * dst_k_row_size));
            mmctx->vec_dot_2x2(ne00, &dst_row0_k[ir0], &dst_row1_k[ir0], ss0, ss0 + src0_stride, src1_col0, src1_col1);

            float * restrict dst_row0_v = (float *) (dst_v->data + ((ir1+0) * dst_k_row_size));
            float * restrict dst_row1_v = (float *) (dst_v->data + ((ir1+1) * dst_k_row_size));
            mmctx->vec_dot_2x2(ne00, &dst_row0_v[ir0], &dst_row1_v[ir0], ss2, ss2 + src2_stride, src1_col0, src1_col1);

            float * restrict dst_row0_q = (float *) (dst_q->data + ((ir1+0) * dst_q_row_size));
            float * restrict dst_row1_q = (float *) (dst_q->data + ((ir1+1) * dst_q_row_size));
            mmctx->vec_dot_2x2(ne00, &dst_row0_q[ir0], &dst_row1_q[ir0], ss3, ss3 + src3_stride, src1_col0, src1_col1);
        }

        // Handle remaining src1 rows (fallback to 2×1)
        for (; ir1 < src1_nrows; ++ir1) {
            const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);

            float * restrict dst_row_k          = (float *) (dst_k->data + (ir1 * dst_k_row_size));
            mmctx->vec_dot_2x1(ne00, &dst_row_k[ir0], ss0, ss0 + src0_stride, src1_col);

            float * restrict dst_row_v          = (float *) (dst_v->data + (ir1 * dst_k_row_size));
            mmctx->vec_dot_2x1(ne00, &dst_row_v[ir0], ss2, ss2 + src2_stride, src1_col);

            float * restrict dst_row_q          = (float *) (dst_q->data + (ir1 * dst_q_row_size));
            mmctx->vec_dot_2x1(ne00, &dst_row_q[ir0], ss3, ss3 + src3_stride, src1_col);
        }

        // Prefetch next (n + vtcm_nrows) rows
        const int pr0 = (ir0 + n_prefetch);
        const int is0 = (pr0 - src0_start_row) & prefetch_mask;
        if (pr0 < src0_end_row_x2) {
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + is0 * src0_stride, src0_row + pr0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src2_ptr + is0 * src2_stride, src2_row + pr0 * src2_row_size),
                           src2_stride, src2_row_size, src2_row_size, 2);
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src3_ptr + is0 * src3_stride, src3_row + pr0 * src3_row_size),
                           src3_stride, src3_row_size, src3_row_size, 2);
        }
    }

    // Process last row (if any)
    if (src0_end_row != src0_end_row_x2) {
        uint32_t  ir0 = src0_end_row_x2;
        const int is0 = (ir0 - src0_start_row) & prefetch_mask;
        dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                       src0_stride, src0_row_size, src0_row_size, 1);
        dma_queue_push(dma_queue, dma_make_ptr(vtcm_src2_ptr + is0 * src2_stride, src2_row + ir0 * src2_row_size),
                       src2_stride, src2_row_size, src2_row_size, 1);
        dma_queue_push(dma_queue, dma_make_ptr(vtcm_src3_ptr + is0 * src3_stride, src3_row + ir0 * src3_row_size),
                       src3_stride, src3_row_size, src3_row_size, 1);

        const uint8_t * ss0 = dma_queue_pop(dma_queue).dst;
        const uint8_t * ss2 = dma_queue_pop(dma_queue).dst;
        const uint8_t * ss3 = dma_queue_pop(dma_queue).dst;

        for (uint32_t ir1 = 0; ir1 < src1_nrows; ++ir1) {
            const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);

            float * restrict dst_row_k          = (float *) (dst_k->data + (ir1 * dst_k_row_size));
            mmctx->vec_dot_1x1(ne00, &dst_row_k[ir0], ss0, src1_col);

            float * restrict dst_row_v          = (float *) (dst_v->data + (ir1 * dst_k_row_size));
            mmctx->vec_dot_1x1(ne00, &dst_row_v[ir0], ss2, src1_col);

            float * restrict dst_row_q          = (float *) (dst_q->data + (ir1 * dst_q_row_size));
            mmctx->vec_dot_1x1(ne00, &dst_row_q[ir0], ss3, src1_col);
        }
    }
}

static void hvx_mm_ffn_2d(unsigned int nth, unsigned int ith, void * data) {
    struct htp_mm_context * mmctx = data;
    struct htp_ops_context * octx = mmctx->octx;

    const struct htp_tensor * restrict src0 = octx->src[0]; // Wgate
    const struct htp_tensor * restrict src1 = octx->src[1]; // y
    const struct htp_tensor * restrict src2 = octx->src[2]; // Wup
    const struct htp_tensor * restrict dst_gate = octx->dsts[0];
    const struct htp_tensor * restrict dst_up = octx->dsts[1];

    const uint32_t ne00 = src0->ne[0];
    const uint32_t ne01 = src0->ne[1];
    const uint32_t ne02 = src0->ne[2];
    const uint32_t ne03 = src0->ne[3];

    const uint32_t ne11 = src1->ne[1];
    const uint32_t ne12 = src1->ne[2];
    const uint32_t ne13 = src1->ne[3];

    const uint32_t src0_nrows = ne01 * ne02 * ne03;
    const uint32_t src1_nrows = ne11 * ne12 * ne13;

    const uint32_t src0_nrows_per_thread = mmctx->src0_nrows_per_thread;
    const uint32_t src0_start_row  = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);
    const uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);

    const size_t dst_row_size  = dst_gate->nb[1];
    const size_t src0_row_size = src0->nb[1];
    const size_t src2_row_size = src2->nb[1];

    const size_t src0_stride = mmctx->vtcm_src0_stride;
    const size_t src2_stride = mmctx->vtcm_src2_stride;
    const size_t src1_stride = mmctx->vtcm_src1_stride;

    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict vtcm_src2_ptr = mmctx->vtcm_src2 + mmctx->vtcm_src2_size_per_thread * ith;
    uint8_t * restrict src1_data = mmctx->vtcm_src1;

    dma_queue * dma_queue = octx->ctx->dma[ith];

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_prefetch = kparams->n_prefetch;
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    const uint32_t prefetch_mask = n_prefetch - 1;

    const uint8_t * restrict src0_row = (const uint8_t *) src0->data;
    const uint8_t * restrict src2_row = (const uint8_t *) src2->data;

    // Prefill spad with src0, src2 rows
    if (src0_start_row < src0_end_row) {
        for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
            const int is0 = (ir0 - src0_start_row);
            if (is0 >= (int)n_prefetch) {
                break;
            }
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src2_ptr + is0 * src2_stride, src2_row + ir0 * src2_row_size),
                           src2_stride, src2_row_size, src2_row_size, 2);
        }
    }

    hvx_mm_run_quant_task(mmctx, ith);

    if (src0_start_row >= src0_end_row) {
        return;
    }

    // Process rows
    for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        const uint8_t * ss0 = dma_queue_pop(dma_queue).dst;
        const uint8_t * ss2 = dma_queue_pop(dma_queue).dst;

        // Process src1 columns in pairs (2×2 tiling)
        uint32_t ir1 = 0;
        for (; ir1 + 1 < src1_nrows; ir1 += 2) {
            const uint8_t * restrict src1_col0 = (const uint8_t *) (src1_data + (ir1+0) * src1_stride);
            const uint8_t * restrict src1_col1 = (const uint8_t *) (src1_data + (ir1+1) * src1_stride);

            float * restrict dst_row0_gate = (float *) (dst_gate->data + ((ir1+0) * dst_row_size));
            float * restrict dst_row1_gate = (float *) (dst_gate->data + ((ir1+1) * dst_row_size));
            mmctx->vec_dot_2x2(ne00, &dst_row0_gate[ir0], &dst_row1_gate[ir0], ss0, ss0 + src0_stride, src1_col0, src1_col1);

            float * restrict dst_row0_up   = (float *) (dst_up->data + ((ir1+0) * dst_row_size));
            float * restrict dst_row1_up   = (float *) (dst_up->data + ((ir1+1) * dst_row_size));
            mmctx->vec_dot_2x2(ne00, &dst_row0_up[ir0], &dst_row1_up[ir0], ss2, ss2 + src2_stride, src1_col0, src1_col1);
        }

        // Handle remaining src1 rows (fallback to 2×1)
        for (; ir1 < src1_nrows; ++ir1) {
            const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);

            float * restrict dst_row_gate     = (float *) (dst_gate->data + (ir1 * dst_row_size));
            mmctx->vec_dot_2x1(ne00, &dst_row_gate[ir0], ss0, ss0 + src0_stride, src1_col);

            float * restrict dst_row_up       = (float *) (dst_up->data + (ir1 * dst_row_size));
            mmctx->vec_dot_2x1(ne00, &dst_row_up[ir0], ss2, ss2 + src2_stride, src1_col);
        }

        // Prefetch next rows
        const int pr0 = (ir0 + n_prefetch);
        const int is0 = (pr0 - src0_start_row) & prefetch_mask;
        if (pr0 < src0_end_row_x2) {
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + is0 * src0_stride, src0_row + pr0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src2_ptr + is0 * src2_stride, src2_row + pr0 * src2_row_size),
                           src2_stride, src2_row_size, src2_row_size, 2);
        }
    }

    // Process last row (if any)
    if (src0_end_row != src0_end_row_x2) {
        uint32_t  ir0 = src0_end_row_x2;
        const int is0 = (ir0 - src0_start_row) & prefetch_mask;
        dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                       src0_stride, src0_row_size, src0_row_size, 1);
        dma_queue_push(dma_queue, dma_make_ptr(vtcm_src2_ptr + is0 * src2_stride, src2_row + ir0 * src2_row_size),
                       src2_stride, src2_row_size, src2_row_size, 1);

        const uint8_t * ss0 = dma_queue_pop(dma_queue).dst;
        const uint8_t * ss2 = dma_queue_pop(dma_queue).dst;

        for (uint32_t ir1 = 0; ir1 < src1_nrows; ++ir1) {
            const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);

            float * restrict dst_row_gate      = (float *) (dst_gate->data + (ir1 * dst_row_size));
            mmctx->vec_dot_1x1(ne00, &dst_row_gate[ir0], ss0, src1_col);

            float * restrict dst_row_up        = (float *) (dst_up->data + (ir1 * dst_row_size));
            mmctx->vec_dot_1x1(ne00, &dst_row_up[ir0], ss2, src1_col);
        }
    }
}

#define DEQUANTIZE_WORKER_LOOP_IMPL(SUFFIX)                                                     \
static void dequantize_tiled_worker_loop_##SUFFIX(unsigned int n, unsigned int i, void *data) { \
    tiled_dequantize_state_t *state = (tiled_dequantize_state_t *)data;                         \
    struct htp_thread_trace * tr = &state->traces[i];                                           \
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);                                  \
    for (unsigned int task_id = i; task_id < (unsigned int)state->n_tasks; task_id += n) {      \
        int start = task_id * state->n_tiles_per_task;                                          \
        int end   = hex_smin(start + state->n_tiles_per_task, state->n_tot_tiles);              \
        dequantize_tiled_weight_to_fp16_task_##SUFFIX(state, start, end);                       \
    }                                                                                           \
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);                                   \
}

DEQUANTIZE_WORKER_LOOP_IMPL(q4_0)
DEQUANTIZE_WORKER_LOOP_IMPL(q4_1)
DEQUANTIZE_WORKER_LOOP_IMPL(iq4_nl)
DEQUANTIZE_WORKER_LOOP_IMPL(mxfp4)
DEQUANTIZE_WORKER_LOOP_IMPL(q8_0)

static void convert_f16_worker_loop(unsigned int n, unsigned int i, void *data) {
    tiled_dequantize_state_t *state = (tiled_dequantize_state_t *)data;
    struct htp_thread_trace * tr = &state->traces[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);
    for (unsigned int task_id = i; task_id < (unsigned int)state->n_tasks; task_id += n) {
        int start = task_id * state->n_tiles_per_task;
        int end   = hex_smin(start + state->n_tiles_per_task, state->n_tot_tiles);
        convert_f16_weight_to_fp16_tiles_task(state, start, end);
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);
}

static void quantize_f32_worker_loop(unsigned int n, unsigned int i, void *data) {
    tiled_dequantize_state_t *state = (tiled_dequantize_state_t *)data;

    struct htp_thread_trace * tr = &state->traces[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, i);

    for (unsigned int task_id = i; task_id < (unsigned int)state->n_tasks; task_id += n) {
        int start = task_id * state->n_tiles_per_task;
        int end   = hex_smin(start + state->n_tiles_per_task, state->n_tot_tiles);
        quantize_f32_weight_to_fp16_tiles_task(state, start, end);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, i);
}

static void transfer_output_chunk_worker_fn(unsigned int n, unsigned int i, void *data) {
    output_transfer_task_state_t *st = (output_transfer_task_state_t *) data;

    struct htp_thread_trace * tr = &st->traces[i];

    int start_chunk_idx = i * st->n_chunks_per_task;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, start_chunk_idx);

    for (unsigned int task_id = i; task_id < (unsigned int)st->n_tasks; task_id += n) {
        int    chunk_idx  = task_id * st->n_chunks_per_task;
        size_t chunk_size = hex_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_per_task);

        float        *dst      = st->dst      + chunk_idx * st->dst_stride;
        const float  *src2     = st->src2     ? (st->src2     + chunk_idx * st->src2_stride) : NULL;
        transfer_output_chunk_fp16_to_fp32(dst, src2, st->vtcm_src, chunk_idx, chunk_size, st->n_cols, st->dst_stride, st->src2_stride, st->dst_cols);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, start_chunk_idx);
}

typedef struct {
    const struct mmid_row_mapping  *matrix_rows;
    __fp16      *dst;
    const float *src;
    uint32_t     n_tasks;
    uint32_t     n_tot_chunks;
    uint32_t     n_chunks_per_task;
    uint32_t     k_block;
    uint32_t     k_stride;
    uint32_t     k_valid;
    struct htp_thread_trace * traces;
    struct htp_context * ctx;
    float              * vtcm_f32_act;
    size_t               vtcm_f32_act_bytes_per_thread;
    uint32_t             dma_step_rows;
    uint32_t             dma_step_rows_shift;
} activation_transfer_task_state_t;

typedef struct {
    __fp16                         *dst;
    const float                    *src;
    uint32_t                        n_rows;
    uint32_t                        k_block;
    uint32_t                        k_stride;
    uint32_t                        k_valid;
    uint32_t                        n_col_chunks;
    struct fastdiv_values           n_threads_div;
    float                          *vtcm_f32_act;
    size_t                          vtcm_f32_act_bytes;
    struct htp_thread_trace        *traces;
    struct htp_context             *ctx;
    uint32_t                        dma_step_rows;
    uint32_t                        dma_step_rows_shift;
} activation_transfer_col_chunk_state_t;

static void transfer_activation_chunk_fp32_to_fp16_dma_pipelined_col_chunk(
        dma_queue *dma_q,
        __fp16 *restrict vtcm_dst,
        const float *restrict src,
        uint32_t n_rows,
        uint32_t k_block,
        uint32_t k_stride,
        uint32_t k_chunk_valid,
        uint32_t c_first,
        uint32_t c_len,
        float *thread_f32_act,
        struct htp_thread_trace *tr,
        uint32_t dma_step_rows,
        uint32_t dma_step_rows_shift) {

    const uint32_t R = dma_step_rows;
    const uint32_t n_rows_padded = hex_align_up(n_rows, HTP_MM_HMX_TILE_N_ROWS);

    const uint32_t n_steps = n_rows_padded >> dma_step_rows_shift;

    // Push step 0
    if (n_steps > 0 && n_rows > 0) {
        uint32_t nrows_to_fetch = hex_smin(n_rows, R);
        dma_queue_push(dma_q, dma_make_ptr(thread_f32_act, src + c_first),
                       c_len * sizeof(float), k_stride * sizeof(float), k_chunk_valid * sizeof(float), nrows_to_fetch);
    }
    // Push step 1
    if (n_steps > 1) {
        uint32_t next_r = R * 1;
        if (next_r < n_rows) {
            uint32_t nrows_to_fetch = hex_smin(n_rows - next_r, R);
            const float *next_src = src + next_r * k_stride + c_first;
            float *next_buf = thread_f32_act + 1 * R * c_len;
            dma_queue_push(dma_q, dma_make_ptr(next_buf, next_src),
                           c_len * sizeof(float), k_stride * sizeof(float), k_chunk_valid * sizeof(float), nrows_to_fetch);
        }
    }
    for (uint32_t s = 0; s < n_steps; ++s) {
        uint32_t r = s << dma_step_rows_shift;
        float *curr_buf = thread_f32_act;

        if (r < n_rows) {
            curr_buf = (float *) dma_queue_pop(dma_q).dst;
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, r);
        for (uint32_t p = 0; p < (R >> 1); ++p) {
            uint32_t row_idx = r + (p << 1);
            float *pair_buf = curr_buf + (p << 1) * c_len;
            bool r0_valid = ((row_idx + 0) < n_rows);
            bool r1_valid = ((row_idx + 1) < n_rows);

            transfer_activation_row_pair_fp32_to_fp16_col_chunk(
                vtcm_dst, pair_buf, pair_buf + c_len, row_idx, k_block, c_first, c_len, k_chunk_valid, r0_valid, r1_valid
            );
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, r);

        // Push step s + 2
        uint32_t next_s = s + 2;
        uint32_t next_r = next_s << dma_step_rows_shift;
        if (next_r < n_rows) {
            uint32_t nrows_to_fetch = hex_smin(n_rows - next_r, R);
            const float *next_src = src + next_r * k_stride + c_first;
            dma_queue_push(dma_q, dma_make_ptr(curr_buf, next_src),
                           c_len * sizeof(float), k_stride * sizeof(float), k_chunk_valid * sizeof(float), nrows_to_fetch);
        }
    }
}

static void transfer_activation_chunk_fp32_to_fp16_col_chunk(
        __fp16 *restrict vtcm_dst,
        const float *restrict src,
        uint32_t n_rows,
        uint32_t k_block,
        uint32_t k_stride,
        uint32_t c_first,
        uint32_t c_len,
        uint32_t k_chunk_valid) {
    const uint32_t n_rows_padded = hex_align_up(n_rows, HTP_MM_HMX_TILE_N_ROWS);
    const uint32_t n_rows_tiled  = (n_rows / HTP_MM_HMX_TILE_N_ROWS) * HTP_MM_HMX_TILE_N_ROWS;

    uint32_t r = 0;

    #pragma unroll(2)
    for (r = 0; r < n_rows_tiled; r += 2) {
        const float *ptr_in0 = src + (r + 0) * k_stride + c_first;
        const float *ptr_in1 = src + (r + 1) * k_stride + c_first;

        transfer_activation_row_pair_fp32_to_fp16_col_chunk(
            vtcm_dst, ptr_in0, ptr_in1, r, k_block, c_first, c_len, k_chunk_valid, true, true
        );
    }

    for (; r < n_rows_padded; r += 2) {
        const bool row0_valid = r       < n_rows;
        const bool row1_valid = (r + 1) < n_rows;

        const float *ptr_in0 = row0_valid ? (src + (r + 0) * k_stride + c_first) : NULL;
        const float *ptr_in1 = row1_valid ? (src + (r + 1) * k_stride + c_first) : NULL;

        transfer_activation_row_pair_fp32_to_fp16_col_chunk(
            vtcm_dst, ptr_in0, ptr_in1, r, k_block, c_first, c_len, k_chunk_valid, row0_valid, row1_valid
        );
    }
}

static void transfer_activation_chunk_col_chunk_worker_fn(unsigned int n, unsigned int i, void *data) {
    activation_transfer_col_chunk_state_t *st = (activation_transfer_col_chunk_state_t *) data;
    struct htp_thread_trace * tr = &st->traces[i];

    uint32_t n_blocks = st->k_block / 32;
    uint32_t b_first = fastdiv(n_blocks * i, &st->n_threads_div);
    uint32_t b_last  = fastdiv(n_blocks * (i + 1), &st->n_threads_div);
    uint32_t c_first = b_first * 32;
    uint32_t c_last = b_last * 32;
    uint32_t c_len = c_last - c_first;

    if (c_len == 0) {
        return;
    }

    uint32_t k_chunk_valid = 0;
    if (st->k_valid > c_first) {
        k_chunk_valid = hex_smin(st->k_valid, c_last) - c_first;
    }

    __fp16 *dst = st->dst;
    const float *src = st->src;

    if (st->vtcm_f32_act) {
        size_t thread_scratch_bytes = hex_align_down(fastdiv(st->vtcm_f32_act_bytes, &st->n_threads_div), 128);
        float *thread_f32_act = (float *)((char *)st->vtcm_f32_act + i * thread_scratch_bytes);

        transfer_activation_chunk_fp32_to_fp16_dma_pipelined_col_chunk(
            st->ctx->dma[i], dst, src, st->n_rows, st->k_block, st->k_stride, k_chunk_valid,
            c_first, c_len, thread_f32_act, tr, st->dma_step_rows, st->dma_step_rows_shift
        );
    } else {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, c_first);
        transfer_activation_chunk_fp32_to_fp16_col_chunk(
            dst, src, st->n_rows, st->k_block, st->k_stride, c_first, c_len, k_chunk_valid
        );
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, c_first);
    }
}

static void transfer_activation_chunk_fp32_to_fp16_dma_pipelined(
        dma_queue *dma_q,
        __fp16 *restrict vtcm_dst,
        const float *restrict src,
        uint32_t n_rows,
        uint32_t k_block,
        uint32_t k_stride,
        uint32_t k_valid,
        float *thread_f32_act,
        struct htp_thread_trace *tr,
        uint32_t dma_step_rows,
        uint32_t dma_step_rows_shift) {

    const uint32_t R = dma_step_rows;
    const uint32_t n_rows_padded = hex_align_up(n_rows, HTP_MM_HMX_TILE_N_ROWS);

    const uint32_t n_steps = n_rows_padded >> dma_step_rows_shift;

    // Push step 0
    if (n_steps > 0 && n_rows > 0) {
        uint32_t nrows_to_fetch = hex_smin(n_rows, R);
        dma_queue_push(dma_q, dma_make_ptr(thread_f32_act, src),
                       k_block * sizeof(float), k_stride * sizeof(float), k_valid * sizeof(float), nrows_to_fetch);
    }
    // Push step 1 (if valid)
    if (n_steps > 1) {
        uint32_t next_r = R * 1;
        if (next_r < n_rows) {
            uint32_t nrows_to_fetch = hex_smin(n_rows - next_r, R);
            const float *next_src = src + next_r * k_stride;
            float *next_buf = thread_f32_act + 1 * R * k_block;
            dma_queue_push(dma_q, dma_make_ptr(next_buf, next_src),
                           k_block * sizeof(float), k_stride * sizeof(float), k_valid * sizeof(float), nrows_to_fetch);
        }
    }
    for (uint32_t s = 0; s < n_steps; ++s) {
        uint32_t r = s << dma_step_rows_shift;
        float *curr_buf = thread_f32_act;

        if (r < n_rows) {
            curr_buf = (float *) dma_queue_pop(dma_q).dst;
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, r);
        for (uint32_t p = 0; p < (R >> 1); ++p) {
            uint32_t row_idx = r + (p << 1);
            float *pair_buf = curr_buf + (p << 1) * k_block;
            bool r0_valid = ((row_idx + 0) < n_rows);
            bool r1_valid = ((row_idx + 1) < n_rows);

            transfer_activation_row_pair_fp32_to_fp16(vtcm_dst, pair_buf, pair_buf + k_block, row_idx, k_block, k_valid, r0_valid, r1_valid);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, r);

        // Push step s + 2
        uint32_t next_s = s + 2;
        uint32_t next_r = next_s << dma_step_rows_shift;
        if (next_r < n_rows) {
            uint32_t nrows_to_fetch = hex_smin(n_rows - next_r, R);
            const float *next_src = src + next_r * k_stride;
            dma_queue_push(dma_q, dma_make_ptr(curr_buf, next_src),
                           k_block * sizeof(float), k_stride * sizeof(float), k_valid * sizeof(float), nrows_to_fetch);
        }
    }
}

static void transfer_activation_chunk_worker_fn(unsigned int n, unsigned int i, void *data) {
    activation_transfer_task_state_t *st = (activation_transfer_task_state_t *) data;

    struct htp_thread_trace * tr = &st->traces[i];

    for (unsigned int task_id = i; task_id < (unsigned int)st->n_tasks; task_id += n) {
        int    chunk_idx  = task_id * st->n_chunks_per_task;
        size_t chunk_size = hex_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_per_task);

        __fp16      *dst = st->dst + chunk_idx * st->k_block;
        const float *src = st->src + chunk_idx * st->k_stride;

        if (st->vtcm_f32_act) {
            float *thread_f32_act = (float *)((char *)st->vtcm_f32_act + i * st->vtcm_f32_act_bytes_per_thread);
            transfer_activation_chunk_fp32_to_fp16_dma_pipelined(
                st->ctx->dma[i], dst, src, chunk_size, st->k_block, st->k_stride, st->k_valid, thread_f32_act, tr, st->dma_step_rows, st->dma_step_rows_shift
            );
        } else {
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
            transfer_activation_chunk_fp32_to_fp16(dst, src, chunk_size, st->k_block, st->k_stride, st->k_valid);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
        }
    }
}

typedef struct {
    const struct mmid_row_mapping  *matrix_rows;
    __fp16                         *dst;
    const float                    *src;
    uint32_t                        n_tasks;
    uint32_t                        n_tot_chunks;
    uint32_t                        n_chunks_per_task;
    uint32_t                        k_block;
    uint32_t                        cur_a;
    uint32_t                        mapping_stride;
    uint32_t                        ne11;
    struct fastdiv_values           ne11_div;
    size_t                          nb11;
    size_t                          nb12;
    uint32_t                        start_row;
    uint32_t                        cne1;
    uint32_t                        k_valid;
    struct htp_thread_trace        *traces;
} activation_transfer_gathered_task_state_t;

typedef struct {
    const struct mmid_row_mapping  *matrix_rows;
    const __fp16                   *vtcm_src;
    float                          *dst;
    uint32_t                        n_tasks;
    uint32_t                        n_tot_chunks;
    uint32_t                        n_chunks_per_task;
    uint32_t                        n_cols;
    uint32_t                        cur_a;
    uint32_t                        mapping_stride;
    size_t                          dst_nb1;
    size_t                          dst_nb2;
    uint32_t                        start_row;
    uint32_t                        cne1;
    struct htp_thread_trace        *traces;
} output_transfer_scattered_task_state_t;

static void transfer_activation_chunk_gathered_worker_fn(unsigned int n, unsigned int i, void *data) {
    activation_transfer_gathered_task_state_t *st = data;
    struct htp_thread_trace * tr = &st->traces[i];
    const uint32_t chunk_idx = i;
    const uint32_t chunk_size = st->n_chunks_per_task;
    const uint32_t vtcm_start_row = chunk_idx * chunk_size;
    const uint32_t start_row = st->start_row + vtcm_start_row;
    if (start_row >= st->cne1) {
        return;
    }
    const uint32_t n_rows = (uint32_t) hex_smin(
        (size_t) (st->cne1 - start_row), (size_t) chunk_size);
    if (n_rows > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
        transfer_activation_chunk_fp32_to_fp16_gathered(
            st->dst, st->src, start_row, vtcm_start_row, n_rows, st->k_block,
            st->matrix_rows, st->cur_a, st->mapping_stride,
            st->ne11, &st->ne11_div, st->nb11, st->nb12, st->cne1, st->k_valid);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
    }
}

static void transfer_activation_chunk_gathered_worker_flat_fn(unsigned int n, unsigned int i, void *data) {
    activation_transfer_gathered_task_state_t *st = data;
    struct htp_thread_trace * tr = &st->traces[i];
    const uint32_t chunk_idx = i;
    const uint32_t chunk_size = st->n_chunks_per_task;
    const uint32_t vtcm_start_row = chunk_idx * chunk_size;
    const uint32_t start_row = st->start_row + vtcm_start_row;
    if (start_row >= st->cne1) {
        return;
    }
    const uint32_t n_rows = (uint32_t) hex_smin(
        (size_t) (st->cne1 - start_row), (size_t) chunk_size);
    if (n_rows > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
        transfer_activation_chunk_fp32_to_fp16_gathered_flat(
            st->dst, st->src, start_row, vtcm_start_row, n_rows, st->k_block,
            st->matrix_rows, st->cur_a, st->mapping_stride,
            st->nb12, st->cne1, st->k_valid);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
    }
}

static void transfer_output_chunk_scattered_worker_fn(unsigned int n, unsigned int i, void *data) {
    output_transfer_scattered_task_state_t *st = data;
    struct htp_thread_trace * tr = &st->traces[i];
    const uint32_t chunk_idx = i;
    const uint32_t chunk_size = st->n_chunks_per_task;
    const uint32_t vtcm_start_row = chunk_idx * chunk_size;
    const uint32_t start_row = st->start_row + vtcm_start_row;
    if (start_row >= st->cne1) {
        return;
    }
    const uint32_t n_rows = (uint32_t) hex_smin(
        (size_t) (st->cne1 - start_row), (size_t) chunk_size);
    if (n_rows > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, chunk_idx);
        transfer_output_chunk_fp16_to_fp32_scattered(
            st->dst, st->vtcm_src, start_row, vtcm_start_row, n_rows, st->n_cols,
            st->matrix_rows, st->cur_a, st->mapping_stride,
            st->dst_nb1, st->dst_nb2, st->cne1);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, chunk_idx);
    }
}

// --- HMX Dispatchers & Entry Points ---

static void dequantize_tiled_weight_chunk_to_fp16_tiles(
        struct htp_context *ctx, __fp16 *vtcm_dst,
        const void *weight_src_ddr,
        int n_cols, int k_block,
        size_t row_stride, int weight_type,
        int n_k_tiles, struct fastdiv_values n_k_tiles_div,
        worker_callback_t dequant_worker_fn, int n_threads) {

    assert(n_cols  % HTP_MM_HMX_TILE_N_COLS == 0);
    assert(k_block % HTP_MM_HMX_TILE_N_COLS == 0);

    size_t n_col_tiles = n_cols / HTP_MM_HMX_TILE_N_COLS;
    size_t n_tot_tiles = n_col_tiles * n_k_tiles;

    size_t n_tiles_per_task = (n_threads == 1) ? n_tot_tiles : hmx_ceil_div(n_tot_tiles, n_threads);

    tiled_dequantize_state_t state;
    state.n_tasks          = (n_tot_tiles + n_tiles_per_task - 1) / n_tiles_per_task;
    state.n_tot_tiles      = n_tot_tiles;
    state.n_tiles_per_task = n_tiles_per_task;
    state.dst              = vtcm_dst;
    state.src              = (const uint8_t *)weight_src_ddr;
    state.n_cols           = n_cols;
    state.k_block          = k_block;
    state.row_stride       = row_stride;
    state.weight_type      = weight_type;
    state.n_k_tiles        = n_k_tiles;
    state.n_k_tiles_div    = n_k_tiles_div;
    state.traces           = ctx->trace;
    state.ctx              = ctx;

    state.tile_size = htp_mm_get_weight_tile_size(weight_type);
    state.aligned_tile_size = htp_mm_get_weight_aligned_tile_size(weight_type);

    if (state.n_tasks == 1 || n_threads == 1) {
        dequant_worker_fn(1, 0, &state);
    } else {
        int n_tasks = hex_smin((int) state.n_tasks, n_threads);
        worker_pool_run_func(ctx->worker_pool, dequant_worker_fn, &state, n_tasks);
    }
}

typedef struct {
    float *dst;
    const float *src2;
    const __fp16 *vtcm_src;
    uint32_t n_rows;
    uint32_t n_cols;
    uint32_t dst_stride;
    uint32_t src2_stride;
    uint32_t dst_cols;
    struct fastdiv_values n_threads_div;
    struct htp_thread_trace *traces;
    struct htp_context *ctx;
} output_transfer_col_chunk_state_t;

static void transfer_output_chunk_col_chunk_worker_fn(unsigned int n, unsigned int i, void *data) {
    (void) n;
    output_transfer_col_chunk_state_t *st = (output_transfer_col_chunk_state_t *) data;
    struct htp_thread_trace * tr = &st->traces[i];

    uint32_t n_blocks = st->n_cols / 32;
    uint32_t b_first = fastdiv(n_blocks * i, &st->n_threads_div);
    uint32_t b_last  = fastdiv(n_blocks * (i + 1), &st->n_threads_div);
    uint32_t c_first = b_first * 32;
    uint32_t c_last  = b_last * 32;
    uint32_t c_len   = c_last - c_first;

    if (c_len == 0) return;

    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, c_first);

    float *dst = st->dst + c_first;
    const float *src2 = st->src2 ? (st->src2 + c_first) : NULL;
    const __fp16 *vtcm_src = st->vtcm_src + b_first * HTP_MM_HMX_TILE_N_ELMS;

    int chunk_dst_cols = (int)st->dst_cols - (int)c_first;
    if (chunk_dst_cols > 0) {
        transfer_output_chunk_fp16_to_fp32_col_chunk(
            dst, src2, vtcm_src, 0, st->n_rows, c_len, st->n_cols,
            st->dst_stride, st->src2_stride, (uint32_t)chunk_dst_cols
        );
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, c_first);
}

static void transfer_output_chunk_threaded(struct htp_context *ctx, float *dst, const float *src2, const __fp16 *vtcm_src,
                                              int n_rows, int n_cols, int dst_stride, uint32_t src2_stride, int dst_cols, int n_threads) {
    assert(n_cols % HTP_MM_HMX_TILE_N_COLS == 0);

    if (n_rows <= 0) return;

    uint32_t n_blocks = (uint32_t)n_cols / 32;
    if (n_threads > 1 && n_blocks >= (uint32_t)n_threads) {
        struct fastdiv_values n_threads_div = init_fastdiv_values(n_threads);
        output_transfer_col_chunk_state_t col_state;
        col_state.dst = dst;
        col_state.src2 = src2;
        col_state.vtcm_src = vtcm_src;
        col_state.n_rows = (uint32_t)n_rows;
        col_state.n_cols = (uint32_t)n_cols;
        col_state.dst_stride = (uint32_t)dst_stride;
        col_state.src2_stride = src2_stride;
        col_state.dst_cols = (uint32_t)dst_cols;
        col_state.n_threads_div = n_threads_div;
        col_state.traces = ctx->trace;
        col_state.ctx = ctx;

        worker_pool_run_func(ctx->worker_pool, transfer_output_chunk_col_chunk_worker_fn, &col_state, n_threads);
        return;
    }

    size_t n_tot_chunks      = n_rows;
    size_t n_chunks_per_task = (n_threads == 1) ? n_tot_chunks : hmx_ceil_div(n_rows, n_threads);
    n_chunks_per_task        = hex_align_up(n_chunks_per_task, 2);

    int actual_threads = hmx_ceil_div(n_rows, n_chunks_per_task);

    output_transfer_task_state_t state;
    state.n_tasks           = actual_threads;
    state.n_tot_chunks      = n_tot_chunks;
    state.n_chunks_per_task = n_chunks_per_task;
    state.dst               = dst;
    state.src2              = src2;
    state.vtcm_src          = vtcm_src;
    state.n_cols            = n_cols;
    state.dst_stride        = dst_stride;
    state.src2_stride       = src2_stride;
    state.dst_cols          = dst_cols;
    state.traces            = ctx->trace;

    if (actual_threads <= 1) {
        transfer_output_chunk_worker_fn(1, 0, &state);
    } else {
        worker_pool_run_func(ctx->worker_pool, transfer_output_chunk_worker_fn, &state, actual_threads);
    }
}

struct activation_transfer_params {
    struct htp_context *          ctx;
    __fp16 *                      dst;
    const float *                 src;
    int                           n_rows;
    int                           k_block;
    int                           k_stride;
    int                           n_threads;
    const struct fastdiv_values * act_threads_div;
    const struct fastdiv_values * k_div;
    int                           k_valid;
    float *                       vtcm_f32_act;
    size_t                        vtcm_f32_act_bytes;
};

static void transfer_activation_chunk_threaded(const struct activation_transfer_params * params) {
    struct htp_context *          ctx                = params->ctx;
    __fp16 *                      dst                = params->dst;
    const float *                 src                = params->src;
    int                           n_rows             = params->n_rows;
    int                           k_block            = params->k_block;
    int                           k_stride           = params->k_stride;
    int                           n_threads          = params->n_threads;
    const struct fastdiv_values * act_threads_div    = params->act_threads_div;
    const struct fastdiv_values * k_div              = params->k_div;
    int                           k_valid            = params->k_valid;
    float *                       vtcm_f32_act       = params->vtcm_f32_act;
    size_t                        vtcm_f32_act_bytes = params->vtcm_f32_act_bytes;

    if (n_rows <= 0) {
        return;
    }

    const size_t n_tasks = (n_rows + 31) >> 5;
    if (n_threads > 1 && k_block > 32 && n_tasks < (size_t) n_threads) {
        // Calculate step rows parameters for column-chunked dma pipelining
        uint32_t dma_step_rows = 2;
        uint32_t dma_step_rows_shift = 1;
        if (vtcm_f32_act && vtcm_f32_act_bytes > 0 && k_block > 0) {
            size_t thread_scratch_bytes = hex_align_down(fastdiv(vtcm_f32_act_bytes, act_threads_div), 128);
            size_t thread_scratch_elements = thread_scratch_bytes / sizeof(float);
            size_t dma_step_rows_max = fastdiv(thread_scratch_elements / 2, k_div);
            if (dma_step_rows_max >= 4) {
                dma_step_rows = 4;
                dma_step_rows_shift = 2;
            }
        }

        activation_transfer_col_chunk_state_t col_state;
        col_state.dst = dst;
        col_state.src = src;
        col_state.n_rows = n_rows;
        col_state.k_block = k_block;
        col_state.k_stride = k_stride;
        col_state.k_valid = k_valid;
        col_state.n_col_chunks = n_threads;
        col_state.n_threads_div = *act_threads_div;
        col_state.vtcm_f32_act = vtcm_f32_act;
        col_state.vtcm_f32_act_bytes = vtcm_f32_act_bytes;
        col_state.traces = ctx->trace;
        col_state.ctx = ctx;
        col_state.dma_step_rows = dma_step_rows;
        col_state.dma_step_rows_shift = dma_step_rows_shift;

        worker_pool_run_func(ctx->worker_pool, transfer_activation_chunk_col_chunk_worker_fn, &col_state, n_threads);
        return;
    }

    assert(k_block % HTP_MM_HMX_TILE_N_COLS == 0 && k_stride % HTP_MM_HMX_TILE_N_COLS == 0);

    size_t n_tot_chunks      = n_rows;
    size_t n_chunks_per_task = (n_threads == 1) ? n_tot_chunks : 32;  // must be multiple of 32 to ensure correct destination address

    activation_transfer_task_state_t state;
    state.n_tasks            = (n_threads == 1) ? 1 : hmx_ceil_div(n_tot_chunks, 32);
    state.n_tot_chunks       = n_tot_chunks;
    state.n_chunks_per_task  = n_chunks_per_task;
    state.dst                = dst;
    state.src                = src;
    state.k_block            = k_block;
    state.k_stride           = k_stride;
    state.k_valid            = k_valid;
    state.traces             = ctx->trace;
    state.ctx                = ctx;
    state.vtcm_f32_act       = vtcm_f32_act;

    int active_threads = hex_smin(n_threads, (int)state.n_tasks);
    state.vtcm_f32_act_bytes_per_thread = hex_align_down(vtcm_f32_act_bytes / active_threads, 128);

    uint32_t dma_step_rows = 2;
    uint32_t dma_step_rows_shift = 1;
    if (vtcm_f32_act && state.vtcm_f32_act_bytes_per_thread > 0 && k_block > 0) {
        size_t thread_scratch_elements = state.vtcm_f32_act_bytes_per_thread / sizeof(float);
        size_t dma_step_rows_max = fastdiv(thread_scratch_elements / 2, k_div);
        if (dma_step_rows_max >= 4) {
            dma_step_rows = 4;
            dma_step_rows_shift = 2;
        }
    }
    state.dma_step_rows      = dma_step_rows;
    state.dma_step_rows_shift = dma_step_rows_shift;

    if (state.n_tasks == 1 || n_threads == 1) {
        transfer_activation_chunk_worker_fn(1, 0, &state);
    } else {
        worker_pool_run_func(ctx->worker_pool, transfer_activation_chunk_worker_fn, &state, active_threads);
    }
}
// --- Async HMX matmul job (for pipeline overlap) ---

typedef struct {
    __fp16 *       output;
    const __fp16 * activation;
    const __fp16 * weight;
    const __fp16 * scales;
    uint32_t       n_row_tiles;
    uint32_t       n_col_tiles;
    uint32_t       n_dot_tiles;
} hmx_matmul_job_t;

static void hmx_matmul_worker_fn(void * data) {
    hmx_matmul_job_t * job = (hmx_matmul_job_t *) data;
    FARF(HIGH, "hmx-mm-job: n_row_tiles %u n_col_tiles %u n_dot_tiles %u", job->n_row_tiles, job->n_col_tiles, job->n_dot_tiles);
    core_dot_chunk_fp16(job->output, job->activation, job->weight, job->scales, job->n_row_tiles, job->n_col_tiles, job->n_dot_tiles);
}

static inline void hmx_matmul_job_init(hmx_matmul_job_t * job,
                                       __fp16 *           output,
                                       const __fp16 *     activation,
                                       const __fp16 *     weight,
                                       const __fp16 *     scales,
                                       uint32_t           n_row_tiles,
                                       uint32_t           n_col_tiles,
                                       uint32_t           n_dot_tiles) {
    job->output      = output;
    job->activation  = activation;
    job->weight      = weight;
    job->scales      = scales;
    job->n_row_tiles = n_row_tiles;
    job->n_col_tiles = n_col_tiles;
    job->n_dot_tiles = n_dot_tiles;
}

static int hmx_mm_2d_f32(struct htp_context *ctx,
                                  float *restrict dst,
                                  const float *restrict src2,
                                  const float *activation,
                                  const uint8_t *weight,
                                  int m, int k, int n,
                                  int act_stride,
                                  int weight_stride,
                                  int weight_type,
                                  int k_valid,
                                  int dst_stride,
                                  uint32_t src2_stride,
                                  int dst_cols,
                                  int m_chunk,
                                  int n_chunk,
                                  int pipeline,
                                  int n_threads,
                                  int act_threads,
                                  const struct fastdiv_values * act_threads_div,
                                  const struct fastdiv_values * k_div,
                                  int tile_size,
                                  int aligned_tile_size,
                                  int vtcm_size) {
    struct htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    if (k % 32 != 0 || n % 32 != 0) { return -1; }
    if (!hex_is_aligned(dst, VLEN) || !hex_is_aligned(activation, VLEN)) { return -1; }

    size_t row_stride = htp_mm_get_tiled_row_stride(weight_type, k);
    if (row_stride == 0) {
        return -1;
    }

    worker_callback_t dequant_worker_fn = NULL;
    switch (weight_type) {
        case HTP_TYPE_Q4_0:   dequant_worker_fn = dequantize_tiled_worker_loop_q4_0; break;
        case HTP_TYPE_IQ4_NL: dequant_worker_fn = dequantize_tiled_worker_loop_iq4_nl; break;
        case HTP_TYPE_Q4_1:   dequant_worker_fn = dequantize_tiled_worker_loop_q4_1; break;
        case HTP_TYPE_MXFP4:  dequant_worker_fn = dequantize_tiled_worker_loop_mxfp4; break;
        case HTP_TYPE_Q8_0:   dequant_worker_fn = dequantize_tiled_worker_loop_q8_0; break;
        case HTP_TYPE_F16:    dequant_worker_fn = convert_f16_worker_loop; break;
        case HTP_TYPE_F32:    dequant_worker_fn = quantize_f32_worker_loop; break;
        default:
            return -1;
    }

    const int n_k_tiles = k / HTP_MM_HMX_TILE_N_COLS;
    const struct fastdiv_values n_k_tiles_div = init_fastdiv_values(n_k_tiles);

    const bool is_quant       = (weight_type != HTP_TYPE_F16 && weight_type != HTP_TYPE_F32);
    const size_t vec_dot_size = k * sizeof(__fp16);
    const size_t vtcm_budget  = ctx->vtcm_size;

    const uint32_t dma_dst_stride  = is_quant ? aligned_tile_size : row_stride;
    const uint32_t dma_src_stride  = is_quant ? tile_size : weight_stride;
    const uint32_t dma_width_bytes = is_quant ? tile_size : row_stride;

    size_t m_chunk_n_rows = m_chunk;
    size_t n_chunk_n_cols = n_chunk;
    size_t vtcm_used      = vtcm_size;

    const size_t qweight_row_stride = is_quant ? (size_t)(n_k_tiles * aligned_tile_size) / 32 : 0;

    struct htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, weight_type, k, m_chunk_n_rows, n_chunk_n_cols, 1, false, pipeline, act_threads, aligned_tile_size);

    vtcm_used = L.total_bytes;
    if (vtcm_used > vtcm_budget) {
        FARF(ERROR, "hmx-mm-2d-precomputed: VTCM overflow: used %zu budget %zu, m %d k %d n %d mc %zu nc %zu",
             vtcm_used, vtcm_budget, m, k, n, m_chunk_n_rows, n_chunk_n_cols);
        return -1;
    }

    uint8_t * const base = (uint8_t *) ctx->vtcm_base;
    __fp16  *vtcm_weight_raw[2] = {
        VTCM_LAYOUT_PTR(__fp16, base, L.off_weight[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(__fp16, base, L.off_weight[1], pipeline)
    };

    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, base, L.off_act);
    float   *vtcm_f32_act    = VTCM_LAYOUT_PTR(float, base, L.off_act_f32);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, base, L.off_dst[0]);
    void    *vtcm_scratch0   = VTCM_LAYOUT_PTR(void, base, L.off_scratch[0]);
    void    *vtcm_scratch1   = VTCM_LAYOUT_PTR_OPTIONAL(void, base, L.off_scratch[1], pipeline);
    void    *vtcm_scratch2   = VTCM_LAYOUT_PTR_OPTIONAL(void, base, L.off_dst[1], pipeline);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, base, L.off_scales);

    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    FARF(HIGH, "hmx-mm-2d: m %d k %d n %d wtype %d mc %zu nc %zu vtcm %zu/%zu",
         m, k, n, weight_type, m_chunk_n_rows, n_chunk_n_cols, vtcm_used, vtcm_budget);

    int n_chunk_cnt = hmx_ceil_div(n, n_chunk_n_cols);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    if (pipeline) {
        // --- Asynchronous Pipelined Loop ---
        hmx_matmul_job_t job_slots[2];  // persistent double-buffered job descriptors

        for (size_t mr = 0; mr < m; mr += m_chunk_n_rows) {
            const size_t n_rows = hex_smin(m - mr, m_chunk_n_rows);

            void *vtcm_weight_bufs[2] = { vtcm_scratch0, vtcm_scratch1 };
            void *vtcm_output_bufs[2] = { vtcm_output,   vtcm_scratch2 };

            struct activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .src = activation + mr * act_stride,
                .n_rows = (int) n_rows,
                .k_block = k,
                .k_stride = act_stride,
                .n_threads = act_threads,
                .act_threads_div = act_threads_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_bytes = L.act_f32_bytes,
            };
            transfer_activation_chunk_threaded(&act_params);

            // Prologue: push A0 and optionally A1 (if n_chunk_cnt > 1)
            const size_t   n_cols_A0 = hex_smin(n - 0 * n_chunk_n_cols, n_chunk_n_cols);
            const uint32_t height_A0 = is_quant ? (n_cols_A0 / 32) * n_k_tiles : n_cols_A0;
            dma_queue_push(ctx->dma[0], dma_make_ptr(vtcm_weight_raw[0], weight),
                           dma_dst_stride, dma_src_stride, dma_width_bytes, height_A0);

            if (1 < n_chunk_cnt) {
                const size_t   n_cols_A1 = hex_smin(n - 1 * n_chunk_n_cols, n_chunk_n_cols);
                const uint32_t height_A1 = is_quant ? (n_cols_A1 / 32) * n_k_tiles : n_cols_A1;
                dma_queue_push(ctx->dma[0], dma_make_ptr(vtcm_weight_raw[1], weight + n_chunk_n_cols * weight_stride),
                               dma_dst_stride, dma_src_stride, dma_width_bytes, height_A1);
            }

            // Main loop: pop A_i -> dequantize A_i -> push A_{i+2} -> submit C_i -> wait C_{i-1} and store D_{i-1}
            for (int i = 0; i < n_chunk_cnt; ++i) {
                const size_t nc    = i * n_chunk_n_cols;
                const size_t nc_p2 = nc + 2 * n_chunk_n_cols;

                const size_t n_cols    = hex_smin(n - nc, n_chunk_n_cols);
                const size_t n_cols_p2 = hex_smin(n - nc_p2, n_chunk_n_cols);

                // 1. pop A_i
                void * curr_raw = dma_queue_pop(ctx->dma[0]).dst;

                // 2. dequantize A_i
                dequantize_tiled_weight_chunk_to_fp16_tiles(
                    ctx, vtcm_weight_bufs[i % 2], curr_raw,
                    n_cols, k, row_stride, weight_type,
                    n_k_tiles, n_k_tiles_div, dequant_worker_fn, n_threads);

                // 3. push A_{i+2} (if i+2 < n_chunk_cnt)
                if (i + 2 < n_chunk_cnt) {
                    const uint32_t height_p2 = is_quant ? (n_cols_p2 / 32) * n_k_tiles : n_cols_p2;
                    dma_queue_push(ctx->dma[0], dma_make_ptr(curr_raw, weight + nc_p2 * weight_stride),
                                   dma_dst_stride, dma_src_stride, dma_width_bytes, height_p2);
                }

                // 4. submit C_i
                hmx_matmul_job_init(&job_slots[i % 2], (__fp16 *) vtcm_output_bufs[i % 2],
                                    (__fp16 *) vtcm_f16_act, (__fp16 *) vtcm_weight_bufs[i % 2],
                                    vtcm_scales, hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS),
                                    hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS), k / HTP_MM_HMX_TILE_N_ROWS);
                hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job_slots[i % 2]));

                // 5. wait C_{i-1} and store D_{i-1} (multi-thread HVX, parallel with C_i)
                if (i > 0) {
                    hmx_queue_pop(ctx->hmx_queue);
                    const size_t nc_prev = (i - 1) * n_chunk_n_cols;
                    const size_t n_cols_prev = hex_smin(n - nc_prev, n_chunk_n_cols);
                    float *output_chunk = dst + (mr * dst_stride + nc_prev);
                    const float *src2_chunk = src2 ? (src2 + mr * src2_stride + nc_prev) : NULL;
                    int chunk_dst_cols = dst_cols - (int)nc_prev;
                    if (chunk_dst_cols > 0) {
                        transfer_output_chunk_threaded(ctx, output_chunk, src2_chunk, vtcm_output_bufs[(i - 1) % 2], n_rows, n_cols_prev, dst_stride, src2_stride, chunk_dst_cols, n_threads);
                    }
                }
            }

            // Epilogue: wait C_{last} and store D_{last}
            hmx_queue_pop(ctx->hmx_queue);
            const size_t nc_last = (n_chunk_cnt - 1) * n_chunk_n_cols;
            const size_t n_cols_last = hex_smin(n - nc_last, n_chunk_n_cols);
            float *output_chunk = dst + (mr * dst_stride + nc_last);
            const float *src2_chunk = src2 ? (src2 + mr * src2_stride + nc_last) : NULL;
            int chunk_dst_cols = dst_cols - (int)nc_last;
            if (chunk_dst_cols > 0) {
                transfer_output_chunk_threaded(ctx, output_chunk, src2_chunk, vtcm_output_bufs[(n_chunk_cnt - 1) % 2], n_rows, n_cols_last, dst_stride, src2_stride, chunk_dst_cols, n_threads);
            }
        }
    } else {
        // --- Synchronous loop (m <= 32 or fallback) ---
        hmx_matmul_job_t job;
        for (size_t mr = 0; mr < m; mr += m_chunk_n_rows) {
            const size_t n_rows = hex_smin(m - mr, m_chunk_n_rows);

            struct activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .src = activation + mr * act_stride,
                .n_rows = (int) n_rows,
                .k_block = k,
                .k_stride = act_stride,
                .n_threads = act_threads,
                .act_threads_div = act_threads_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_bytes = L.act_f32_bytes,
            };
            transfer_activation_chunk_threaded(&act_params);

            // A0: Pre-fetch the first weight chunk (nc = 0)
            if (n > 0) {
                const size_t n_cols = hex_smin(n, n_chunk_n_cols);
                const uint32_t height = is_quant ? (n_cols / 32) * n_k_tiles : n_cols;
                dma_queue_push(ctx->dma[0], dma_make_ptr(vtcm_weight_raw[0], weight), dma_dst_stride, dma_src_stride, dma_width_bytes, height);
            }

            for (size_t nc = 0; nc < n; nc += n_chunk_n_cols) {
                const size_t n_cols = hex_smin(n - nc, n_chunk_n_cols);
                const size_t n_row_tiles = hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS);
                const size_t n_col_tiles = hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS);

                // A: Wait for weight DMA
                void * curr_raw = dma_queue_pop(ctx->dma[0]).dst;

                // B: Weight Dequantize (Threaded)
                dequantize_tiled_weight_chunk_to_fp16_tiles(
                    ctx, vtcm_scratch0, curr_raw,
                    n_cols, k, row_stride, weight_type,
                    n_k_tiles, n_k_tiles_div, dequant_worker_fn, n_threads);

                // Start weight DMA for the next chunk early
                const size_t nc_next = nc + n_chunk_n_cols;
                if (nc_next < n) {
                    const size_t n_cols_next = hex_smin(n - nc_next, n_chunk_n_cols);
                    const uint32_t height_next = is_quant ? (n_cols_next / 32) * n_k_tiles : n_cols_next;
                    dma_queue_push(ctx->dma[0], dma_make_ptr(curr_raw, weight + nc_next * weight_stride), dma_dst_stride, dma_src_stride, dma_width_bytes, height_next);
                }

                // C: HMX Compute (Queue-based)
                hmx_matmul_job_init(&job, vtcm_output, vtcm_f16_act, vtcm_scratch0, vtcm_scales, n_row_tiles, n_col_tiles, k / HTP_MM_HMX_TILE_N_ROWS);
                hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job));
                hmx_queue_pop(ctx->hmx_queue);

                // D: Output Store
                float *output_chunk = dst + (mr * dst_stride + nc);
                const float *src2_chunk = src2 ? (src2 + mr * src2_stride + nc) : NULL;
                int chunk_dst_cols = dst_cols - (int)nc;
                if (chunk_dst_cols > 0) {
                    transfer_output_chunk_threaded(ctx, output_chunk, src2_chunk, vtcm_output, n_rows, n_cols, dst_stride, src2_stride, chunk_dst_cols, n_threads);
                }
            }
        }
    }

    return 0;
}

static inline int hmx_mm_batch_r2(const hmx_mm_f16_f32_batched_params_t *params) {
    return params->ne02 > 0 ? params->ne12 / params->ne02 : 1;
}

static inline int hmx_mm_batch_r3(const hmx_mm_f16_f32_batched_params_t *params) {
    return params->ne03 > 0 ? params->ne13 / params->ne03 : 1;
}

static inline const __fp16 *hmx_mm_weight_batch_ptr(const hmx_mm_f16_f32_batched_params_t *params,
                                                        int dst_b2, int dst_b3) {
    const int r2 = hmx_mm_batch_r2(params);
    const int r3 = hmx_mm_batch_r3(params);
    return (const __fp16 *) ((const uint8_t *) params->weight +
                             (size_t) (dst_b2 / r2) * params->src0_nb2 +
                             (size_t) (dst_b3 / r3) * params->src0_nb3);
}

static inline const float *hmx_mm_activation_batch_ptr(const hmx_mm_f16_f32_batched_params_t *params,
                                                           int dst_b2, int dst_b3) {
    return (const float *) ((const uint8_t *) params->activation +
                            (size_t) dst_b2 * params->src1_nb2 +
                            (size_t) dst_b3 * params->src1_nb3);
}

static inline float *hmx_mm_dst_batch_ptr(const hmx_mm_f16_f32_batched_params_t *params,
                                              int dst_b2, int dst_b3) {
    return (float *) ((uint8_t *) params->dst +
                      (size_t) dst_b2 * params->dst_nb2 +
                      (size_t) dst_b3 * params->dst_nb3);
}

static inline const float *hmx_mm_src2_batch_ptr(const hmx_mm_f16_f32_batched_params_t *params,
                                               int src2_b2, int src2_b3) {
    return params->src2 ? (const float *) ((const uint8_t *) params->src2 +
                      (size_t) src2_b2 * params->src2_nb2 +
                      (size_t) src2_b3 * params->src2_nb3) : NULL;
}

static int hmx_mm_f16_f32_batched_simple(struct htp_context *ctx,
                                                        const hmx_mm_f16_f32_batched_params_t *params,
                                                        int m_chunk, int n_chunk, int pipeline, int n_threads, int act_threads, int vtcm_size,
                                                        const struct fastdiv_values * act_threads_div, const struct fastdiv_values * k_div) {
    int ret = 0;
    for (int b3 = 0; b3 < params->ne13 && ret == 0; ++b3) {
        for (int b2 = 0; b2 < params->ne12 && ret == 0; ++b2) {
            ret = hmx_mm_2d_f32(ctx, hmx_mm_dst_batch_ptr(params, b2, b3),
                                           hmx_mm_src2_batch_ptr(params, b2, b3),
                                           hmx_mm_activation_batch_ptr(params, b2, b3),
                                           (const uint8_t *)hmx_mm_weight_batch_ptr(params, b2, b3),
                                           params->m, params->k, params->n,
                                           params->act_stride, params->weight_stride * (int)sizeof(__fp16),
                                           HTP_TYPE_F16, params->k, params->dst_stride, params->src2_stride, params->n,
                                           m_chunk, n_chunk, pipeline, n_threads, act_threads,
                                           act_threads_div, k_div, 0, 0, vtcm_size);
        }
    }
    return ret;
}

static int hmx_mm_f16_f32_batched(struct htp_context *ctx, const hmx_mm_f16_f32_batched_params_t *params,
                               int m_chunk, int n_chunk, int pipeline, int n_threads, int act_threads,
                               const struct fastdiv_values * act_threads_div,
                               const struct fastdiv_values * k_div,
                               int vtcm_size) {
    if (params->act_stride < params->k || params->weight_stride < params->k || params->dst_stride < params->n) { return -1; }
    if (params->ne02 <= 0 || params->ne03 <= 0 || params->ne12 <= 0 || params->ne13 <= 0) { return -1; }
    if (params->ne12 % params->ne02 != 0 || params->ne13 % params->ne03 != 0) { return -1; }
    if (params->k % 32 != 0 || params->n % 32 != 0) { return -1; }
    if (!hex_is_aligned(params->dst, VLEN) || !hex_is_aligned(params->activation, VLEN)) { return -1; }

    const int group_size = hmx_mm_batch_r2(params);
    const size_t vtcm_budget  = ctx->vtcm_size;

    // Check if the precomputed parameters are grouped or simple.
    // If simple, or if group_size <= 1, we use simple fallback loop.
    // Grouped path is only valid if group_size > 1 and it fits within VTCM budget.
    bool run_grouped = (group_size > 1 && (size_t)vtcm_size <= vtcm_budget);
    if (!run_grouped) {
        return hmx_mm_f16_f32_batched_simple(ctx, params, m_chunk, n_chunk, pipeline, n_threads, act_threads, vtcm_size, act_threads_div, k_div);
    }

    struct htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    const size_t vec_dot_size = params->k * sizeof(__fp16);

    const bool use_dma_activation = (params->act_stride > params->k);
    const size_t f32_scratch_size = use_dma_activation
        ? hex_align_up((size_t)act_threads * HTP_MM_DMA_ACT_MULTIPLIER * (size_t) params->k * sizeof(float), HTP_MM_HMX_TILE_SIZE) : 0;

    size_t m_chunk_n_rows = m_chunk;
    size_t n_chunk_n_cols = n_chunk;
    size_t vtcm_used = vtcm_size;

    struct htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_F16_BATCHED, HTP_TYPE_F16, params->k, m_chunk_n_rows, n_chunk_n_cols, group_size, use_dma_activation, false, act_threads, 0);

    if (L.total_bytes > vtcm_budget) {
        FARF(HIGH, "%s: grouped layout overflowed VTCM, falling back to simple batched loop", __func__);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);
        return hmx_mm_f16_f32_batched_simple(ctx, params, m_chunk, n_chunk, pipeline, n_threads, act_threads, vtcm_size, act_threads_div, k_div);
    }

    uint8_t * const base = (uint8_t *) ctx->vtcm_base;
    __fp16  *vtcm_weight     = VTCM_LAYOUT_PTR(__fp16, base, L.off_weight[0]);
    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, base, L.off_act);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, base, L.off_dst[0]);
    void    *vtcm_scratch0   = VTCM_LAYOUT_PTR(void, base, L.off_scratch[0]);
    void    *vtcm_scratch1   = VTCM_LAYOUT_PTR(void, base, L.off_scratch[1]);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, base, L.off_scales);
    float   *vtcm_f32_act    = VTCM_LAYOUT_PTR_OPTIONAL(float, base, L.off_act_f32, use_dma_activation);

    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    FARF(HIGH, "%s: grouped path m=%d k=%d n=%d group=%d streams=%d mc=%zu nc=%zu vtcm=%zu/%zu",
            __func__, params->m, params->k, params->n, group_size, params->ne13,
            m_chunk_n_rows, n_chunk_n_cols,
            L.total_bytes, vtcm_budget);

    const size_t fp16_row_bytes   = (size_t) params->k * sizeof(__fp16);
    const size_t weight_row_bytes = (size_t) params->weight_stride * sizeof(__fp16);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hmx_matmul_job_t job;

    for (int b3 = 0; b3 < params->ne13; ++b3) {
        for (int b2_base = 0; b2_base < params->ne12; b2_base += group_size) {
            const __fp16 *weight_group = hmx_mm_weight_batch_ptr(params, b2_base, b3);

            for (size_t mr = 0; mr < (size_t) params->m; mr += m_chunk_n_rows) {
                const size_t n_rows = hex_smin((size_t) params->m - mr, m_chunk_n_rows);
                const size_t n_row_tiles = hmx_ceil_div((int) n_rows, HTP_MM_HMX_TILE_N_ROWS);

                // Pre-load activations for all heads in the group (once per m_chunk).
                // When the source is strided (permuted Q), use 2D DMA to gather
                // contiguous rows into a VTCM scratch buffer first, then HVX
                // converts from the contiguous VTCM buffer.  This avoids L2 cache
                // thrashing from HVX loads at large strides.
                for (int g = 0; g < group_size; ++g) {
                    const float *activation_chunk = hmx_mm_activation_batch_ptr(params, b2_base + g, b3) + mr * params->act_stride;
                    __fp16 *vtcm_act_g = vtcm_f16_act + (size_t) g * L.act_head_stride;
                    struct activation_transfer_params act_params = {
                        .ctx = ctx,
                        .dst = vtcm_act_g,
                        .src = activation_chunk,
                        .n_rows = (int) n_rows,
                        .k_block = params->k,
                        .k_stride = params->act_stride,
                        .n_threads = act_threads,
                        .act_threads_div = act_threads_div,
                        .k_div = k_div,
                        .k_valid = params->k,
                        .vtcm_f32_act = vtcm_f32_act,
                        .vtcm_f32_act_bytes = L.act_f32_bytes,
                    };
                    transfer_activation_chunk_threaded(&act_params);
                }

                // Prologue: Push A0 and A1 (if exists)
                {
                    const size_t n_cols_first = hex_smin((size_t) params->n, n_chunk_n_cols);
                    dma_queue_push(ctx->dma[0], dma_make_ptr(vtcm_scratch0, weight_group),
                                      fp16_row_bytes, weight_row_bytes, fp16_row_bytes, n_cols_first);
                }
                if (n_chunk_n_cols < (size_t) params->n) {
                    const size_t n_cols_second = hex_smin((size_t) params->n - n_chunk_n_cols, n_chunk_n_cols);
                    dma_queue_push(ctx->dma[0], dma_make_ptr(vtcm_scratch1, weight_group + params->weight_stride),
                                      fp16_row_bytes, weight_row_bytes, fp16_row_bytes, n_cols_second);
                }

                for (size_t nc = 0; nc < (size_t) params->n; nc += n_chunk_n_cols) {
                    const size_t n_cols      = hex_smin((size_t) params->n - nc, n_chunk_n_cols);
                    const size_t n_col_tiles = hmx_ceil_div((int) n_cols, HTP_MM_HMX_TILE_N_COLS);

                    {
                        void * curr_raw = dma_queue_pop(ctx->dma[0]).dst;

                        hmx_interleave_rows_to_tiles(vtcm_weight, (const __fp16 *) curr_raw, n_cols, params->k, params->k, 0, n_cols);

                        const size_t nc_next = nc + n_chunk_n_cols * 2;
                        if (nc_next < (size_t) params->n) {
                            const size_t n_cols_next = hex_smin((size_t) params->n - nc_next, n_chunk_n_cols);
                            const __fp16 *next_weight_chunk = weight_group + nc_next * params->weight_stride;

                            dma_queue_push(ctx->dma[0], dma_make_ptr(curr_raw, next_weight_chunk),
                                              fp16_row_bytes, weight_row_bytes, fp16_row_bytes, n_cols_next);
                        }
                    }

                    // Reuse the interleaved weight for every q_head in this GQA group
                    for (int g = 0; g < group_size; ++g) {
                        {
                            const __fp16 * vtcm_act_g = vtcm_f16_act + (size_t) g * L.act_head_stride;
                            hmx_matmul_job_init(&job, vtcm_output, vtcm_act_g, vtcm_weight, vtcm_scales, n_row_tiles, n_col_tiles, params->k / 32);
                            hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job));
                            hmx_queue_pop(ctx->hmx_queue);
                        }

                        {
                            float *output = hmx_mm_dst_batch_ptr(params, b2_base + g, b3) + mr * params->dst_stride + nc;
                            const float *src2_chunk = params->src2 ? (hmx_mm_src2_batch_ptr(params, b2_base + g, b3) + mr * params->src2_stride + nc) : NULL;
                            int chunk_dst_cols = params->n - (int)nc;
                            if (chunk_dst_cols > 0) {
                                transfer_output_chunk_threaded(ctx, output, src2_chunk, vtcm_output, (int) n_rows, (int) n_cols,
                                                               params->dst_stride, params->src2_stride, chunk_dst_cols, ctx->n_threads);
                            }
                        }
                    }
                }
            }
        }
    }

    return 0;
}

static void transfer_activation_chunk_gathered_threaded(
            struct htp_context *ctx,
            __fp16 *dst,
            const float *src,
            int start_row,
            int n_rows,
            int k_block,
            const struct mmid_row_mapping *matrix_rows,
            int cur_a,
            int mapping_stride,
            int ne11,
            size_t nb11,
            size_t nb12,
            int cne1,
            int n_threads,
            int k_valid) {
    if (n_rows <= 0 || start_row < 0 || start_row >= cne1) return;

    const uint32_t valid_rows = (uint32_t) hex_smin(
        (size_t) n_rows, (size_t) (cne1 - start_row));
    if (valid_rows == 0) return;

    activation_transfer_gathered_task_state_t state = {
        .dst               = dst,
        .src               = src,
        .n_tasks           = 1,
        .n_tot_chunks      = valid_rows,
        .n_chunks_per_task = valid_rows,
        .k_block           = k_block,
        .matrix_rows       = matrix_rows,
        .cur_a             = cur_a,
        .mapping_stride    = mapping_stride,
        .ne11              = ne11,
        .ne11_div          = ne11 > 1 ? init_fastdiv_values(ne11) : (struct fastdiv_values){0, 0},
        .nb11              = nb11,
        .nb12              = nb12,
        .start_row         = start_row,
        .cne1              = cne1,
        .k_valid           = k_valid,
        .traces            = ctx->trace,
    };

    worker_callback_t worker_fn = ne11 == 1 ? transfer_activation_chunk_gathered_worker_flat_fn :
                                              transfer_activation_chunk_gathered_worker_fn;

    // Each gathered helper pads its call to a complete 32-row HMX tile.  If a
    // tile is split between workers, those padded writes overlap and workers
    // whose start row is beyond cne1 underflow the remaining-row calculation.
    // Keep one gather call per VTCM tile chunk; the inner conversion remains
    // vectorized across K.
    worker_fn(1, 0, &state);
    (void) n_threads;
}

static void transfer_output_chunk_scattered_threaded(
            struct htp_context *ctx,
            float *dst,
            const __fp16 *vtcm_src,
            int start_row,
            int n_rows,
            int n_cols,
            const struct mmid_row_mapping *matrix_rows,
            int cur_a,
            int mapping_stride,
            size_t dst_nb1,
            size_t dst_nb2,
            int cne1,
            int n_threads) {
    if (n_rows <= 0 || start_row < 0 || start_row >= cne1) return;

    const uint32_t valid_rows = (uint32_t) hex_smin(
        (size_t) n_rows, (size_t) (cne1 - start_row));
    if (valid_rows == 0) return;

    output_transfer_scattered_task_state_t state = {
        .vtcm_src          = vtcm_src,
        .dst               = dst,
        .n_tasks           = 1,
        .n_tot_chunks      = valid_rows,
        .n_chunks_per_task = valid_rows,
        .n_cols            = n_cols,
        .matrix_rows       = matrix_rows,
        .cur_a             = cur_a,
        .mapping_stride    = mapping_stride,
        .dst_nb1           = dst_nb1,
        .dst_nb2           = dst_nb2,
        .start_row         = start_row,
        .cne1              = cne1,
        .traces            = ctx->trace,
    };

    transfer_output_chunk_scattered_worker_fn(1, 0, &state);
    (void) n_threads;
}

static int hmx_mm_id_2d_f32(struct htp_context *ctx,
                                         float *restrict dst,
                                         const float *activation,
                                         const uint8_t *weight,
                                         int m, int k, int n,
                                         int k_valid,
                                         int ne11,
                                         size_t act_nb1, size_t act_nb2,
                                         size_t dst_nb1, size_t dst_nb2,
                                         int weight_stride,
                                         int weight_type,
                                         bool raw_q4_0,
                                         const struct mmid_row_mapping *matrix_rows,
                                         int cur_a,
                                         int mapping_stride) {
    struct htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    const int cne1 = m;
    const int m_padded = hex_align_up(m, 32);

    if (k % 32 != 0 || n % 32 != 0) { return -1; }
    if (!hex_is_aligned(dst, VLEN) || !hex_is_aligned(activation, VLEN)) { return -1; }
    if (raw_q4_0 &&
        (weight_type != HTP_TYPE_Q4_0 ||
         weight_stride != (k / 32) * (int) HTP_RAW_Q4_0_BLOCK_BYTES)) {
        FARF(ERROR, "hmx-mm-id-2d: invalid raw Q4_0 row stride %d for k %d", weight_stride, k);
        return -1;
    }

    size_t row_stride = htp_mm_get_tiled_row_stride(weight_type, k);
    if (row_stride == 0) {
        return -1;
    }

    worker_callback_t dequant_worker_fn = NULL;
    switch (weight_type) {
        case HTP_TYPE_Q4_0:   dequant_worker_fn = dequantize_tiled_worker_loop_q4_0; break;
        case HTP_TYPE_IQ4_NL: dequant_worker_fn = dequantize_tiled_worker_loop_iq4_nl; break;
        case HTP_TYPE_Q4_1:   dequant_worker_fn = dequantize_tiled_worker_loop_q4_1; break;
        case HTP_TYPE_MXFP4:  dequant_worker_fn = dequantize_tiled_worker_loop_mxfp4; break;
        case HTP_TYPE_Q8_0:   dequant_worker_fn = dequantize_tiled_worker_loop_q8_0; break;
        case HTP_TYPE_F16:    dequant_worker_fn = convert_f16_worker_loop; break;
        case HTP_TYPE_F32:    dequant_worker_fn = quantize_f32_worker_loop; break;
        default:
            return -1;
    }

    const int n_k_tiles = k / HTP_MM_HMX_TILE_N_COLS;
    const struct fastdiv_values n_k_tiles_div = init_fastdiv_values(n_k_tiles);

    const int n_threads = ctx->n_threads;
    const bool is_quant   = (weight_type != HTP_TYPE_F16 && weight_type != HTP_TYPE_F32);

    const size_t vec_dot_size = k * sizeof(__fp16);
    const size_t vtcm_budget  = ctx->vtcm_size;
    size_t vtcm_used = 0;

    int tile_size = htp_mm_get_weight_tile_size(weight_type);
    int aligned_tile_size = htp_mm_get_weight_aligned_tile_size(weight_type);

    const uint32_t dma_dst_stride  = is_quant ? aligned_tile_size : row_stride;
    const uint32_t dma_src_stride  = is_quant ? tile_size : weight_stride;
    const uint32_t dma_width_bytes = is_quant ? tile_size : row_stride;

    const size_t qweight_row_stride = is_quant ? (size_t)(n_k_tiles * aligned_tile_size) / 32 : 0;
    const size_t weight_row_stride = is_quant ? qweight_row_stride : row_stride;

    size_t size_per_n = 0, size_per_m = 0, size_per_mn = 0;
    htp_mm_hmx_get_2d_chunk_costs(weight_type, k, /*pipeline=*/false, aligned_tile_size,
                                  &size_per_n, &size_per_m, &size_per_mn);

    size_t m_chunk_n_rows = 0, n_chunk_n_cols = 0;
    if (htp_mm_hmx_compute_chunks(vtcm_budget, /*overhead=*/256, size_per_n, size_per_m, size_per_mn,
                           m_padded, n,
                           /*m_block_cost=*/(size_t) n * HTP_MM_HMX_COST_W_DEQUANT,
                           /*n_block_cost=*/(size_t) m_padded * HTP_MM_HMX_COST_A_CONVERT, &m_chunk_n_rows, &n_chunk_n_cols, &vtcm_used)) {
        FARF(ERROR, "hmx-mm-id-2d: VTCM too small : m %d k %d n %d budget %zu", m_padded, k, n, vtcm_budget);
        return -1;
    }

    const size_t weight_area_size = hex_align_up(n_chunk_n_cols * weight_row_stride, HTP_MM_HMX_TILE_SIZE);
    const size_t act_area_size    = hex_align_up(m_chunk_n_rows * vec_dot_size, HTP_MM_HMX_TILE_SIZE);
    const size_t output_area_size = hex_align_up(m_chunk_n_rows * n_chunk_n_cols * sizeof(__fp16), HTP_MM_HMX_TILE_SIZE);

    size_t scratch0_size = hex_align_up(n_chunk_n_cols * vec_dot_size, HTP_MM_HMX_TILE_SIZE);

    uint8_t *vtcm_ptr      = (uint8_t *) ctx->vtcm_base;
    __fp16  *vtcm_weight   = weight_area_size ? (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, weight_area_size) : NULL;
    __fp16  *vtcm_f16_act  = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, act_area_size);
    __fp16  *vtcm_output   = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, output_area_size);
    void    *vtcm_scratch0 = vtcm_seq_alloc(&vtcm_ptr, scratch0_size);
    __fp16  *vtcm_scales   = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);

    vtcm_used = vtcm_ptr - (uint8_t *) ctx->vtcm_base;
    if (vtcm_used > vtcm_budget) {
        FARF(ERROR, "hmx-mm-id-2d: VTCM overflow: used %zu budget %zu", vtcm_used, vtcm_budget);
        return -1;
    }

    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hmx_matmul_job_t job;

    for (size_t mr = 0; mr < (size_t) m_padded; mr += m_chunk_n_rows) {
        const size_t n_rows = hex_smin(m_padded - mr, m_chunk_n_rows);
        const size_t n_row_tiles = hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS);

        transfer_activation_chunk_gathered_threaded(
            ctx, vtcm_f16_act, activation, (int) mr, (int) n_rows, k,
            matrix_rows, cur_a, mapping_stride, ne11, act_nb1, act_nb2, cne1, n_threads, k_valid);

        // A0: Pre-fetch the first already-tiled weight chunk (nc = 0).
        // Raw Q4_0 is loaded synchronously per chunk below because the raw
        // staging area reuses vtcm_scratch0 before that area becomes FP16.
        if (!raw_q4_0 && n > 0) {
            const size_t n_cols = hex_smin((size_t) n, n_chunk_n_cols);
            const uint32_t height = is_quant ? (n_cols / 32) * n_k_tiles : n_cols;
            dma_queue_push(ctx->dma[0], dma_make_ptr(vtcm_weight, weight),
                           dma_dst_stride, dma_src_stride, dma_width_bytes, height);
        }

        for (size_t nc = 0; nc < (size_t) n; nc += n_chunk_n_cols) {
            const size_t n_cols = hex_smin((size_t) n - nc, n_chunk_n_cols);
            const size_t n_col_tiles = hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS);

            // A: Load one weight chunk. In low-memory raw-Q4_0 mode, stage raw
            // GGUF rows in the existing FP16 scratch area, then convert only
            // this chunk into the existing tiled-weight area. No model-sized
            // HTP_REPACK allocation and no additional VTCM reservation.
            void * curr_tiled = NULL;
            if (raw_q4_0) {
                const size_t raw_row_size = (size_t) weight_stride;
                const size_t raw_chunk_size = n_cols * raw_row_size;
                if (raw_chunk_size > scratch0_size) {
                    FARF(ERROR,
                         "hmx-mm-id-2d: raw chunk too large: raw %zu scratch %zu",
                         raw_chunk_size, scratch0_size);
                    return -1;
                }

                dma_queue_push(
                    ctx->dma[0],
                    dma_make_ptr(vtcm_scratch0, weight + nc * raw_row_size),
                    raw_row_size, raw_row_size, raw_row_size, n_cols);
                uint8_t * raw_chunk = (uint8_t *) dma_queue_pop(ctx->dma[0]).dst;
                uint8_t * tiled_chunk = (uint8_t *) vtcm_weight;
                htp_raw_q4_0_to_tiled_threaded(
                    ctx, raw_chunk, raw_row_size, tiled_chunk,
                    n_cols, (uint32_t) k, (uint32_t) n_threads);
                curr_tiled = vtcm_weight;
            } else {
                // Wait for the already-tiled weight DMA queued above.
                curr_tiled = dma_queue_pop(ctx->dma[0]).dst;
            }

            // B: Weight Dequantize (Threaded)
            dequantize_tiled_weight_chunk_to_fp16_tiles(
                ctx, vtcm_scratch0, curr_tiled,
                n_cols, k, row_stride, weight_type,
                n_k_tiles, n_k_tiles_div, dequant_worker_fn, n_threads
            );

            // Start the next already-tiled DMA early. Raw Q4_0 cannot prefetch
            // here because vtcm_scratch0 now contains the current FP16 weights.
            const size_t nc_next = nc + n_chunk_n_cols;
            if (!raw_q4_0 && nc_next < (size_t) n) {
                const size_t n_cols_next = hex_smin((size_t) n - nc_next, n_chunk_n_cols);
                const uint32_t height_next = is_quant ? (n_cols_next / 32) * n_k_tiles : n_cols_next;
                dma_queue_push(ctx->dma[0], dma_make_ptr(curr_tiled, weight + nc_next * weight_stride),
                               dma_dst_stride, dma_src_stride, dma_width_bytes, height_next);
            }

            // C: HMX Compute (Queue-based)
            hmx_matmul_job_init(&job, vtcm_output, vtcm_f16_act, vtcm_scratch0, vtcm_scales, n_row_tiles, n_col_tiles, k / HTP_MM_HMX_TILE_N_ROWS);
            hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job));
            hmx_queue_pop(ctx->hmx_queue);

            // D: Output Store
            transfer_output_chunk_scattered_threaded(
                ctx, dst + nc, vtcm_output, (int) mr, (int) n_rows, (int) n_cols,
                matrix_rows, cur_a, mapping_stride, dst_nb1, dst_nb2, cne1, n_threads);
        }
    }

    return 0;
}

// --- Dispatchers and Public Entry Points ---

static int hmx_mm_op_matmul(struct htp_ops_context * octx, const struct htp_mm_kernel_params * kparams) {
    htp_matmul_tensors_preamble;

    int k = (int) src0->ne[0];
    int n = (int) src0->ne[1];
    const int m_total    = (int) src1->ne[1];
    const int act_stride = (int)(src1->nb[1] / sizeof(float));
    const int wgt_stride = (int)(src0->nb[1] / sizeof(__fp16));

    const float * src2_ptr = NULL;
    uint32_t src2_stride = 0;
    size_t src2_nb2 = 0;
    size_t src2_nb3 = 0;
    if (src2) {
        src2_ptr = (const float *) src2->data;
        src2_stride = (src2->ne[1] == 1) ? 0 : (uint32_t) (src2->nb[1] / sizeof(float));
        src2_nb2 = (src2->ne[2] == 1) ? 0 : src2->nb[2];
        src2_nb3 = (src2->ne[3] == 1) ? 0 : src2->nb[3];
    }

    int ret = -1;
    const int n_threads = MIN(kparams->n_threads, (int) octx->n_threads);
    if (kparams->kernel_type == HTP_MM_KERNEL_HMX_F16_BATCHED) {
        hmx_mm_f16_f32_batched_params_t batch_params = {
            .dst             = (float *) dst->data,
            .src2            = src2_ptr,
            .activation      = (float *) src1->data,
            .weight          = (const __fp16 *) src0->data,
            .m               = m_total,
            .k               = k,
            .n               = n,
            .act_stride      = act_stride,
            .weight_stride   = wgt_stride,
            .dst_stride      = (int) (dst->nb[1] / sizeof(float)),
            .src2_stride     = src2_stride,
            .ne02            = ne02,
            .ne03            = ne03,
            .ne12            = ne12,
            .ne13            = ne13,
            .src0_nb2        = src0->nb[2],
            .src0_nb3        = src0->nb[3],
            .src1_nb2        = src1->nb[2],
            .src1_nb3        = src1->nb[3],
            .dst_nb2         = dst->nb[2],
            .dst_nb3         = dst->nb[3],
            .src2_nb2        = src2_nb2,
            .src2_nb3        = src2_nb3,
        };
        ret = hmx_mm_f16_f32_batched(octx->ctx, &batch_params,
                                     kparams->m_chunk, kparams->n_chunk,
                                     kparams->pipeline, n_threads,
                                     kparams->n_act_threads,
                                     &kparams->div_n_act_threads,
                                     &kparams->div_ne00_padded,
                                     kparams->vtcm_size);
    } else {
        ret = hmx_mm_2d_f32(
            octx->ctx, (float*) dst->data, src2_ptr, (float*) src1->data, (const uint8_t *) src0->data,
            m_total, k, n, act_stride, (int) src0->nb[1], (int) src0->type, (int) src1->ne[0],
            (int)(dst->nb[1] / sizeof(float)), src2_stride, (int)dst->ne[0],
            kparams->m_chunk, kparams->n_chunk, kparams->pipeline, n_threads,
            kparams->n_act_threads,
            &kparams->div_n_act_threads,
            &kparams->div_ne00_padded,
            kparams->tile_size, kparams->aligned_tile_size, kparams->vtcm_size
        );
    }

    if (ret != 0) {
        FARF(ERROR, "HMX matmul failed (ret=%d)\n", ret);
        return HTP_STATUS_INTERNAL_ERR;
    }
    return HTP_STATUS_OK;
}

int op_matmul(struct htp_ops_context * octx) {
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;

    if (kparams->n_hmx) {
        return hmx_mm_op_matmul(octx, kparams);
    }

    return hvx_mm_matmul(octx);
}

static int hmx_mm_op_matmul_id(
    struct htp_ops_context * octx,
    struct htp_mm_context * mmctx
) {
    const uint32_t *                matrix_row_counts = mmctx->matrix_row_counts;
    const struct mmid_row_mapping * matrix_rows       = mmctx->matrix_rows;
    htp_matmul_tensors_preamble;
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const int n_ids = octx->src[2]->ne[0];
    const int n_as  = ne02;
    const bool raw_q4_0 = kparams->kernel_type == HTP_MM_KERNEL_HMX_2D_RAW_Q4_0;

    for (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
        const int32_t cne1 = matrix_row_counts[cur_a];
        if (cne1 == 0) continue;

        int ret = hmx_mm_id_2d_f32(octx->ctx, (float*) dst->data, (float*) src1->data,
                                   (const uint8_t *) src0->data + cur_a * nb02,
                                   cne1, ne00, ne01,
                                   ne10,
                                   ne11,
                                   nb11, nb12,
                                   nb1, nb2,
                                   (int) src0->nb[1], (int) src0->type,
                                   raw_q4_0,
                                   matrix_rows, cur_a, mmctx->mapping_stride);
        if (ret != 0) {
            FARF(ERROR, "HMX matmul failed for expert %u, error %d\n", cur_a, ret);
            return HTP_STATUS_NO_SUPPORT;
        }
    }

    return HTP_STATUS_OK;
}

static int hvx_mm_matmul_id(
    struct htp_ops_context * octx,
    struct htp_mm_context * mmctx,
    work_queue_func_t hvx_mmid_task_func
) {
    htp_matmul_tensors_preamble;
    const uint32_t src0_row_size_padded = mmctx->src0_row_size_padded;
    const uint32_t src1_nrows           = mmctx->src1_nrows;

    struct htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const struct htp_tensor * restrict ids = octx->src[2];
    const size_t src0_row_size = nb01;

    const uint32_t qk = QK_Q8_0_TILED;
    const uint32_t nb = (ne10 + qk - 1) / qk;
    const uint32_t total_nb = src1_nrows * nb;

    work_queue_func_t quant_task_func;
    uint32_t n_quant_tasks = 1;

    uint32_t dbg_flags = 0;
    if ((uint32_t) octx->kernel_params[HTP_MM_DEBUG_CTRL_WORD_MAGIC] ==
        HTP_MM_DEBUG_CTRL_MAGIC) {
        dbg_flags = (uint32_t) octx->kernel_params[HTP_MM_DEBUG_CTRL_WORD_FLAGS];
    }
    const bool force_quant_block =
        (dbg_flags & HTP_MM_DEBUG_FLAG_FORCE_QUANT_BLOCK) != 0;
    const bool force_quant_row =
        (dbg_flags & HTP_MM_DEBUG_FLAG_FORCE_QUANT_ROW) != 0;
    const bool use_quant_block = force_quant_block ||
        (!force_quant_row && src1_nrows < octx->n_threads);

    if (use_quant_block) {
        mmctx->dbg_v121_quant_mode = HTP_MMID_QUANT_BLOCK;
        n_quant_tasks = MIN(total_nb, octx->n_threads);
        quant_task_func = (src0->type == HTP_TYPE_Q4_1) ? quantize_f32_q8_1_tiled_block : quantize_f32_q8_0_tiled_block;
        for (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_first = (total_nb * ith) / n_quant_tasks;
            uint32_t ib_last  = (total_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_first[ith] = ib_first;
            mmctx->quant_ib_last[ith]  = ib_last;
            mmctx->quant_r[ith]        = ib_first / nb;
            mmctx->quant_c[ith]        = ib_first % nb;
        }
    } else {
        mmctx->dbg_v121_quant_mode = HTP_MMID_QUANT_ROW;
        n_quant_tasks = MIN(src1_nrows, octx->n_threads);
        quant_task_func = (src0->type == HTP_TYPE_Q4_1) ? quantize_f32_q8_1_tiled : quantize_f32_q8_0_tiled;
    }
    size_t src1_row_size  = (src0->type == HTP_TYPE_Q4_1) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);

    struct htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, ne10, src1_nrows, octx->n_threads,
                                 0, src0_row_size, src1_row_size, 0, kparams->n_prefetch, true, false, false);

    size_t vtcm_size = kparams->vtcm_size > 0 ? (size_t)kparams->vtcm_size : L.total_bytes;

    FARF(HIGH, "matmul-id-%s : src0-spad-size %zu src1-spad-size %zu src2-spad-size 0 dst-spad-size %zu (%zu)\n", mmctx->type,
         L.src0_bytes, L.src1_bytes, L.dst_bytes, vtcm_size);

    FARF(HIGH, "matmul-id-%s : %ux%ux%ux%u * %ux%ux%ux%u (%ux%ux%ux%u) -> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", mmctx->type,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
         ids->ne[0], ids->ne[1], ids->ne[2], ids->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], src0->data,
         src1->data, dst->data);

    // Make sure the reserved vtcm size is sufficient
    if (octx->ctx->vtcm_size < vtcm_size) {
        FARF(ERROR, "matmul-id-%s : current VTCM reservation %zu is too small, needed %zu\n", mmctx->type, octx->ctx->vtcm_size, vtcm_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * const base = (uint8_t *) octx->ctx->vtcm_base;
    mmctx->vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src1);
    mmctx->vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
    mmctx->vtcm_src2 = NULL;
    mmctx->vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);

    octx->src1_spad.src  = NULL;
    octx->src0_spad.src  = NULL;
    octx->src2_spad.src  = NULL;
    octx->dst_spad.src   = NULL;

    mmctx->vtcm_src0_stride = src0_row_size_padded;
    mmctx->vtcm_src1_stride = src1_row_size;

    mmctx->vtcm_src0_size_per_thread = L.src0_bytes / octx->n_threads;
    mmctx->vtcm_src1_size_per_thread = L.src1_bytes;
    mmctx->vtcm_src2_size_per_thread = 0;
    mmctx->vtcm_dst_size_per_thread  = L.dst_bytes / octx->n_threads;

    mmctx->n_quant_rows_per_thread = (src1_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->quant_task_func = quant_task_func;
    mmctx->n_quant_tasks = n_quant_tasks;
    atomic_init(&mmctx->quant_barrier, n_quant_tasks);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    worker_pool_run_func(octx->ctx->worker_pool, hvx_mmid_task_func, mmctx, octx->n_threads);

    return HTP_STATUS_OK;
}

static inline void scan_expert_ids_n(
    const struct htp_tensor * ids,
    const uint32_t n_ids,
    uint32_t n_as,
    uint32_t * counts,
    struct mmid_row_mapping * matrix_rows,
    uint32_t mapping_stride
) {
    const size_t ids_nb1 = ids->nb[1];
    const uint8_t * ids_data = (const uint8_t *) ids->data;

    for (uint32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
        const int32_t * row_ptr = (const int32_t *) (ids_data + iid1 * ids_nb1);
        for (uint32_t id = 0; id < n_ids; ++id) {
            const int32_t i02 = row_ptr[id];
            if (i02 < 0) {
                continue;
            }
            assert(i02 < n_as);

            if (matrix_rows) {
                matrix_rows[i02 * mapping_stride + counts[i02]] = (struct mmid_row_mapping) { id, iid1 };
            }
            counts[i02] += 1;
        }
    }
}

static inline void scan_expert_ids(
    const struct htp_tensor * ids,
    uint32_t n_ids,
    uint32_t n_as,
    uint32_t * counts,
    struct mmid_row_mapping * matrix_rows,
    uint32_t mapping_stride
) {
    const size_t ids_nb0 = ids->nb[0];

    if (ids_nb0 == 4) {
        switch (n_ids) {
            case 8:  scan_expert_ids_n(ids, 8,     n_as, counts, matrix_rows, mapping_stride); break;
            case 4:  scan_expert_ids_n(ids, 4,     n_as, counts, matrix_rows, mapping_stride); break;
            case 2:  scan_expert_ids_n(ids, 2,     n_as, counts, matrix_rows, mapping_stride); break;
            default: scan_expert_ids_n(ids, n_ids, n_as, counts, matrix_rows, mapping_stride); break;
        }
    } else {
        // Strided fallback
        const size_t ids_nb1 = ids->nb[1];
        const uint8_t * ids_data = (const uint8_t *) ids->data;
        for (uint32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            const int32_t * row_ptr = (const int32_t *) (ids_data + iid1 * ids_nb1);
            for (uint32_t id = 0; id < n_ids; ++id) {
                const int32_t i02 = *(const int32_t *) ((const uint8_t *) row_ptr + id * ids_nb0);
                if (i02 < 0) {
                    continue;
                }
                assert(i02 < n_as);

                if (matrix_rows) {
                    matrix_rows[i02 * mapping_stride + counts[i02]] = (struct mmid_row_mapping) { id, iid1 };
                }
                counts[i02] += 1;
            }
        }
    }
}

int op_matmul_id(struct htp_ops_context * octx) {
    htp_matmul_tensors_preamble;

    struct htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    struct htp_mm_context mmctx_struct = {0};
    struct htp_mm_context * mmctx = &mmctx_struct;
    mmctx->octx = octx;

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;

    const struct htp_tensor * restrict ids = octx->src[2];

    const size_t src0_row_size = nb01;
    const size_t dst_row_size  = nb1;

    const size_t src0_row_size_padded = hex_round_up(src0_row_size, 128);

    const uint32_t src0_nrows = ne01;  // per expert
    const uint32_t src1_nrows = ne11 * ne12 * ne13;

    mmctx->src0_nrows_per_thread = (src0_nrows + octx->n_threads - 1) / octx->n_threads;
    mmctx->src0_nrows_per_thread = hex_round_up(mmctx->src0_nrows_per_thread, 32);

    // row groups
    const int n_ids = ids->ne[0];  // n_expert_used
    const int n_as  = ne02;        // n_expert

    uint8_t  * mapping_buf       = octx->ctx->ddr_spad_base;
    uint32_t   mapping_stride    = 1;
    uint32_t * matrix_row_counts = (uint32_t *) mapping_buf;
    struct mmid_row_mapping * matrix_rows = NULL;

    if (src1_nrows > 1) {
        const size_t matrix_row_counts_size = n_as * sizeof(uint32_t);
        assert(octx->ctx->ddr_spad_size >= matrix_row_counts_size);

        hex_l2fetch_block((const void *) ids->data, ids->ne[1] * ids->nb[1]);

        memset(matrix_row_counts, 0, matrix_row_counts_size);
        scan_expert_ids(ids, n_ids, n_as, matrix_row_counts, NULL, 0);

        uint32_t max_count = hvx_reduce_max_i32((const uint8_t *) matrix_row_counts, n_as);
        mapping_stride = max_count > 0 ? max_count : 1;

        size_t matrix_row_map_size  = n_as * mapping_stride * sizeof(struct mmid_row_mapping);
        const size_t total_map_size = matrix_row_counts_size + matrix_row_map_size;

        if (total_map_size > octx->ctx->ddr_spad_size) {
            mapping_buf = memalign(128, total_map_size);
            if (!mapping_buf) {
                return HTP_STATUS_INTERNAL_ERR;
            }
        }

        matrix_row_counts = (uint32_t *) mapping_buf;
        matrix_rows       = (struct mmid_row_mapping *) (mapping_buf + matrix_row_counts_size);

        memset(matrix_row_counts, 0, n_as * sizeof(uint32_t));
        scan_expert_ids(ids, n_ids, n_as, matrix_row_counts, matrix_rows, mapping_stride);
    }

    mmctx->matrix_row_counts    = matrix_row_counts;
    mmctx->matrix_rows          = matrix_rows;
    mmctx->mapping_stride       = mapping_stride;
    mmctx->mm_div_ne11          = kparams->div_ne11;
    mmctx->src0_row_size_padded = src0_row_size_padded;
    mmctx->src1_nrows           = src1_nrows;

    // v10.12: the first 32 kernel-param words are the normal matmul payload.
    // Extra words 32..37 carry runtime debug controls and are not used by compute.
    const int32_t * dbg_ctrl_words = (const int32_t *) octx->kernel_params;
    if ((uint32_t) dbg_ctrl_words[HTP_MM_DEBUG_CTRL_WORD_MAGIC] == HTP_MM_DEBUG_CTRL_MAGIC) {
        mmctx->dbg_v112_enabled      = 1;
        mmctx->dbg_v112_want_expert = dbg_ctrl_words[HTP_MM_DEBUG_CTRL_WORD_EXPERT];
        mmctx->dbg_v112_want_ct      = (uint32_t) MAX(dbg_ctrl_words[HTP_MM_DEBUG_CTRL_WORD_CT], 0);
        mmctx->dbg_v112_want_cid     = (uint32_t) MAX(dbg_ctrl_words[HTP_MM_DEBUG_CTRL_WORD_CID], 0);
        uint32_t dk = (uint32_t) MAX(dbg_ctrl_words[HTP_MM_DEBUG_CTRL_WORD_K], 32);
        dk = (dk / 32u) * 32u;
        mmctx->dbg_v112_want_k = dk;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    int s;
    if (kparams->n_hmx) {
        s = hmx_mm_op_matmul_id(octx, mmctx);
    } else if (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW_RAW_Q4_0 &&
               src0->type == HTP_TYPE_Q4_0) {
        // Raw Q4_0 staging path: convert each 32-row working set in VTCM.
        s = hvx_mm_matmul_id(octx, mmctx,
                src1_nrows > 1 ? hvx_mm_id_raw_q4_0 : hvx_mv_id_raw_q4_0);
    } else {
        if (hvx_mm_init_vec_dot(mmctx, src0->type) == 0) {
            s = hvx_mm_matmul_id(octx, mmctx, src1_nrows > 1 ? hvx_mm_id : hvx_mv_id);
        } else {
            s = HTP_STATUS_NO_SUPPORT;
        }
    }

    if (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW_RAW_Q4_0 &&
        mmctx->dbg_v103_claimed) {
        struct htp_mm_kernel_params * dbg_kparams =
            (struct htp_mm_kernel_params *) octx->kernel_params;

        // The op has finished, so these execution fields are dead. Reuse five int32 slots
        // as a tiny DSP->host return channel without changing the fixed 128-byte ABI.
        dbg_kparams->kernel_type   = (int32_t) HTP_MM_DEBUG_RETURN_MAGIC;
        dbg_kparams->pipeline      = (int32_t) mmctx->dbg_v103_expert;
        dbg_kparams->m_chunk       = (int32_t) mmctx->dbg_v103_raw_fnv;
        dbg_kparams->n_chunk       = (int32_t) mmctx->dbg_v103_tile_fnv;
        dbg_kparams->n_threads     = (int32_t) mmctx->dbg_v103_src_off;
        dbg_kparams->n_act_threads    = (int32_t) mmctx->dbg_v106_ct;
        dbg_kparams->n_hmx            = (int32_t) mmctx->dbg_v106_ith;
        dbg_kparams->n_prefetch       = (int32_t) mmctx->dbg_v106_valid_rows;

        dbg_kparams->tile_size        = (int32_t) mmctx->dbg_v107_ne00;
        dbg_kparams->aligned_tile_size= (int32_t) mmctx->dbg_v107_ne10;
        dbg_kparams->src1_row_size    = (int32_t) mmctx->dbg_v107_nb01;
        dbg_kparams->vtcm_size        = (int32_t) mmctx->dbg_v107_nb02;
        dbg_kparams->vtcm_src0_size   = (int32_t) mmctx->dbg_v107_src1_stride;
        dbg_kparams->vtcm_src1_size   = (int32_t) mmctx->dbg_v107_rm1;
        dbg_kparams->vtcm_src2_size   = (int32_t) mmctx->dbg_v107_rm2;
        dbg_kparams->vtcm_src3_size   = (int32_t) mmctx->dbg_v107_ir1;
        dbg_kparams->vtcm_dst_size    = (int32_t) mmctx->dbg_v107_src1_off;

        // v10.8.1: kernel_params is a fixed 128-byte blob. The op has finished,
        // so slots 17..20 are dead fastdiv storage and can safely carry debug
        // output bits to main.c. Use the raw int32 view instead of nonexistent
        // named struct fields.
        int32_t * dbg_words = (int32_t *) dbg_kparams;
        dbg_words[17] = (int32_t) mmctx->dbg_v108_out0;
        dbg_words[18] = (int32_t) mmctx->dbg_v108_out1;
        dbg_words[19] = (int32_t) mmctx->dbg_v108_out2;
        dbg_words[20] = (int32_t) mmctx->dbg_v108_out3;

        dbg_words[21] = (int32_t) mmctx->dbg_v109_quant_fnv;
        dbg_words[22] = (int32_t) mmctx->dbg_v109_dot_fnv;
        dbg_words[23] = (int32_t) mmctx->dbg_v109_q8_head;
        dbg_words[24] = (int32_t) mmctx->dbg_v109_scale_head;
        dbg_words[25] = (int32_t) mmctx->dbg_v109_src_f0_bits;
        dbg_words[26] = (int32_t) mmctx->dbg_v109_src_max_bits;
        dbg_words[27] = (int32_t) mmctx->dbg_v109_row_fnv;

        // v10.11: last four words of the existing 128-byte kernel_params blob.
        dbg_words[28] = (int32_t) mmctx->dbg_v111_hvx_k32_bits;
        dbg_words[29] = (int32_t) mmctx->dbg_v111_ref_k32_bits;
        dbg_words[30] = (int32_t) mmctx->dbg_v111_int_dot;
        dbg_words[31] = (int32_t) mmctx->dbg_v111_scales_fp16;
        // v10.14: exact HVX integer accumulator, row 0, selected tile.
        dbg_words[16] = mmctx->dbg_v114_hvx_int_dot;

        // v10.15 packed element diagnostics in debug-only kernel_params tail.
        for (uint32_t pi = 0; pi < 8; ++pi) {
            uint32_t wa = 0, ws = 0, wq = 0;
            memcpy(&wa, &mmctx->dbg_v115_q8_actual[pi * 4u], 4);
            memcpy(&ws, &mmctx->dbg_v115_q8_scalar[pi * 4u], 4);
            memcpy(&wq, &mmctx->dbg_v115_q4_weight[pi * 4u], 4);
            dbg_words[40 + pi] = (int32_t) wa;
            dbg_words[48 + pi] = (int32_t) ws;
            dbg_words[56 + pi] = (int32_t) wq;
        }
        for (uint32_t pi = 0; pi < 16; ++pi) {
            uint32_t wd = 0;
            memcpy(&wd, &mmctx->dbg_v115_prod_delta[pi * 2u], 4);
            dbg_words[64 + pi] = (int32_t) wd;
        }
        dbg_words[112] = (int32_t) mmctx->dbg_v118_full_ref_bits;
    }

    if (mapping_buf != octx->ctx->ddr_spad_base) {
        free(mapping_buf);
    }

    return s;
}

int op_matmul_qkv(struct htp_ops_context * octx) {
    struct htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    const struct htp_tensor * restrict src0 = octx->src[0]; // Wk
    const struct htp_tensor * restrict src1 = octx->src[1]; // x
    const struct htp_tensor * restrict src2 = octx->src[2]; // Wv
    const struct htp_tensor * restrict src3 = octx->src[3]; // Wq
    const struct htp_tensor * restrict dst_k = octx->dsts[0];
    const struct htp_tensor * restrict dst_v = octx->dsts[1];
    const struct htp_tensor * restrict dst_q = octx->dsts[2];

    bool is_repacked = (src0->type == HTP_TYPE_Q4_0 || src0->type == HTP_TYPE_Q4_1 ||
                        src0->type == HTP_TYPE_Q8_0 || src0->type == HTP_TYPE_IQ4_NL ||
                        src0->type == HTP_TYPE_MXFP4);

    struct htp_mm_context mmctx_struct = {0};
    struct htp_mm_context * mmctx = &mmctx_struct;
    mmctx->octx = octx;

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;

    const uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const uint32_t src1_nrows = src1->ne[1] * src1->ne[2] * src1->ne[3];

    // Compute src0_nrows_per_thread
    mmctx->src0_nrows_per_thread  = (src0_nrows + octx->n_threads - 1) / octx->n_threads;
    if (is_repacked) {
        mmctx->src0_nrows_per_thread = hex_round_up(mmctx->src0_nrows_per_thread, 32);
    } else {
        mmctx->src0_nrows_per_thread += (mmctx->src0_nrows_per_thread & 1); // round up to even
    }

    const size_t src0_row_size = src0->nb[1];
    const size_t src0_row_size_padded = hex_round_up(src0_row_size, 128);

    if (hvx_mm_init_vec_dot(mmctx, src0->type) != 0) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t qk = QK_Q8_0_TILED;
    const uint32_t nb = (src1->ne[0] + qk - 1) / qk;
    const uint32_t total_nb = src1_nrows * nb;

    worker_callback_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    if (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT) {
        n_quant_tasks = MIN(src1_nrows, octx->n_threads);
        quant_task_func = (src0->type == HTP_TYPE_Q4_1) ? quantize_f32_q8_1_flat : quantize_f32_q8_0_flat;
    } else if (src1_nrows < octx->n_threads) {
        n_quant_tasks = MIN(total_nb, octx->n_threads);
        quant_task_func = (src0->type == HTP_TYPE_Q4_1) ? quantize_f32_q8_1_tiled_block : quantize_f32_q8_0_tiled_block;
        for (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_first = (total_nb * ith) / n_quant_tasks;
            uint32_t ib_last  = (total_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_first[ith] = ib_first;
            mmctx->quant_ib_last[ith]  = ib_last;
            mmctx->quant_r[ith]        = ib_first / nb;
            mmctx->quant_c[ith]        = ib_first % nb;
        }
    } else {
        n_quant_tasks = MIN(src1_nrows, octx->n_threads);
        quant_task_func = (src0->type == HTP_TYPE_Q4_1) ? quantize_f32_q8_1_tiled : quantize_f32_q8_0_tiled;
    }

    size_t src1_row_size;
    if (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT) {
        src1_row_size = (src0->type == HTP_TYPE_Q4_1) ? htp_mm_q8_1_flat_row_size(src1->ne[0]) : htp_mm_q8_0_flat_row_size(src1->ne[0]);
    } else {
        src1_row_size = (src0->type == HTP_TYPE_Q4_1) ? htp_mm_q8_1_tiled_row_size(src1->ne[0]) : htp_mm_q8_0_tiled_row_size(src1->ne[0]);
    }

    struct htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, src1->ne[0], src1_nrows, octx->n_threads,
                                 0, src0_row_size, src1_row_size, 0, kparams->n_prefetch, false, true, false);

    size_t vtcm_size = kparams->vtcm_size > 0 ? (size_t)kparams->vtcm_size : L.total_bytes;

    if (octx->ctx->vtcm_size < vtcm_size) {
        FARF(ERROR, "matmul-qkv: current VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_size, vtcm_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * const base = (uint8_t *) octx->ctx->vtcm_base;
    mmctx->vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src1);
    mmctx->vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
    mmctx->vtcm_src2 = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src2);
    mmctx->vtcm_src3 = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src3);
    mmctx->vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);

    octx->src1_spad.src  = NULL;
    octx->src0_spad.src  = NULL;
    octx->src2_spad.src  = NULL;
    octx->src3_spad.src  = NULL;
    octx->dst_spad.src   = NULL;

    mmctx->vtcm_src0_stride = is_repacked ? 0 : src0_row_size_padded;
    mmctx->vtcm_src2_stride = is_repacked ? 0 : src0_row_size_padded;
    mmctx->vtcm_src3_stride = is_repacked ? 0 : src0_row_size_padded;
    mmctx->vtcm_src1_stride = src1_row_size;

    mmctx->vtcm_src0_size_per_thread = L.src0_bytes / octx->n_threads;
    mmctx->vtcm_src1_size_per_thread = L.src1_bytes;
    mmctx->vtcm_src2_size_per_thread = L.src2_bytes / octx->n_threads;
    mmctx->vtcm_src3_size_per_thread = L.src3_bytes / octx->n_threads;
    mmctx->vtcm_dst_size_per_thread  = L.dst_bytes / octx->n_threads;

    mmctx->n_quant_rows_per_thread = (src1_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->quant_task_func = quant_task_func;
    mmctx->n_quant_tasks = n_quant_tasks;
    atomic_init(&mmctx->quant_barrier, n_quant_tasks);

    // Run fused matmul
    const uint32_t n_matmul_jobs = octx->n_threads;
    worker_callback_t matmul_job_func;
    if (is_repacked) {
        if (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT) {
            switch (src0->type) {
                case HTP_TYPE_Q4_0:   matmul_job_func = hvx_mm_qkv_2d_repacked_q4_0_flat;   break;
                case HTP_TYPE_Q4_1:   matmul_job_func = hvx_mm_qkv_2d_repacked_q4_1_flat;   break;
                case HTP_TYPE_Q8_0:   matmul_job_func = hvx_mm_qkv_2d_repacked_q8_0_flat;   break;
                case HTP_TYPE_IQ4_NL: matmul_job_func = hvx_mm_qkv_2d_repacked_iq4nl_flat;  break;
                case HTP_TYPE_MXFP4:  matmul_job_func = hvx_mm_qkv_2d_repacked_mxfp4_flat;  break;
                default:              return HTP_STATUS_NO_SUPPORT;
            }
        } else {
            switch (src0->type) {
                case HTP_TYPE_Q4_0:   matmul_job_func = hvx_mm_qkv_2d_repacked_q4_0;   break;
                case HTP_TYPE_Q4_1:   matmul_job_func = hvx_mm_qkv_2d_repacked_q4_1;   break;
                case HTP_TYPE_Q8_0:   matmul_job_func = hvx_mm_qkv_2d_repacked_q8_0;   break;
                case HTP_TYPE_IQ4_NL: matmul_job_func = hvx_mm_qkv_2d_repacked_iq4nl;  break;
                case HTP_TYPE_MXFP4:  matmul_job_func = hvx_mm_qkv_2d_repacked_mxfp4;  break;
                default:              return HTP_STATUS_NO_SUPPORT;
            }
        }
    } else {
        matmul_job_func = hvx_mm_qkv_2d;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    worker_pool_run_func(octx->ctx->worker_pool, matmul_job_func, mmctx, n_matmul_jobs);

    return HTP_STATUS_OK;
}

int op_matmul_ffn(struct htp_ops_context * octx) {
    struct htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    const struct htp_tensor * restrict src0 = octx->src[0]; // Wgate
    const struct htp_tensor * restrict src1 = octx->src[1]; // y
    const struct htp_tensor * restrict src2 = octx->src[2]; // Wup
    const struct htp_tensor * restrict dst_gate = octx->dsts[0];
    const struct htp_tensor * restrict dst_up = octx->dsts[1];

    bool is_repacked = (src0->type == HTP_TYPE_Q4_0 || src0->type == HTP_TYPE_Q4_1 ||
                        src0->type == HTP_TYPE_Q8_0 || src0->type == HTP_TYPE_IQ4_NL ||
                        src0->type == HTP_TYPE_MXFP4);

    struct htp_mm_context mmctx_struct = {0};
    struct htp_mm_context * mmctx = &mmctx_struct;
    mmctx->octx = octx;

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;

    const uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const uint32_t src1_nrows = src1->ne[1] * src1->ne[2] * src1->ne[3];

    // Compute src0_nrows_per_thread
    mmctx->src0_nrows_per_thread  = (src0_nrows + octx->n_threads - 1) / octx->n_threads;
    if (is_repacked) {
        mmctx->src0_nrows_per_thread = hex_round_up(mmctx->src0_nrows_per_thread, 32);
    } else {
        mmctx->src0_nrows_per_thread += (mmctx->src0_nrows_per_thread & 1); // round up to even
    }

    const size_t src0_row_size = src0->nb[1];
    const size_t src0_row_size_padded = hex_round_up(src0_row_size, 128);

    if (hvx_mm_init_vec_dot(mmctx, src0->type) != 0) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t qk = QK_Q8_0_TILED;
    const uint32_t nb = (src1->ne[0] + qk - 1) / qk;
    const uint32_t total_nb = src1_nrows * nb;

    worker_callback_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    if (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT) {
        n_quant_tasks = MIN(src1_nrows, octx->n_threads);
        quant_task_func = (src0->type == HTP_TYPE_Q4_1) ? quantize_f32_q8_1_flat : quantize_f32_q8_0_flat;
    } else if (src1_nrows < octx->n_threads) {
        n_quant_tasks = MIN(total_nb, octx->n_threads);
        quant_task_func = (src0->type == HTP_TYPE_Q4_1) ? quantize_f32_q8_1_tiled_block : quantize_f32_q8_0_tiled_block;
        for (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_first = (total_nb * (ith + 0)) / n_quant_tasks;
            uint32_t ib_last  = (total_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_first[ith] = ib_first;
            mmctx->quant_ib_last[ith]  = ib_last;
            mmctx->quant_r[ith]        = ib_first / nb;
            mmctx->quant_c[ith]        = ib_first % nb;
        }
    } else {
        n_quant_tasks = MIN(src1_nrows, octx->n_threads);
        quant_task_func = (src0->type == HTP_TYPE_Q4_1) ? quantize_f32_q8_1_tiled : quantize_f32_q8_0_tiled;
    }

    size_t src1_row_size;
    if (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT) {
        src1_row_size = (src0->type == HTP_TYPE_Q4_1) ? htp_mm_q8_1_flat_row_size(src1->ne[0]) : htp_mm_q8_0_flat_row_size(src1->ne[0]);
    } else {
        src1_row_size = (src0->type == HTP_TYPE_Q4_1) ? htp_mm_q8_1_tiled_row_size(src1->ne[0]) : htp_mm_q8_0_tiled_row_size(src1->ne[0]);
    }

    struct htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, src1->ne[0], src1_nrows, octx->n_threads,
                                 0, src0_row_size, src1_row_size, 0, kparams->n_prefetch, false, false, true);

    size_t vtcm_size = kparams->vtcm_size > 0 ? (size_t)kparams->vtcm_size : L.total_bytes;

    if (octx->ctx->vtcm_size < vtcm_size) {
        FARF(ERROR, "matmul-ffn: current VTCM reservation %zu is too small, needed %zu\n", octx->ctx->vtcm_size, vtcm_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * const base = (uint8_t *) octx->ctx->vtcm_base;
    mmctx->vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src1);
    mmctx->vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
    mmctx->vtcm_src2 = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src2);
    mmctx->vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);

    octx->src1_spad.src  = NULL;
    octx->src0_spad.src  = NULL;
    octx->src2_spad.src  = NULL;
    octx->dst_spad.src   = NULL;

    mmctx->vtcm_src0_stride = is_repacked ? 0 : src0_row_size_padded;
    mmctx->vtcm_src2_stride = is_repacked ? 0 : src0_row_size_padded;
    mmctx->vtcm_src1_stride = src1_row_size;

    mmctx->vtcm_src0_size_per_thread = L.src0_bytes / octx->n_threads;
    mmctx->vtcm_src1_size_per_thread = L.src1_bytes;
    mmctx->vtcm_src2_size_per_thread = L.src2_bytes / octx->n_threads;
    mmctx->vtcm_dst_size_per_thread  = L.dst_bytes / octx->n_threads;

    mmctx->n_quant_rows_per_thread = (src1_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->quant_task_func = quant_task_func;
    mmctx->n_quant_tasks = n_quant_tasks;
    atomic_init(&mmctx->quant_barrier, n_quant_tasks);

    // Run fused matmul
    const uint32_t n_matmul_jobs = octx->n_threads;
    worker_callback_t matmul_job_func;
    if (is_repacked) {
        if (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT) {
            switch (src0->type) {
                case HTP_TYPE_Q4_0:   matmul_job_func = hvx_mm_ffn_2d_repacked_q4_0_flat;   break;
                case HTP_TYPE_Q4_1:   matmul_job_func = hvx_mm_ffn_2d_repacked_q4_1_flat;   break;
                case HTP_TYPE_Q8_0:   matmul_job_func = hvx_mm_ffn_2d_repacked_q8_0_flat;   break;
                case HTP_TYPE_IQ4_NL: matmul_job_func = hvx_mm_ffn_2d_repacked_iq4nl_flat;  break;
                case HTP_TYPE_MXFP4:  matmul_job_func = hvx_mm_ffn_2d_repacked_mxfp4_flat;  break;
                default:              return HTP_STATUS_NO_SUPPORT;
            }
        } else {
            switch (src0->type) {
                case HTP_TYPE_Q4_0:   matmul_job_func = hvx_mm_ffn_2d_repacked_q4_0;   break;
                case HTP_TYPE_Q4_1:   matmul_job_func = hvx_mm_ffn_2d_repacked_q4_1;   break;
                case HTP_TYPE_Q8_0:   matmul_job_func = hvx_mm_ffn_2d_repacked_q8_0;   break;
                case HTP_TYPE_IQ4_NL: matmul_job_func = hvx_mm_ffn_2d_repacked_iq4nl;  break;
                case HTP_TYPE_MXFP4:  matmul_job_func = hvx_mm_ffn_2d_repacked_mxfp4;  break;
                default:              return HTP_STATUS_NO_SUPPORT;
            }
        }
    } else {
        matmul_job_func = hvx_mm_ffn_2d;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    worker_pool_run_func(octx->ctx->worker_pool, matmul_job_func, mmctx, n_matmul_jobs);

    return HTP_STATUS_OK;
}
