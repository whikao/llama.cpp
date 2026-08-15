#ifndef HTP_OPS_H
#define HTP_OPS_H

#include <assert.h>

// ggml-common.h must be included prio to this header

enum htp_status {
    HTP_STATUS_OK             = 1,
    HTP_STATUS_INTERNAL_ERR   = 2,
    HTP_STATUS_NO_SUPPORT     = 3,
    HTP_STATUS_INVAL_PARAMS   = 4,
    HTP_STATUS_VTCM_TOO_SMALL = 5,
};

// First set of values must match the ggml_type.
// Duplicated here because we can't include full ggml.h in the htp build.
// We have some static_asserts in the cpp code to ensure things are in sync.
enum htp_data_type {
    HTP_TYPE_F32    = 0,
    HTP_TYPE_F16    = 1,
    HTP_TYPE_Q4_0   = 2,
    HTP_TYPE_Q4_1   = 3,
    HTP_TYPE_Q8_0   = 8,
    HTP_TYPE_IQ4_NL = 20,
    HTP_TYPE_I32    = 26,
    HTP_TYPE_I64    = 27,
    HTP_TYPE_MXFP4  = 39,

    // types used internally for repack, dyn.quant, etc
    HTP_TYPE_Q4_0_TILED = 200,
    HTP_TYPE_Q4_1_TILED,
    HTP_TYPE_Q8_0_TILED,
    HTP_TYPE_MXFP4_TILED,

    HTP_TYPE_INVALID
};

// Constats for internal types
#define QK_Q4_0_TILED  256  // 32x32 Q4_0 tiled layout
#define QK_Q8_0_TILED  128  // 32x32 Q8_0 tiled layout
#define QK_MXFP4_TILED 256  // 32x32 MXFP4 tiled layout



// Mask to enable various stages of the Ops.
// Used for debugging and profiling.
enum htp_op_stage {
    HTP_OPSTAGE_QUEUE    = (1 << 0),  // Enable Queueing (ie calls into NPU)
    HTP_OPSTAGE_COMPUTE  = (1 << 1),  // Enable Compute
};

// Do not reorder first 4 (used as an index)
enum htp_op_code {
    HTP_OP_MUL = 0,
    HTP_OP_ADD = 1,
    HTP_OP_SUB = 2,
    HTP_OP_DIV = 3,
    HTP_OP_MUL_MAT,
    HTP_OP_MUL_MAT_ID,
    HTP_OP_MUL_MAT_QKV,
    HTP_OP_MUL_MAT_FFN,
    HTP_OP_MUL_MAT_ADD,
    HTP_OP_RMS_NORM,
    HTP_OP_RMS_NORM_MUL,
    HTP_OP_UNARY_SILU,
    HTP_OP_UNARY_GELU,
    HTP_OP_UNARY_SIGMOID,
    HTP_OP_UNARY_EXP,
    HTP_OP_UNARY_NEG,
    HTP_OP_UNARY_SOFTPLUS,
    HTP_OP_UNARY_TANH,
    HTP_OP_GLU_SWIGLU,
    HTP_OP_GLU_SWIGLU_OAI,
    HTP_OP_GLU_GEGLU,
    HTP_OP_SOFTMAX,
    HTP_OP_ADD_ID,
    HTP_OP_ROPE,
    HTP_OP_FLASH_ATTN_EXT,
    HTP_OP_SET_ROWS,
    HTP_OP_GET_ROWS,
    HTP_OP_SCALE,
    HTP_OP_CPY,
    HTP_OP_ARGSORT,
    HTP_OP_SQR,
    HTP_OP_SQRT,
    HTP_OP_SUM_ROWS,
    HTP_OP_SSM_CONV,
    HTP_OP_REPEAT,
    HTP_OP_CUMSUM,
    HTP_OP_FILL,
    HTP_OP_DIAG,
    HTP_OP_SOLVE_TRI,
    HTP_OP_L2_NORM,
    HTP_OP_GATED_DELTA_NET,
    HTP_OP_TRI,
    HTP_OP_PAD,
    HTP_OP_NORM,
    HTP_OP_CONCAT,
    HTP_OP_CLAMP,
    HTP_OP_IM2COL,

    HTP_OP_INVALID
};

#define HTP_OP_MAX_DIMS    4    // aka GGML_MAX_DIMS
#define HTP_OP_MAX_INPUTS  6    // aka GGML_MAX_SRCS
#define HTP_OP_MAX_OUTPUTS 4
#define HTP_OP_MAX_PARAMS  16   // aka GGML_MAX_OP_PARAMS
// v10.12: preserve the original first 32 words and reserve 8 extra
// Host -> DSP runtime MMID debug-control words.
#define HTP_OP_MAX_KERN_PARAMS 176

#define HTP_MM_DEBUG_CTRL_MAGIC 0x44423132u /* "DB12" */
#define HTP_MM_DEBUG_CTRL_WORD_MAGIC   32
#define HTP_MM_DEBUG_CTRL_WORD_EXPERT  33
#define HTP_MM_DEBUG_CTRL_WORD_CT      34
#define HTP_MM_DEBUG_CTRL_WORD_CID     35
#define HTP_MM_DEBUG_CTRL_WORD_K       36
#define HTP_MM_DEBUG_CTRL_WORD_FLAGS   37


#define HTP_OP_MAX_BUFS    16
#define HTP_OP_MAX_TENSORS 8192 // must stay under 64K (uint16)

#define HTP_OP_MAX_VMEM_DEFAULT (3355443200u)

#define HTP_MMAP_MAX_VMEM  (2147483648u)

enum htp_tensor_flags {
    HTP_TENSOR_COMPUTE = (1U << 0), // Tensor buffer temporal compute data (not weights)
    HTP_TENSOR_DIRTY   = (1U << 1)  // Tensor buffer is dirty and needs to be flushed
};

// Tensor descriptor
struct htp_tensor {
    uint32_t data;                 // Buffer offset in the messages, and data pointer on the NPU
    uint32_t reserved;             // Reserved for alignment padding (must be multiple of 8)
    uint32_t size;                 // Data size in bytes
    uint32_t flags;                // Buffer / tensor flags
    uint32_t type;                 // Data type
    uint16_t bi;                   // Buffer index
    uint16_t ti;                   // Tensor index
    uint32_t ne[HTP_OP_MAX_DIMS];  // Number of elements
    uint32_t nb[HTP_OP_MAX_DIMS];  // Stride in bytes (see ggml.h ggml_tensor)
};

// Buffer descriptor
struct htp_buf_desc {
    uint64_t base;     // base address
    uint64_t size;     // total size
    uint32_t flags;    // buffer flags (unused)
    uint32_t fd;       // file descriptor
};

enum htp_op_flags {
    HTP_OPFLAGS_SKIP_COMPUTE  = (1U << 0), // Skip actual computation (used for profiling)
};

// Op descriptor
struct htp_op_desc {
    uint32_t opcode;                    // GGML/HTP Op
    uint32_t flags;                     // Op flags
    int32_t  params[HTP_OP_MAX_PARAMS]; // Params for the op, e.g. epsilon of RMS norm
    int32_t  kernel_params[HTP_OP_MAX_KERN_PARAMS]; // generic blob for host-precomputed parameters
    uint16_t src[HTP_OP_MAX_INPUTS];    // Input tensors indices
    uint16_t dst[HTP_OP_MAX_OUTPUTS];   // Output tensor indices
    uint16_t pad[2];                    // padding to align to 64 bits
};

#ifndef HTP_MAX_NTHREADS
#define HTP_MAX_NTHREADS 10
#endif

#define HTP_TRACE_MAX_EVENTS 256

enum htp_profiler_mode {
    HTP_PROF_DISABLED = 0,
    HTP_PROF_BASIC    = 1,
    HTP_PROF_PMU      = 2,
    HTP_PROF_TRACE    = 3,
};

enum htp_trace_event_id {
    HTP_TRACE_EVT_DMA                 = 0,
    HTP_TRACE_EVT_L2FLUSH             = 1,
    HTP_TRACE_EVT_INIT                = 2,
    HTP_TRACE_EVT_BUFF                = 3,

    HTP_TRACE_EVT_HVX_COMP            = 20,
    HTP_TRACE_EVT_HVX_A_QUANT         = 21,
    HTP_TRACE_EVT_HVX_A_PREP          = 22,
    HTP_TRACE_EVT_HVX_W_DEQUANT       = 23,
    HTP_TRACE_EVT_HVX_W_PREP          = 24,
    HTP_TRACE_EVT_HVX_O_PROC          = 25,
    HTP_TRACE_EVT_HVX_FA_QK           = 26,
    HTP_TRACE_EVT_HVX_FA_SFM          = 27,
    HTP_TRACE_EVT_HVX_FA_Q_PREP       = 28,
    HTP_TRACE_EVT_HVX_FA_K_PREP       = 29,
    HTP_TRACE_EVT_HVX_FA_V_PREP       = 30,

    HTP_TRACE_EVT_HMX_COMP            = 40,
};

struct htp_trace_desc {
    uint32_t cycles;  // lower 32-bits of cycle counter
    uint16_t id;      // Event ID
    uint16_t info;    // bit 15: is_stop. bits 14-0: tile/chunk index or other metadata.
};

#define HTP_PROF_PMU_NCNT 8

// Profile descriptor
struct htp_prof_desc {
    uint32_t opcode;                 // GGML/HTP Op
    uint32_t usecs;                  // Number of usec
    uint32_t cycles_start;           // Start cycle counter
    uint32_t cycles_stop;            // Stop cycle counter
    uint32_t pmu[HTP_PROF_PMU_NCNT]; // PMU counters
};

struct htp_opbatch_req {
    uint32_t id;          // Batch id
    uint32_t n_bufs;      // Number of buffers
    uint32_t n_tensors;   // Number of tensors
    uint32_t n_ops;       // Number of ops
    uint32_t n_traces;    // Number of trace descriptors per thread
    uint32_t pad;         // unused
    // struct htp_buf_desc  bufs[];    -- dspqueue buf 0
    // struct htp_tensor    tensors[]; -- dspqueue buf 0
    // struct htp_op_desc   ops[];     -- dspqueue buf 0
};

#define HTP_OPBATCH_DEBUG_RAW_Q4_0_MAGIC 0x51343034u  /* "Q404": v10.4 raw-Q4_0 debug response */

struct htp_opbatch_rsp {
    uint32_t id;         // Batch id
    uint32_t status;     // HTP_STATUS_...
    uint32_t n_bufs;     // Number of buffers
    uint32_t n_tensors;  // Number of tensors
    uint32_t n_ops;      // Number of op profile descriptors
    uint32_t n_traces[HTP_MAX_NTHREADS + 1];
    uint32_t usecs;          // Number of usec
    uint32_t pad;            // align to 8 bytes
    uint64_t cycles_start;   // Start cycle counter
    uint64_t cycles_stop;    // Stop cycle counter

    // v10.4: explicit DSP -> Host raw-Q4_0 correctness return.
    // This structure itself is transported by dspqueue_write(), whose buffer flags
    // already flush the DSP sender and invalidate the CPU recipient.
    uint32_t dbg_magic;      // HTP_OPBATCH_DEBUG_RAW_Q4_0_MAGIC when valid
    uint32_t dbg_op_index;   // op index within this batch
    uint32_t dbg_expert;     // selected expert id captured by DSP
    uint32_t dbg_src_off;    // expert * nb02 used by DSP
    uint32_t dbg_raw_fnv;    // FNV-1a of raw selected-expert working set
    uint32_t dbg_tile_fnv;   // FNV-1a of converted tiled Q4_0 tile
    uint32_t dbg_ct;         // exact 32-row tile index hashed by DSP
    uint32_t dbg_ith;        // DSP worker thread that processed dbg_ct
    uint32_t dbg_valid_rows; // valid rows in dbg_ct

    // v10.7: exact dot-call parameters for the captured expert/ct/cid=0.
    uint32_t dbg_ne00;
    uint32_t dbg_ne10;
    uint32_t dbg_nb01;
    uint32_t dbg_nb02;
    uint32_t dbg_src1_stride;
    uint32_t dbg_rm1;
    uint32_t dbg_rm2;
    uint32_t dbg_ir1;
    uint32_t dbg_src1_off;

    // v10.8.1: post-dot float bit patterns returned through dspqueue response.
    uint32_t dbg_out0;
    uint32_t dbg_out1;
    uint32_t dbg_out2;
    uint32_t dbg_out3;

    // v10.9: src1/Q8 activation diagnostics.
    uint32_t dbg_q8_quant_fnv;
    uint32_t dbg_q8_dot_fnv;
    uint32_t dbg_q8_head;
    uint32_t dbg_q8_scale_head;
    uint32_t dbg_src_f0_bits;
    uint32_t dbg_src_max_bits;
    uint32_t dbg_q8_row_fnv;

    // v10.11: first K=32 row-0 kernel/reference comparison.
    uint32_t dbg_hvx_k32_bits;
    uint32_t dbg_ref_k32_bits;
    int32_t  dbg_k32_int_dot;
    uint32_t dbg_k32_scales_fp16;

    // v10.12 runtime-selected cumulative-K checkpoint.
    uint32_t dbg_runtime_k;

    // v10.14: row-0 integer accumulator produced by the exact HVX
    // accum_4bit_32x1() path for the selected single tile.
    int32_t dbg_hvx_int_dot;

    // v10.15: element-wise selected-tile diagnostics.
    int8_t  dbg_q8_actual[32];
    int8_t  dbg_q8_scalar[32];
    int8_t  dbg_q4_weight[32];
    int16_t dbg_prod_delta[32];

    // struct htp_prof_desc profs[];  -- dspqueue buf 0
};

#endif /* HTP_OPS_H */
