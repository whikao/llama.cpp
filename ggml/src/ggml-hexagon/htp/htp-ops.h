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

// Runtime MMID diagnostic flags.  Bit 0 preserves the original v10.12
// enable marker; bits 1/2 select one activation-quantization implementation
// for an A/B run without requiring another rebuild.
#define HTP_MM_DEBUG_FLAG_ENABLED           (1u << 0)
#define HTP_MM_DEBUG_FLAG_FORCE_QUANT_BLOCK (1u << 1)
#define HTP_MM_DEBUG_FLAG_FORCE_QUANT_ROW   (1u << 2)

enum htp_mmid_quant_mode {
    HTP_MMID_QUANT_AUTO  = 0,
    HTP_MMID_QUANT_BLOCK = 1,
    HTP_MMID_QUANT_ROW   = 2,
};

// v10.19 generic post-op trace controls.
// Host marks a target op and the following N ops; DSP snapshots dst0 after each.
#define HTP_POSTOP_TRACE_MAGIC          0x54523139u /* "TR19" */
#define HTP_POSTOP_TRACE_WORD_MAGIC     120
#define HTP_POSTOP_TRACE_WORD_ORDINAL   121
#define HTP_POSTOP_TRACE_MAX_RECORDS    8

// v10.20: detailed MUL_MAT_ID slice diagnostics for ne2/batch problems.
#define HTP_MMID_SLICE_TRACE_MAX        10


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

struct htp_postop_trace_record {
    uint32_t valid;
    uint32_t ordinal;
    uint32_t op_index;
    uint32_t opcode;
    uint32_t dst_index;
    uint32_t type;
    uint32_t size;
    uint32_t hash_bytes;
    uint32_t fnv;
    uint32_t word0;
    uint32_t word1;
    uint32_t word2;
    uint32_t word3;
    uint32_t ne0;
    uint32_t ne1;
    uint32_t ne2;
    uint32_t ne3;
};

struct htp_mmid_slice_trace_record {
    uint32_t valid;
    uint32_t slice;

    uint32_t src1_ne0, src1_ne1, src1_ne2, src1_ne3;
    uint32_t src1_nb0, src1_nb1, src1_nb2, src1_nb3;
    uint32_t src1_hash, src1_hash_bytes;
    uint32_t src1_word0, src1_word1, src1_word2, src1_word3;
    uint32_t src1_nonfinite, src1_maxabs_bits;

    uint32_t ids_ne0, ids_ne1, ids_nb0, ids_nb1;
    int32_t ids0, ids1, ids2, ids3, ids4, ids5, ids6, ids7;

    uint32_t dst_ne0, dst_ne1, dst_ne2, dst_ne3;
    uint32_t dst_nb0, dst_nb1, dst_nb2, dst_nb3;
    uint32_t dst_hash, dst_hash_bytes;
    uint32_t dst_word0, dst_word1, dst_word2, dst_word3;
    uint32_t dst_nonfinite, dst_maxabs_bits;

    // v10.21: values captured inside the raw-Q4_0 MMID worker at the exact
    // first dot call selected for this slice.  All pointer values are offsets
    // from a stable tensor/VTCM base, never process-specific absolute values.
    uint32_t internal_valid;
    uint32_t quant_mode;
    uint32_t n_threads;
    uint32_t n_quant_tasks;
    uint32_t src1_nrows;
    uint32_t quant_rows_per_thread;

    uint32_t expert;
    uint32_t cid;
    uint32_t ct;
    uint32_t ith;
    uint32_t rm1;
    uint32_t rm2;
    uint32_t ir1;
    uint32_t q8_row;
    uint32_t valid_rows;
    uint32_t mapping_stride;
    uint32_t expert_mapping_count;

    uint32_t src_orig_off;
    uint32_t src_orig_hash;
    uint32_t src_orig_hash_bytes;
    uint32_t src_quant_off;
    uint32_t src_quant_hash;
    uint32_t src_orig_word0, src_orig_word1, src_orig_word2, src_orig_word3;

    uint32_t q8_vtcm_off;
    uint32_t q8_stride;
    uint32_t q8_hash;
    uint32_t q8_hash_bytes;
    uint32_t q8_word0, q8_word1, q8_word2, q8_word3;
    uint32_t q8_scale0, q8_scale1;

    uint32_t q4_model_off;
    uint32_t q4_raw_vtcm_off;
    uint32_t q4_raw_hash;
    uint32_t q4_raw_hash_bytes;
    uint32_t q4_raw_word0;
    uint32_t q4_raw_scale0;

    uint32_t q4_tiled_vtcm_off;
    uint32_t q4_tiled_hash;
    uint32_t q4_tiled_hash_bytes;
    uint32_t q4_tiled_word0;
    uint32_t q4_tiled_scale0;

    uint32_t dst_off;
    uint32_t dot_out0, dot_out1, dot_out2, dot_out3;
};

// v10.25.2: DSP -> host phase totals for low-memory raw-Q4_0 HMX MMID.
// Times are converted to microseconds on DSP before transport.
struct htp_hmx_raw_profile_record {
    uint32_t valid;
    uint32_t op_count;
    uint32_t expert_calls;
    uint32_t total_us;
    uint32_t activation_us;
    uint32_t raw_dma_us;
    uint32_t raw_to_tiled_us;
    uint32_t dequant_us;
    uint32_t hmx_us;
    uint32_t output_us;
};

// v10.37: DSP -> host phase totals for the single-token raw-Q4_0 HVX
// ffn_down_exps path.  Worker and phase values are sums across DSP workers;
// wall_us is the longest worker duration accumulated once per profiled op.
struct htp_hvx_raw_decode_profile_record {
    uint32_t valid;
    uint32_t op_count;
    uint32_t expert_calls;
    uint32_t tile_count;
    uint32_t ne00;
    uint32_t ne01;
    uint32_t worker_us;
    uint32_t wall_us;
    uint32_t raw_dma_us;
    uint32_t raw_to_tiled_us;
    uint32_t dot_us;
    uint32_t reserved;
};

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
    uint32_t dbg_full_ref_bits;

    // v10.15: element-wise selected-tile diagnostics.
    int8_t  dbg_q8_actual[32];
    int8_t  dbg_q8_scalar[32];
    int8_t  dbg_q4_weight[32];
    int16_t dbg_prod_delta[32];

    // v10.19: up to 8 consecutive post-op checkpoints.
    uint32_t dbg_trace_count;
    struct htp_postop_trace_record dbg_trace[HTP_POSTOP_TRACE_MAX_RECORDS];
    uint32_t dbg_mmid_slice_count;
    struct htp_mmid_slice_trace_record dbg_mmid_slice[HTP_MMID_SLICE_TRACE_MAX];
    struct htp_hmx_raw_profile_record dbg_hmx_raw_profile;
    struct htp_hvx_raw_decode_profile_record dbg_hvx_raw_decode_profile;

    // struct htp_prof_desc profs[];  -- dspqueue buf 0
};

#endif /* HTP_OPS_H */
