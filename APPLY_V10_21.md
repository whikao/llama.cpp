# Hexagon HMX raw-Q4_0 MMID v10.33 phone overlay

This overlay is for `whikao/llama.cpp`. Extract it from the repository root.
The apply-note filename is retained so extraction cleanly updates the tracked
v10.21 note instead of leaving another stale file.

## What changes

- v10.33 targets the bottleneck measured by v10.32 without changing DSP math.
  The phone copied 4,069.39 MiB of selected raw Q4_0 experts in 3.405688
  seconds across 3,453 `SET` calls (about 71.4% of measured host time), while
  all explicit synchronizations together used only 0.071097 seconds.
- The Hexagon backend now implements `set_tensor_async` only as an optimization
  for normal-buffer Q4_0 tensors whose name contains `_exps.weight`. Those
  independent sparse ranges are copied by a persistent host worker pool, then
  fenced before graph submission, explicit synchronization, and session
  destruction. All other tensor uploads retain the previous synchronous path.
- `GGML_HEXAGON_HOST_COPY_THREADS` controls the pool. The helper exposes it as
  `HOST_COPY_THREADS`, defaults to `4`, caps it at `8`, and records it in
  metadata. Values `0` and `1` select the original synchronous behavior for a
  clean A/B fallback. Worker-creation failure also falls back synchronously.
- `DBG_V133_HOST_COPY_CONFIG` records the active pool and
  `DBG_V133_HOST_COPY_BATCH` records jobs, bytes and wall time at each graph
  fence. This is a host-copy experiment; v10.31 DSP conversion and HMX output
  equations remain byte-for-byte unchanged.
- v10.32 is a host-side timing build. It leaves the verified v10.31 DSP math
  and memory layout unchanged, and records elapsed microseconds for backend
  tensor uploads/downloads, graph execution and explicit synchronization as
  `DBG_V132_HOST_*` lines. These records isolate the roughly 5.2 prompt seconds
  now outside the profiled MMID kernel before the next performance change.
- The cooled v10.31 repeat confirmed the optimization: it generated `你好`, all
  28 slices remained finite, and the full reference difference stayed
  `-5.36e-09`. Prompt evaluation was 5.827 seconds. Across the same 135 records
  and 4,797 experts, raw-to-tiled was 0.416591 of 0.629966 profiled MMID
  seconds. Compared with v10.30, conversion is 8.02x faster and prompt
  evaluation is 1.45x faster. The earlier 8.822-second v10.31 run had almost
  identical 0.416621-second conversion time, confirming that its extra delay
  was outside this DSP phase.
- v10.31 replaces the last full-tile strided scalar loop: the 32 fp16 Q4_0
  scales are gathered as low halfwords of 32 words, compacted in row order by
  one halfword `vdeal`, and written with one 64-byte predicated vector store.
  The partial-row fallback remains unchanged.
- v10.30 is the fastest correct phone result so far. It generated `你好`, all
  28 slices stayed finite, and its first full reference difference remained
  `-5.36e-09`. Prompt evaluation fell from v10.28's 14.036 seconds to 8.417
  seconds (1.67x faster), while raw-to-tiled fell from 8.323 to 3.342 seconds
  (59.8%). Conversion still represented 93.72% of profiled MMID time, so the
  remaining scalar scale copies are the v10.31 target.
- v10.30 removes the gather post-processing scalar row loop. The verified
  v10.28 gathered words remain in HVX registers; word shifts plus masks form
  the same four Q4_0 packed-byte equations, and two byte `vdeal` operations
  compact byte 0 of all 32 words into row order. Each result is written with
  one 32-byte predicated vector store. Gather addressing and memory use are
  unchanged.
- v10.29 remained correct but is rejected for performance. It generated `你好`
  with all 28 slices finite, yet prompt evaluation rose from 14.036 to 25.163
  seconds. Across the same 135 records and 4,797 experts, raw-to-tiled rose
  from 8.323 to 18.026 seconds. The four-row scalar packing and full unroll
  increased instruction/register pressure instead of improving the converter.
- The v10.28 phone run is correct and fast: exit code `0`, generated token
  `你好`, all 28 diagnostic slices finite, and the first full reference differs
  by only `-5.36e-09`. Prompt evaluation fell from the correct v10.26 result
  of 43.044 seconds to 14.036 seconds (3.07x faster). Across 135 profile
  records and 4,797 selected-expert calls, raw-to-tiled conversion still used
  8.323 of 8.605 MMID seconds (96.723%), which remains the v10.30 target.
- v10.28 fixes the v10.27 gather address observed on the phone. Passing
  `raw + 2` as the word-gather base caused the target to round the base down
  and gather Q4_0 scale bytes: the first packed byte became `0xd4` from scale
  bytes `0x54,0x1d`, instead of reference `0x96` from Q bytes `0x06,0x89`.
  The gather base now remains aligned at `raw`; the per-group Q offset is added
  to every vector offset, where unaligned final element addresses are valid.
- The rejected v10.27 result completed in 12.861 seconds but emitted a newline
  token instead of `你好`. Finite-slice checks remained clean, demonstrating
  why the token/reference checks are required in addition to NaN detection.
- v10.27.1 makes phone installation resumable on unstable GitHub connections.
  The helper downloads `SHA256SUMS` first, names a persistent private cache
  entry by the expected digest, resumes the large asset with `curl -C -`, and
  switches `current` only after the complete SHA256 and archive layout pass.
  An interrupted download no longer disappears with the temporary directory.
- v10.27 replaces the remaining full-tile strided scalar Q-byte reads with
  four word-granularity HVX gathers per 32-row K tile. Gather results use the
  tile's 128-byte scale/alignment area as temporary VTCM storage, so memory use
  is unchanged. Partial-row tiles retain the byte-equivalent v10.26 fallback.
- v10.26 reduced prompt evaluation from 54.354 to 43.044 seconds (20.8%) and
  raw-to-tiled time from 42.784 to 35.191 seconds (17.7%). The result remained
  correct, but conversion still represented 99.196% of profiled MMID time;
  v10.27 targets the strided loads left inside that phase.
- v10.26 optimizes the measured bottleneck without changing the Q4_0 bytes
  consumed by the existing dequantizer. The raw-to-tiled converter now walks
  each raw row once, reuses each adjacent Q-byte pair for both K halves, writes
  two output rows per halfword, and avoids clearing the 576 logical tile bytes
  that a full tile immediately overwrites. The old and new transforms match
  byte-for-byte across full and partial-row tests.
- The v10.25.2 phone profile contained 135 raw-HMX MMID records and 4,797
  selected-expert calls. Of 43.062 seconds measured inside MMID, raw-to-tiled
  conversion consumed 42.784 seconds (99.354%). Raw DMA was 0.333%, tiled
  dequantization 0.165%, and HMX compute only 0.035%. The v10.26 change is
  deliberately limited to that 99.354% phase.
- Adds one `DBG_V125_HMX_RAW_PROFILE` record per raw-HMX `MUL_MAT_ID`
  invocation. It reports accumulated microseconds for activation gather, raw
  DMA, parallel raw-to-tiled conversion, tiled dequantization, HMX compute and
  output scatter across all selected experts. The instrumentation does not
  change tensor data or kernel selection.
- v10.25.1 emits that record at the DSP `ALWAYS` level. The first v10.25
  build used `HIGH`, which is filtered by the release DSP logging mask and
  therefore produced no profile records even though inference succeeded.
- v10.25.2 transports the phase totals in the coherent `dspqueue` response and
  prints them on the Android host. The phone does not route DSP FARF output to
  `llama-cli` stdout even at `ALWAYS`; the response path is the same verified
  channel already used by the v10.20/v10.21 slice records.
- Parallelizes the low-memory raw-Q4_0 to tiled transform added in v10.23.
  Independent 32-row weight tiles are now distributed across the existing HTP
  worker pool before the already-threaded dequantizer runs. The byte layout,
  640-byte aligned K-tile stride, HMX input and memory footprint are unchanged.
- Targets the v10.23 performance regression where the correct raw-HMX path
  needed about 461.7 seconds for 10 prompt tokens, versus about 51.6 seconds
  for the HVX control. This is a performance change; the phone result is still
  required to measure the actual gain.
- Fixes the remaining HMX `MUL_MAT_ID` format mismatch exposed by the v10.22
  HMX/HVX comparison. Low-memory mode stages expert weights in original GGUF
  Q4_0 row layout, while the HMX dequantizer previously read those bytes as
  already-tiled weights.
- Keeps the model in host/raw storage. For each HMX output-column chunk, raw
  rows are DMA-copied into the existing FP16 scratch area, converted into the
  existing tiled-weight area, then dequantized back into the reused FP16
  scratch area. It does not create the roughly 15 GiB model-sized HTP_REPACK
  allocation and does not add another VTCM reservation.
- Fixes the HMX `MUL_MAT_ID` gathered/scattered row workers used by the
  `ne2=10` prompt batch. Padded 32-row chunks were split between eight workers
  even when an expert had only one or two valid rows. Unsigned subtraction
  underflowed for later workers and overlapping padded writes consumed invalid
  expert mappings.
- Uses exactly one gathered/scattered call per HMX VTCM tile chunk, bounds the
  work to the expert's valid row count, and guards every worker against a start
  row beyond `cne1`.
- Makes the Termux helper match the successful direct command: no timeout by
  default, no 32-bit vendor path, no inherited ADSP path, and no global MMID
  control packet unless requested.
- Adds `NHMX=0|1` to the helper for an explicit HVX-only comparison.
- Retains the v10.21 `DBG_V121_MMID_*` records and the HVX
  `GGML_HEXAGON_MMID_QUANT_MODE=auto|block|row` diagnostic selector.

## Apply in Termux

```bash
cd "$HOME/whikao-llama.cpp"
tar -xzf /sdcard/Download/llama-hexagon-hmx-raw-mmid-v10.33.tar.gz -C .
git diff --check
git status --short
```

Inspect and publish the change using your normal Git workflow, then start the
existing Android Hexagon workflow. After the workflow succeeds, refresh the
phone helper and install the new rolling release:

```bash
install -m 700 \
  "$HOME/whikao-llama.cpp/scripts/snapdragon/termux/phone-debug.sh" \
  "$HOME/phone-debug.sh"

"$HOME/phone-debug.sh" install
```

Run one correctness and throughput case:

```bash
termux-wake-lock
PROMPT='请用一句话介绍你自己。' TOKENS=32 HOST_COPY_THREADS=4 \
  "$HOME/phone-debug.sh" run htp
```

If the parallel result is wrong or slower, the exact synchronous control is:

```bash
PROMPT='请用一句话介绍你自己。' TOKENS=32 HOST_COPY_THREADS=1 \
  "$HOME/phone-debug.sh" run htp
```

Leave `GGML_HEXAGON_MMID_RAW_Q4_0=1` enabled. Setting it to `0` selects the
model-sized HTP_REPACK placement and can be killed by Android on this phone.

The helper prints the exact `share-this.tar.gz` path at the end. Attach that
small archive to the debugging chat. If an HVX-only comparison is requested,
run:

```bash
PROMPT='你好。' NHMX=0 "$HOME/phone-debug.sh" run htp
```
