# Hexagon HMX raw-Q4_0 MMID v10.25 phone overlay

This overlay is for `whikao/llama.cpp`. Extract it from the repository root.
The apply-note filename is retained so extraction cleanly updates the tracked
v10.21 note instead of leaving another stale file.

## What changes

- Adds one `DBG_V125_HMX_RAW_PROFILE` record per raw-HMX `MUL_MAT_ID`
  invocation. It reports accumulated microseconds for activation gather, raw
  DMA, parallel raw-to-tiled conversion, tiled dequantization, HMX compute and
  output scatter across all selected experts. The instrumentation does not
  change tensor data or kernel selection.
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
tar -xzf /sdcard/Download/llama-hexagon-hmx-raw-mmid-v10.25.tar.gz -C .
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

Run the HMX regression case:

```bash
termux-wake-lock
PROMPT='你好。' "$HOME/phone-debug.sh" run htp
```

Leave `GGML_HEXAGON_MMID_RAW_Q4_0=1` enabled. Setting it to `0` selects the
model-sized HTP_REPACK placement and can be killed by Android on this phone.

The helper prints the exact `share-this.tar.gz` path at the end. Attach that
small archive to the debugging chat. If an HVX-only comparison is requested,
run:

```bash
PROMPT='你好。' NHMX=0 "$HOME/phone-debug.sh" run htp
```
