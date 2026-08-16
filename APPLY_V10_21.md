# Hexagon HMX MMID v10.22 phone overlay

This overlay is for `whikao/llama.cpp`. Extract it from the repository root.
The apply-note filename is retained so extraction cleanly updates the tracked
v10.21 note instead of leaving a second stale file.

## What changes

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
tar -xzf /sdcard/Download/llama-hexagon-hmx-mmid-v10.22.tar.gz -C .
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

The helper prints the exact `share-this.tar.gz` path at the end. Attach that
small archive to the debugging chat. If an HVX-only comparison is requested,
run:

```bash
PROMPT='你好。' NHMX=0 "$HOME/phone-debug.sh" run htp
```
