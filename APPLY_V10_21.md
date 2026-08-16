# Hexagon MMID v10.21 phone overlay

This overlay is for `whikao/llama.cpp`. Extract it from the repository root.

## What changes

- Adds `DBG_V121_MMID_MAP`, `DBG_V121_MMID_Q8`, and
  `DBG_V121_MMID_Q4_DOT` records for each traced MMID slice.
- Records the logical and actual quantizer source offsets, Q8 VTCM row,
  selected expert/mapping, raw and tiled Q4 working sets, actual dot inputs,
  and the first four dot outputs.
- Adds `GGML_HEXAGON_MMID_QUANT_MODE=auto|block|row` so the row and block
  activation quantizers can be compared without rebuilding.
- Keeps the Termux backend-discovery and `--single-turn` fixes.

## Apply in Termux

```bash
cd "$HOME/whikao-llama.cpp"
tar -xzf /sdcard/Download/llama-hexagon-mmid-v10.21.tar.gz -C .
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

Run the most useful first A/B case (force the block quantizer even when
`ne2=10`):

```bash
termux-wake-lock
PROMPT='你好。' MMID_QUANT_MODE=block \
  "$HOME/phone-debug.sh" run htp
```

The helper prints the exact `share-this.tar.gz` path at the end. Attach that
small archive to the debugging chat. If a row-mode comparison is requested,
run:

```bash
PROMPT='你好。' MMID_QUANT_MODE=row \
  "$HOME/phone-debug.sh" run htp
```
